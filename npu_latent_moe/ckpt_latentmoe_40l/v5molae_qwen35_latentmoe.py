from __future__ import annotations

import inspect
import shutil
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeMLP,
        Qwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeExperts,
        Qwen3_5MoeTopKRouter,
        Qwen3_5MoeRMSNorm,
    )
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
except Exception:
    Qwen3_5MoeForCausalLM = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeMLP = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeSparseMoeBlock = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeExperts = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeTopKRouter = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeRMSNorm = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeConfig = object  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# NPU fused-op helpers (ported from MindSpeed-MM qwen3_5_moe modeling)
# -----------------------------------------------------------------------------
# Detect Ascend NPU + torch_npu at import time. This mirrors MS-MM's
# `IS_NPU_AVAILABLE`. We cache the result to avoid repeated import attempts.
try:
    import torch_npu  # type: ignore  # noqa: F401
    _HAS_TORCH_NPU = hasattr(torch, "npu") and torch.npu.is_available()
except Exception:
    _HAS_TORCH_NPU = False


def _npu_group_gemm(x: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
    """Grouped matmul across experts using NPU fused kernel.

    x:          [S_total, in_dim]
    weight:     [E, in_dim, out_dim]
    group_list: [E]  (int, tokens per expert)
    returns:    [S_total, out_dim]

    Inlined copy of `mindspeed_mm.models.common.gmm.npu_group_gemm` so this file
    remains self-contained under `trust_remote_code=True`.
    """
    import torch_npu  # type: ignore
    return torch_npu.npu_grouped_matmul(
        [x], [weight], bias=None, group_list=group_list,
        split_item=2, group_type=0, group_list_type=1,
    )[0]


class Qwen3_5NpuRMSNorm(Qwen3_5MoeRMSNorm):
    """NPU-fused drop-in replacement for Qwen3_5MoeRMSNorm.

    Same parameters and arithmetic identity as the HF parent (weights are
    (1 + w) due to Qwen3.5's zero-centered init). Forward dispatches to
    `torch_npu.npu_rms_norm` when NPU is available, otherwise falls back to
    the parent's eager implementation.

    Used by `Qwen3_5LatentMoeForCausalLM._replace_rmsnorm_with_npu`, which
    walks the model tree at init time and swaps `Qwen3_5MoeRMSNorm` instances
    in-place (parameters are re-bound, not copied).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_TORCH_NPU:
            import torch_npu  # type: ignore
            return torch_npu.npu_rms_norm(
                x.float(), 1.0 + self.weight.float(), self.eps
            )[0].type_as(x)
        return super().forward(x)


# -----------------------------------------------------------------------------
# SVD / factorization helpers
# -----------------------------------------------------------------------------


def _svd_fp32(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(matrix.shape)}.")
    m32 = matrix.to(torch.float32)
    return torch.linalg.svd(m32, full_matrices=False)


def low_rank_approx(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Best rank approximation in Frobenius norm via truncated SVD."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(matrix.shape)}.")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    max_rank = min(matrix.shape)
    r = min(int(rank), int(max_rank))
    if r == max_rank:
        return matrix.clone()
    U, S, Vh = _svd_fp32(matrix)
    approx = (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]
    return approx.to(dtype=matrix.dtype)


def _check_same_device_dtype(matrices: Sequence[torch.Tensor]) -> None:
    if not matrices:
        raise ValueError("Expected a non-empty sequence of tensors.")
    device = matrices[0].device
    dtype = matrices[0].dtype
    for i, m in enumerate(matrices):
        if m.device != device:
            raise ValueError(f"Tensor at index {i} is on {m.device}, expected {device}.")
        if m.dtype != dtype:
            raise ValueError(f"Tensor at index {i} has dtype {m.dtype}, expected {dtype}.")


def _check_2d_same_shape(matrices: Sequence[torch.Tensor]) -> Tuple[int, int]:
    if not matrices:
        raise ValueError("Expected a non-empty sequence of 2D matrices.")
    shape = matrices[0].shape
    if len(shape) != 2:
        raise ValueError(f"Expected 2D matrices, got shape {tuple(shape)}.")
    for i, m in enumerate(matrices):
        if m.shape != shape:
            raise ValueError(
                f"All matrices must share the same shape. Got {tuple(m.shape)} at index {i}, expected {tuple(shape)}."
            )
    return int(shape[0]), int(shape[1])


def factorize_shared_left(
    matrices: Sequence[torch.Tensor],
    latent_dim: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Factorize M_i = L @ R_i with one shared left factor L.

    All matrices must have shape [in_dim, out_dim].
    Returns:
        shared_left: [in_dim, latent_dim]
        right_blocks: list of [latent_dim, out_dim]
    """
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if not matrices:
        raise ValueError("Expected a non-empty sequence of matrices.")
    _check_same_device_dtype(matrices)
    in_dim, _ = _check_2d_same_shape(matrices)

    concat = torch.cat([m.to(torch.float32) for m in matrices], dim=1)
    U, S, Vh = torch.linalg.svd(concat, full_matrices=False)
    r = min(int(latent_dim), int(S.numel()))

    shared_left = torch.zeros((in_dim, latent_dim), device=concat.device, dtype=torch.float32)
    right_concat = torch.zeros((latent_dim, concat.shape[1]), device=concat.device, dtype=torch.float32)

    if r > 0:
        s_sqrt = torch.sqrt(S[:r])
        shared_left[:, :r] = U[:, :r] * s_sqrt.unsqueeze(0)
        right_concat[:r, :] = s_sqrt.unsqueeze(1) * Vh[:r, :]

    out_dims = [int(m.shape[1]) for m in matrices]
    right_blocks = list(torch.split(right_concat, out_dims, dim=1))
    return shared_left.to(dtype=matrices[0].dtype).contiguous(), [
        b.to(dtype=matrices[0].dtype).contiguous() for b in right_blocks
    ]


def factorize_shared_right(
    matrices: Sequence[torch.Tensor],
    latent_dim: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Factorize M_i = L_i @ R with one shared right factor R.

    All matrices must have shape [in_dim, out_dim].
    Returns:
        left_blocks: list of [in_dim, latent_dim]
        shared_right: [latent_dim, out_dim]
    """
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if not matrices:
        raise ValueError("Expected a non-empty sequence of matrices.")
    _check_same_device_dtype(matrices)
    _check_2d_same_shape(matrices)

    shared_left_T, right_blocks_T = factorize_shared_left([m.T.contiguous() for m in matrices], latent_dim)
    shared_right = shared_left_T.T.contiguous()
    left_blocks = [b.T.contiguous() for b in right_blocks_T]
    return left_blocks, shared_right


def factorize_tied_gate_up(
    gate_matrices: Sequence[torch.Tensor],
    up_matrices: Sequence[torch.Tensor],
    latent_dim: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Jointly factorize gate and up matrices with the SAME shared left factor.

    Returns:
        shared_left: [hidden_dim, latent_dim]
        gate_blocks: list of [latent_dim, intermediate_dim]
        up_blocks:   list of [latent_dim, intermediate_dim]
    """
    if len(gate_matrices) != len(up_matrices):
        raise ValueError("gate_matrices and up_matrices must have the same length.")
    shared_left, blocks = factorize_shared_left(list(gate_matrices) + list(up_matrices), latent_dim)
    n = len(gate_matrices)
    return shared_left, blocks[:n], blocks[n:]


# -----------------------------------------------------------------------------
# Latent expert container
# -----------------------------------------------------------------------------


class Qwen3_5LatentExperts(nn.Module):
    """Packed expert container for LatentMoE-style Qwen3.5 expert weights.

    Stores weights in HuggingFace-compatible layout:
        gate_up_proj: [E, 2 * intermediate_dim, latent_dim]
        down_proj:    [E, latent_dim, intermediate_dim]

    When `config.use_grouped_expert_matmul=True` AND Ascend NPU is available,
    the forward pass uses `torch_npu.npu_moe_token_permute/unpermute`,
    `npu_grouped_matmul`, and `npu_swiglu` for a fused fast path (ported from
    MindSpeed-MM `qwen3_5_moe` modeling). The weight tensors are permuted
    in-place to NPU-friendly layout `[E, in_dim, out_dim]` on the first NPU
    forward; this is inference-only — do not re-save after this happens.
    """

    def __init__(
        self,
        config,
        *,
        latent_dim: Optional[int] = None,
        act_fn=None,
    ) -> None:
        super().__init__()

        self.num_experts = int(config.num_experts)
        self.intermediate_dim = int(config.moe_intermediate_size)
        self.latent_dim = int(latent_dim if latent_dim is not None else getattr(config, "moe_latent_dim"))

        if act_fn is None:
            hidden_act = getattr(config, "hidden_act", None)
            if hidden_act is None:
                raise AttributeError("config.hidden_act is required when act_fn is not provided.")
            from transformers.activations import ACT2FN
            self.act_fn = ACT2FN[hidden_act]
        else:
            self.act_fn = act_fn

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.latent_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.latent_dim, self.intermediate_dim)
        )

        # NPU fast-path opt-in (defaults to False for CPU / GPU / HF compat).
        self.use_grouped_expert_matmul = bool(getattr(config, "use_grouped_expert_matmul", False))
        self._npu_layout_ready = False

    def _ensure_npu_layout(self) -> None:
        """Permute weights in-place once to layout expected by npu_grouped_matmul.

        gate_up_proj: [E, 2I, latent] -> [E, latent, 2I]
        down_proj:    [E, latent, I] -> [E, I, latent]
        """
        if self._npu_layout_ready:
            return
        with torch.no_grad():
            self.gate_up_proj.data = self.gate_up_proj.data.transpose(1, 2).contiguous()
            self.down_proj.data = self.down_proj.data.transpose(1, 2).contiguous()
        self._npu_layout_ready = True

    def _forward_npu_fused(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """NPU fused MoE forward.

        dtype policy: the module preserves `hidden_states.dtype` end-to-end.
        MS-MM's reference code casts `output.to(top_k_weights.dtype)` and
        passes `top_k_weights` straight into `npu_moe_token_unpermute`, which
        assumes the router returned bf16. The HF `Qwen3_5MoeTopKRouter` in
        some installed versions returns fp32 routing weights (softmax is done
        in fp32 and not cast back), so that idiom upcasts the whole MoE output
        to fp32, which then leaks through the residual into the next layer's
        conv1d (bf16 weight) and crashes. We cast probs to the output dtype
        instead, and a final `.to(hidden_states.dtype)` guards against the
        kernel itself upcasting internally.
        """
        import torch_npu  # type: ignore

        self._ensure_npu_layout()
        orig_dtype = hidden_states.dtype

        permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(
            hidden_states, top_k_index.to(torch.int32)
        )
        tokens_per_expert = torch.histc(
            top_k_index, bins=self.num_experts, min=0, max=self.num_experts
        )
        intermediate = _npu_group_gemm(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
        activated = torch_npu.npu_swiglu(intermediate, dim=-1)
        output = _npu_group_gemm(activated, self.down_proj, tokens_per_expert)

        probs = top_k_weights.to(output.dtype)
        final_hidden_states = torch_npu.npu_moe_token_unpermute(
            output, row_ids_map, probs=probs
        )
        return final_hidden_states.to(orig_dtype)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.where(expert_mask.sum(dim=(-1, -2)) > 0)[0]

        for expert_idx in expert_hit:
            expert_idx = int(expert_idx.item())
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass on latent hidden states.

        Args:
            hidden_states: [tokens, latent_dim]
            top_k_index:   [tokens, top_k]
            top_k_weights: [tokens, top_k]
        """
        if hidden_states.ndim != 2:
            raise ValueError(f"Expected hidden_states shape [tokens, latent_dim], got {tuple(hidden_states.shape)}.")
        if hidden_states.shape[-1] != self.latent_dim:
            raise ValueError(f"hidden_states last dim must be {self.latent_dim}, got {hidden_states.shape[-1]}.")

        if self.use_grouped_expert_matmul and _HAS_TORCH_NPU:
            return self._forward_npu_fused(hidden_states, top_k_index, top_k_weights)
        return self._forward_eager(hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def from_qwen35_experts(
        experts_module: nn.Module,
        *,
        latent_dim: int,
        target_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, "Qwen3_5LatentExperts"]:
        """Convert packed Qwen3.5 expert tensors to a LatentMoE-style container.

        Returns:
            latent_down_proj_weight: [latent_dim, hidden_dim] for nn.Linear(hidden_dim -> latent_dim)
            latent_up_proj_weight:   [hidden_dim, latent_dim] for nn.Linear(latent_dim -> hidden_dim)
            latent_experts:          packed latent expert container
        """
        if not hasattr(experts_module, "gate_up_proj") or not hasattr(experts_module, "down_proj"):
            raise AttributeError("Expected Qwen3.5-style experts with gate_up_proj and down_proj parameters.")

        gate_up = experts_module.gate_up_proj.detach()
        down = experts_module.down_proj.detach()
        if gate_up.ndim != 3 or down.ndim != 3:
            raise ValueError("Expected gate_up_proj and down_proj to be 3D tensors.")

        num_experts, two_intermediate, hidden_dim = gate_up.shape
        if two_intermediate % 2 != 0:
            raise ValueError(f"gate_up_proj second dimension must be even, got {two_intermediate}.")
        intermediate_dim = two_intermediate // 2
        expected_down_shape = (num_experts, hidden_dim, intermediate_dim)
        if tuple(down.shape) != expected_down_shape:
            raise ValueError(f"down_proj must have shape {expected_down_shape}, got {tuple(down.shape)}.")

        act_fn = getattr(experts_module, "act_fn", None)
        if act_fn is None:
            act_fn = F.silu

        gate_rows: List[torch.Tensor] = []
        up_rows: List[torch.Tensor] = []
        down_rows: List[torch.Tensor] = []
        for e in range(num_experts):
            gate_rows.append(gate_up[e, :intermediate_dim, :].T.contiguous())
            up_rows.append(gate_up[e, intermediate_dim:, :].T.contiguous())
            down_rows.append(down[e].T.contiguous())

        if target_rank is None:
            target_rank = min(hidden_dim, intermediate_dim)

        # Step 1: rank reduction
        gate_rr = [low_rank_approx(m, target_rank) for m in gate_rows]
        up_rr = [low_rank_approx(m, target_rank) for m in up_rows]
        down_rr = [low_rank_approx(m, target_rank) for m in down_rows]

        # Step 2: tie gate and up through the same shared left factor
        shared_down_row, gate_latent_blocks, up_latent_blocks = factorize_tied_gate_up(
            gate_rr,
            up_rr,
            latent_dim,
        )
        # Step 3: factorize down with a shared right factor
        down_left_blocks, shared_up_row = factorize_shared_right(down_rr, latent_dim)

        # Build the latent expert container
        cfg = type("_Cfg", (), {})()
        cfg.num_experts = num_experts
        cfg.moe_intermediate_size = intermediate_dim
        cfg.moe_latent_dim = latent_dim
        cfg.hidden_act = getattr(experts_module, "hidden_act", "silu")

        latent_experts = Qwen3_5LatentExperts(cfg, latent_dim=latent_dim, act_fn=act_fn)
        device = gate_up.device
        dtype = gate_up.dtype
        latent_experts = latent_experts.to(device=device, dtype=dtype)

        with torch.no_grad():
            # gate_up_proj: [E, 2I, latent_dim]
            latent_experts.gate_up_proj.zero_()
            latent_experts.gate_up_proj[:, :intermediate_dim, :].copy_(
                torch.stack([b.T.contiguous() for b in gate_latent_blocks], dim=0).to(device=device, dtype=dtype)
            )
            latent_experts.gate_up_proj[:, intermediate_dim:, :].copy_(
                torch.stack([b.T.contiguous() for b in up_latent_blocks], dim=0).to(device=device, dtype=dtype)
            )
            # down_proj: [E, latent_dim, I]
            latent_experts.down_proj.zero_()
            latent_experts.down_proj.copy_(
                torch.stack([b.T.contiguous() for b in down_left_blocks], dim=0).to(device=device, dtype=dtype)
            )

        return (
            shared_down_row.to(device=device, dtype=dtype).contiguous(),
            shared_up_row.to(device=device, dtype=dtype).contiguous(),
            latent_experts,
        )


# -----------------------------------------------------------------------------
# Latent MoE block
# -----------------------------------------------------------------------------


class Qwen3_5LatentMoeSparseMoeBlock(nn.Module):
    """LatentMoE replacement for Qwen3.5 SparseMoeBlock."""

    def __init__(self, config: Qwen3_5MoeConfig):
        super().__init__()
        self.config = config
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.intermediate_size = int(config.moe_intermediate_size)

        latent_dim = getattr(config, "moe_latent_dim", None)
        latent_factor = getattr(config, "moe_latent_factor", None)
        if latent_dim is None and latent_factor is None:
            raise ValueError("LatentMoE block requires config.moe_latent_dim or config.moe_latent_factor.")
        if latent_dim is None:
            latent_factor = int(latent_factor)
            if self.hidden_size % latent_factor != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} must be divisible by moe_latent_factor={latent_factor}."
                )
            latent_dim = self.hidden_size // latent_factor
        self.latent_dim = int(latent_dim)

        # Router and shared expert remain unchanged
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        self.latent_down_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=False)
        self.latent_up_proj = nn.Linear(self.latent_dim, config.hidden_size, bias=False)
        self.experts = Qwen3_5LatentExperts(config, latent_dim=self.latent_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden_dim}.")

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)

        latent_input = self.latent_down_proj(hidden_states_reshaped)
        expert_output = self.experts(latent_input, selected_experts, routing_weights)
        expert_output = self.latent_up_proj(expert_output)

        shared_expert_output = torch.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    @classmethod
    def from_qwen35_block(
        cls,
        moe_block: nn.Module,
        *,
        config: Qwen3_5MoeConfig,
        latent_dim: int,
        target_rank: Optional[int] = None,
    ) -> "Qwen3_5LatentMoeSparseMoeBlock":
        if not hasattr(moe_block, "experts"):
            raise AttributeError("The provided module has no `experts` attribute.")
        experts = moe_block.experts
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            raise TypeError("Expected moe_block.experts to be Qwen3.5 packed experts.")

        latent_config = config
        setattr(latent_config, "moe_latent_dim", int(latent_dim))
        if int(latent_dim) > 0 and int(latent_config.hidden_size) % int(latent_dim) == 0:
            setattr(latent_config, "moe_latent_factor", int(latent_config.hidden_size // latent_dim))
        else:
            setattr(latent_config, "moe_latent_factor", None)
        setattr(latent_config, "use_latent_moe", True)

        latent_block = cls(latent_config)
        device = experts.gate_up_proj.device
        dtype = experts.gate_up_proj.dtype
        latent_block = latent_block.to(device=device, dtype=dtype)

        with torch.no_grad():
            latent_block.gate.load_state_dict(moe_block.gate.state_dict(), strict=True)
            latent_block.shared_expert.load_state_dict(moe_block.shared_expert.state_dict(), strict=True)
            latent_block.shared_expert_gate.load_state_dict(moe_block.shared_expert_gate.state_dict(), strict=True)

            down_row, up_row, latent_experts = Qwen3_5LatentExperts.from_qwen35_experts(
                experts,
                latent_dim=latent_dim,
                target_rank=target_rank,
            )

            # Copy shared projections
            latent_block.latent_down_proj.weight.copy_(down_row.T.contiguous().to(device=device, dtype=dtype))
            latent_block.latent_up_proj.weight.copy_(up_row.T.contiguous().to(device=device, dtype=dtype))
            # Copy latent expert parameters
            latent_block.experts.gate_up_proj.copy_(latent_experts.gate_up_proj.to(device=device, dtype=dtype))
            latent_block.experts.down_proj.copy_(latent_experts.down_proj.to(device=device, dtype=dtype))

        return latent_block


class Qwen3_5LatentMoeForCausalLM(Qwen3_5MoeForCausalLM):
    """Causal LM wrapper that can instantiate LatentMoE blocks from config."""

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "use_latent_moe", False):
            self._replace_sparse_moe_blocks_with_latent()
        if getattr(config, "use_npu_rmsnorm", False):
            self._replace_rmsnorm_with_npu()

    def _replace_sparse_moe_blocks_with_latent(self) -> None:
        def default_predicate(module: nn.Module) -> bool:
            has_experts = hasattr(module, "experts")
            has_gate = hasattr(module, "gate")
            has_shared_expert = hasattr(module, "shared_expert")
            has_shared_expert_gate = hasattr(module, "shared_expert_gate")
            experts = getattr(module, "experts", None)
            looks_like_qwen35_experts = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")
            return bool(has_experts and has_gate and has_shared_expert and has_shared_expert_gate and looks_like_qwen35_experts)

        def _recurse(parent: nn.Module) -> None:
            for name, child in list(parent.named_children()):
                if default_predicate(child):
                    setattr(parent, name, Qwen3_5LatentMoeSparseMoeBlock(self.config))
                else:
                    _recurse(child)

        _recurse(self)

    def _replace_rmsnorm_with_npu(self) -> None:
        """Swap every Qwen3_5MoeRMSNorm instance in this model with Qwen3_5NpuRMSNorm.

        Parameters are re-bound (`new.weight = old.weight`), so the swap does
        not copy or reinitialise weights. Triggered by `config.use_npu_rmsnorm`.
        No-op on non-NPU hardware because the subclass's forward already
        degrades to the parent implementation.
        """
        def _recurse(parent: nn.Module) -> None:
            for name, child in list(parent.named_children()):
                if type(child) is Qwen3_5MoeRMSNorm:
                    new = Qwen3_5NpuRMSNorm.__new__(Qwen3_5NpuRMSNorm)
                    nn.Module.__init__(new)
                    new.eps = child.eps
                    new.weight = child.weight
                    setattr(parent, name, new)
                else:
                    _recurse(child)

        _recurse(self)


# -----------------------------------------------------------------------------
# Model conversion helpers
# -----------------------------------------------------------------------------


def convert_qwen35_sparse_moe_block_inplace(
    moe_block: nn.Module,
    *,
    config: Qwen3_5MoeConfig,
    latent_dim: int,
    target_rank: Optional[int] = None,
) -> nn.Module:
    """Replace one Qwen3.5 MoE block with its LatentMoE counterpart."""
    return Qwen3_5LatentMoeSparseMoeBlock.from_qwen35_block(
        moe_block,
        config=config,
        latent_dim=latent_dim,
        target_rank=target_rank,
    )


def _count_matching_modules(model: nn.Module, predicate) -> int:
    """Count modules in `model` (recursively) that satisfy `predicate`."""
    count = 0
    def _rec(parent: nn.Module) -> None:
        nonlocal count
        for _, child in parent.named_children():
            if predicate(child):
                count += 1
            else:
                _rec(child)
    _rec(model)
    return count


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    hours = int(seconds // 3600)
    return f"{hours}h {int((seconds % 3600) // 60):02d}m"


def convert_qwen35_model_to_latentmoe_inplace(
    model: nn.Module,
    *,
    latent_dim: int,
    target_rank: Optional[int] = None,
    moe_block_predicate=None,
    verbose: bool = True,
) -> nn.Module:
    """Recursively replace Qwen3.5 sparse MoE blocks in a model."""
    if not hasattr(model, "config"):
        raise AttributeError("model must have a config attribute.")

    config = model.config
    hidden_size = int(config.hidden_size)
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if hidden_size % latent_dim == 0:
        latent_factor = hidden_size // latent_dim
    else:
        latent_factor = None

    setattr(config, "use_latent_moe", True)
    setattr(config, "moe_latent_dim", int(latent_dim))
    setattr(config, "moe_latent_factor", latent_factor)

    def default_predicate(module: nn.Module) -> bool:
        has_experts = hasattr(module, "experts")
        has_gate = hasattr(module, "gate")
        has_shared_expert = hasattr(module, "shared_expert")
        has_shared_expert_gate = hasattr(module, "shared_expert_gate")
        experts = getattr(module, "experts", None)
        looks_like_qwen35_experts = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")
        return bool(has_experts and has_gate and has_shared_expert and has_shared_expert_gate and looks_like_qwen35_experts)

    predicate = moe_block_predicate or default_predicate

    # Pre-count total MoE blocks so we can show "[i/N]" progress and a rolling ETA.
    # Conversion is dominated by two large SVDs per block, which on CPU can take
    # tens of seconds per layer — without this the user has no way to gauge progress.
    total_blocks = _count_matching_modules(model, predicate) if verbose else 0
    converted = 0
    start_time = time.perf_counter()
    if verbose:
        print(f"[LatentMoE] Found {total_blocks} MoE block(s) to convert.", flush=True)

    def _recurse(parent: nn.Module, path: str = "") -> None:
        nonlocal converted
        for name, child in list(parent.named_children()):
            child_path = f"{path}.{name}" if path else name
            if predicate(child):
                if verbose:
                    converted += 1
                    print(
                        f"[LatentMoE] [{converted:>3d}/{total_blocks}] converting {child_path} ...",
                        end="",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                setattr(
                    parent,
                    name,
                    convert_qwen35_sparse_moe_block_inplace(
                        child,
                        config=config,
                        latent_dim=latent_dim,
                        target_rank=target_rank,
                    ),
                )
                if verbose:
                    dt = time.perf_counter() - t0
                    elapsed = time.perf_counter() - start_time
                    remaining = total_blocks - converted
                    eta = (elapsed / converted) * remaining if converted > 0 else 0.0
                    print(f" done in {dt:.1f}s  (ETA {_format_eta(eta)})", flush=True)
            else:
                _recurse(child, child_path)

    _recurse(model)
    if verbose:
        total_elapsed = time.perf_counter() - start_time
        print(
            f"[LatentMoE] Converted {converted} block(s) in {_format_eta(total_elapsed)}.",
            flush=True,
        )
    return model


# -----------------------------------------------------------------------------
# Save / load helpers
# -----------------------------------------------------------------------------


def prepare_qwen35_latentmoe_config(config, *, latent_dim: int) -> object:
    """Mark a config so the saved checkpoint reloads the latent architecture."""
    hidden_size = int(config.hidden_size)
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")

    if hidden_size % latent_dim == 0:
        latent_factor = hidden_size // latent_dim
    else:
        latent_factor = None

    setattr(config, "use_latent_moe", True)
    setattr(config, "moe_latent_dim", int(latent_dim))
    setattr(config, "moe_latent_factor", latent_factor)
    setattr(config, "architectures", [Qwen3_5LatentMoeForCausalLM.__name__])

    module_name = Path(__file__).stem
    setattr(
        config,
        "auto_map",
        {
            "AutoModelForCausalLM": f"{module_name}.{Qwen3_5LatentMoeForCausalLM.__name__}",
        },
    )
    return config


def save_qwen35_latentmoe_pretrained(
    model: nn.Module,
    save_directory: str | Path,
    *,
    tokenizer=None,
    safe_serialization: bool = True,
) -> Path:
    """Save the converted model in Hugging Face format and copy this Python file."""
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "config"):
        raise AttributeError("model must have a config attribute.")
    if getattr(model.config, "moe_latent_dim", None) is None:
        raise ValueError("model.config.moe_latent_dim must be set before saving.")

    prepare_qwen35_latentmoe_config(model.config, latent_dim=int(model.config.moe_latent_dim))

    source_file = Path(inspect.getfile(Qwen3_5LatentMoeForCausalLM)).resolve()
    shutil.copy2(source_file, save_dir / source_file.name)

    model.save_pretrained(save_dir, safe_serialization=safe_serialization)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)
    return save_dir


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


@torch.no_grad()
def qwen35_expert_reconstruction_error(
    original_experts: nn.Module,
    latent_experts: Qwen3_5LatentExperts,
    *,
    num_tokens: int = 32,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> float:
    """Relative output error on random inputs for sanity-checking."""
    if device is None:
        device = next(latent_experts.parameters()).device
    if dtype is None:
        dtype = next(latent_experts.parameters()).dtype

    hidden_dim = latent_experts.latent_dim
    x = torch.randn(num_tokens, hidden_dim, device=device, dtype=dtype)
    top_k = 2
    top_k_index = torch.randint(0, latent_experts.num_experts, (num_tokens, top_k), device=device)
    top_k_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

    orig = original_experts(x, top_k_index, top_k_weights)
    new = latent_experts(x, top_k_index, top_k_weights)
    denom = orig.norm().clamp_min(1e-12)
    return float((orig - new).norm() / denom)


__all__ = [
    "low_rank_approx",
    "factorize_shared_left",
    "factorize_shared_right",
    "factorize_tied_gate_up",
    "Qwen3_5LatentExperts",
    "Qwen3_5LatentMoeSparseMoeBlock",
    "Qwen3_5LatentMoeForCausalLM",
    "convert_qwen35_sparse_moe_block_inplace",
    "convert_qwen35_model_to_latentmoe_inplace",
    "prepare_qwen35_latentmoe_config",
    "save_qwen35_latentmoe_pretrained",
    "qwen35_expert_reconstruction_error",
]