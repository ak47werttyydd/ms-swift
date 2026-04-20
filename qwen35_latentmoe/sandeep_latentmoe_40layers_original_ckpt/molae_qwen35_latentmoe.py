from __future__ import annotations

import inspect
import math
import shutil
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeModel,
        Qwen3_5MoeMLP,
        Qwen3_5MoeSparseMoeBlock,
        Qwen3_5MoeExperts,
        Qwen3_5MoeTopKRouter,
    )
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
except Exception:
    Qwen3_5MoeForCausalLM = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeForConditionalGeneration = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeModel = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeMLP = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeSparseMoeBlock = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeExperts = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeTopKRouter = nn.Module  # type: ignore[assignment]
    Qwen3_5MoeConfig = object  # type: ignore[assignment]


class Qwen3_5LatentMoeConfig(Qwen3_5MoeConfig):
    model_type = 'qwen3_5_latentmoe'


# -----------------------------------------------------------------------------
# Low-rank / factorization helpers
# -----------------------------------------------------------------------------


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


def low_rank_approx(matrix: torch.Tensor, rank: int) -> torch.Tensor:
    """Best rank-approximation in Frobenius norm via truncated SVD."""
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(matrix.shape)}.")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}.")
    max_rank = min(matrix.shape)
    r = min(int(rank), int(max_rank))
    if r == max_rank:
        return matrix.clone()
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return (U[:, :r] * S[:r].unsqueeze(0)) @ Vh[:r, :]


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

    concat = torch.cat(matrices, dim=1)  # [in_dim, sum(out_dim)]
    orig_dtype = concat.dtype
    U, S, Vh = torch.linalg.svd(concat.float(), full_matrices=False)
    U, S, Vh = U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)
    r = min(int(latent_dim), int(S.numel()))

    shared_left = torch.zeros((in_dim, latent_dim), device=concat.device, dtype=concat.dtype)
    right_concat = torch.zeros((latent_dim, concat.shape[1]), device=concat.device, dtype=concat.dtype)
    if r > 0:
        s_sqrt = torch.sqrt(S[:r])
        shared_left[:, :r] = U[:, :r] * s_sqrt.unsqueeze(0)
        right_concat[:r, :] = s_sqrt.unsqueeze(1) * Vh[:r, :]

    out_dims = [int(m.shape[1]) for m in matrices]
    right_blocks = list(torch.split(right_concat, out_dims, dim=1))
    return shared_left.contiguous(), [b.contiguous() for b in right_blocks]


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

    # M_i ≈ L_i @ R  <=>  M_i^T ≈ R^T @ L_i^T
    shared_left_T, right_blocks_T = factorize_shared_left([m.T.contiguous() for m in matrices], latent_dim)
    shared_right = shared_left_T.T.contiguous()  # [latent_dim, out_dim]
    left_blocks = [b.T.contiguous() for b in right_blocks_T]  # [in_dim, latent_dim]
    return left_blocks, shared_right


def factorize_tied_gate_up(
    gate_matrices: Sequence[torch.Tensor],
    up_matrices: Sequence[torch.Tensor],
    latent_dim: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """Jointly factorize gate and up matrices with the SAME shared left factor.

    The returned shared_left is the tied latent-down projection.
    """
    if len(gate_matrices) != len(up_matrices):
        raise ValueError("gate_matrices and up_matrices must have the same length.")
    concat_mats = list(gate_matrices) + list(up_matrices)
    shared_left, blocks = factorize_shared_left(concat_mats, latent_dim)
    n = len(gate_matrices)
    return shared_left, blocks[:n], blocks[n:]


# -----------------------------------------------------------------------------
# Latent expert container and LatentMoE block for Qwen3.5
# -----------------------------------------------------------------------------


class Qwen3_5LatentExperts(nn.Module):
    """Packed expert container for LatentMoE-style Qwen3.5 expert weights."""

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

        # Guard: detect legacy per-expert checkpoint format and fail early
        # instead of silently loading zeros under DeepSpeed Zero3.
        self._register_load_state_dict_pre_hook(self._check_legacy_state_dict)

    def _check_legacy_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ) -> None:
        """Pre-hook: reject legacy per-expert unpacked checkpoints.

        Legacy format (per-expert ``Qwen3MoeMLP`` keys) is incompatible with
        DeepSpeed Zero3 — the runtime hook that repacks them into packed tensors
        is silently bypassed by Zero3's parameter loading, resulting in all-zero
        expert weights and a broken model.

        Use ``repack_experts.py --convert`` to pre-convert the checkpoint before
        training.
        """
        # Already packed -> OK.
        if (f"{prefix}gate_up_proj" in state_dict
                or f"{prefix}down_proj" in state_dict):
            return

        # Legacy format detected -> fail loudly.
        if f"{prefix}0.gate_proj.weight" in state_dict:
            raise RuntimeError(
                f'Legacy per-expert checkpoint detected (found key '
                f'"{prefix}0.gate_proj.weight"). This format is incompatible '
                f'with DeepSpeed Zero3 and will result in all-zero expert '
                f'weights. Run:\n\n'
                f'    python qwen35_latentmoe/repack_experts.py --convert '
                f'<checkpoint_dir>\n\n'
                f'to convert to packed format before training.'
            )

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

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, tokens]
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hit:
            expert_idx = int(expert_idx_tensor[0].item())
            if expert_idx >= self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    @staticmethod
    def from_qwen35_experts(
        experts_module: nn.Module,
        *,
        latent_dim: int,
        target_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, "Qwen3_5LatentExperts"]:
        """Convert packed Qwen3.5 expert tensors to a LatentMoE-style container.

        Returns:
            latent_down_proj_weight: [latent_dim, hidden_dim] row-vector form
            latent_up_proj_weight:   [hidden_dim, latent_dim] row-vector form
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

        act_fn = getattr(experts_module, "act_fn", F.silu)

        # Convert packed row-vector weights into [in_dim, out_dim] matrices.
        gate_rows: List[torch.Tensor] = []  # [hidden_dim, intermediate_dim]
        up_rows: List[torch.Tensor] = []    # [hidden_dim, intermediate_dim]
        down_rows: List[torch.Tensor] = []  # [intermediate_dim, hidden_dim]
        for e in range(num_experts):
            gate_rows.append(gate_up[e, :intermediate_dim, :].T.contiguous())
            up_rows.append(gate_up[e, intermediate_dim:, :].T.contiguous())
            down_rows.append(down[e].T.contiguous())

        if target_rank is None:
            target_rank = min(hidden_dim, intermediate_dim)

        # Step 1: rank reduction.
        gate_rr = [low_rank_approx(m, target_rank) for m in gate_rows]
        up_rr = [low_rank_approx(m, target_rank) for m in up_rows]
        down_rr = [low_rank_approx(m, target_rank) for m in down_rows]

        # Step 2: tie gate and up through the same shared left factor.
        shared_down_row, gate_latent_blocks, up_latent_blocks = factorize_tied_gate_up(
            gate_rr,
            up_rr,
            latent_dim,
        )

        # Step 3: factorize down with a shared right factor.
        down_left_blocks, shared_up_row = factorize_shared_right(down_rr, latent_dim)

        # Build the latent expert container.
        cfg = type("_Cfg", (), {})()
        cfg.num_experts = num_experts
        cfg.moe_intermediate_size = intermediate_dim
        cfg.moe_latent_dim = latent_dim
        cfg.hidden_act = getattr(experts_module, "hidden_act", None)
        if cfg.hidden_act is None:
            # Fallback to the module's resolved activation if available.
            cfg.hidden_act = "silu"

        latent_experts = Qwen3_5LatentExperts(cfg, latent_dim=latent_dim, act_fn=act_fn)

        device = gate_up.device
        dtype = gate_up.dtype
        latent_experts = latent_experts.to(device=device, dtype=dtype)

        with torch.no_grad():
            # gate_up_proj: [E, 2I, latent_dim]
            latent_experts.gate_up_proj.zero_()
            latent_experts.gate_up_proj[:, :intermediate_dim, :].copy_(
                torch.stack([b.T.contiguous() for b in gate_latent_blocks], dim=0)
            )
            latent_experts.gate_up_proj[:, intermediate_dim:, :].copy_(
                torch.stack([b.T.contiguous() for b in up_latent_blocks], dim=0)
            )

            # down_proj: [E, latent_dim, I]
            latent_experts.down_proj.zero_()
            latent_experts.down_proj.copy_(
                torch.stack([b.T.contiguous() for b in down_left_blocks], dim=0)
            )

        return shared_down_row.contiguous(), shared_up_row.contiguous(), latent_experts


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
            raise ValueError(
                "LatentMoE block requires config.moe_latent_dim or config.moe_latent_factor to be set."
            )
        if latent_dim is None:
            if self.hidden_size % int(latent_factor) != 0:
                raise ValueError(
                    f"hidden_size={self.hidden_size} must be divisible by moe_latent_factor={latent_factor}."
                )
            latent_dim = self.hidden_size // int(latent_factor)
        self.latent_dim = int(latent_dim)

        # Router and shared expert remain unchanged.
        self.gate = Qwen3_5MoeTopKRouter(config)
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

        # Block-level shared latent projections.
        self.latent_down_proj = nn.Linear(config.hidden_size, self.latent_dim, bias=False)
        self.latent_up_proj = nn.Linear(self.latent_dim, config.hidden_size, bias=False)

        # Latent packed experts.
        self.experts = Qwen3_5LatentExperts(config, latent_dim=self.latent_dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(f"Expected hidden size {self.hidden_size}, got {hidden_dim}.")

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        shared_input = hidden_states_reshaped

        shared_expert_output = self.shared_expert(shared_input)
        _, routing_weights, selected_experts = self.gate(shared_input)

        latent_input = self.latent_down_proj(shared_input)
        expert_output = self.experts(latent_input, selected_experts, routing_weights)
        expert_output = self.latent_up_proj(expert_output)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(shared_input)) * shared_expert_output

        expert_output = expert_output + shared_expert_output
        expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)
        return expert_output

    @classmethod
    def from_qwen35_block(
        cls,
        moe_block: nn.Module,
        *,
        latent_dim: int,
        target_rank: Optional[int] = None,
        config=None,
    ) -> "Qwen3_5LatentMoeSparseMoeBlock":
        """Create a LatentMoE block from a standard Qwen3.5 MoE block."""
        if not hasattr(moe_block, "experts"):
            raise AttributeError("The provided module has no `experts` attribute.")
        experts = moe_block.experts
        if not (hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")):
            raise TypeError("This helper expects moe_block.experts to be Qwen3.5 packed experts.")

        if config is None:
            config = getattr(moe_block, "config", None)
        if config is None:
            raise AttributeError(
                "'config' was not found on moe_block and was not passed explicitly. "
                "Pass config= when calling from_qwen35_block()."
            )
        setattr(config, "moe_latent_dim", int(latent_dim))
        setattr(config, "moe_latent_factor", int(config.hidden_size // latent_dim) if latent_dim else None)
        setattr(config, "use_latent_moe", True)

        latent_block = cls(config)
        latent_block = latent_block.to(device=experts.gate_up_proj.device, dtype=experts.gate_up_proj.dtype)

        # Copy shared expert and router weights exactly.
        with torch.no_grad():
            latent_block.gate.load_state_dict(moe_block.gate.state_dict(), strict=True)
            latent_block.shared_expert.load_state_dict(moe_block.shared_expert.state_dict(), strict=True)
            latent_block.shared_expert_gate.load_state_dict(moe_block.shared_expert_gate.state_dict(), strict=True)

            down_row, up_row, latent_experts = Qwen3_5LatentExperts.from_qwen35_experts(
                experts,
                latent_dim=latent_dim,
                target_rank=target_rank,
            )

            # Copy shared projections.
            latent_block.latent_down_proj.weight.copy_(down_row.T.contiguous())
            latent_block.latent_up_proj.weight.copy_(up_row.T.contiguous())

            # Copy latent expert parameters.
            latent_block.experts.gate_up_proj.copy_(latent_experts.gate_up_proj)
            latent_block.experts.down_proj.copy_(latent_experts.down_proj)

        return latent_block


def _replace_sparse_moe_blocks_with_latent(root: nn.Module, moe_config) -> None:
    """Walk the module tree and replace Qwen3.5 SparseMoeBlocks with LatentMoE blocks.

    `moe_config` must carry the flat text-model MoE attributes (`hidden_size`,
    `num_experts`, `moe_intermediate_size`, `shared_expert_intermediate_size`,
    `moe_latent_dim`/`moe_latent_factor`, ...). For `Qwen3_5MoeConfig`
    (nested multimodal), pass `config.text_config`.
    """

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
                setattr(parent, name, Qwen3_5LatentMoeSparseMoeBlock(moe_config))
            else:
                _recurse(child)

    _recurse(root)


class Qwen3_5LatentMoeForCausalLM(Qwen3_5MoeForCausalLM):
    """Causal LM wrapper that can instantiate LatentMoE blocks from config."""
    config_class = Qwen3_5LatentMoeConfig

    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "use_latent_moe", False):
            self._replace_sparse_moe_blocks_with_latent()

    def _replace_sparse_moe_blocks_with_latent(self) -> None:
        # For CausalLM, self.config is the flat Qwen3_5MoeTextConfig.
        _replace_sparse_moe_blocks_with_latent(self, self.config)


class Qwen3_5LatentMoeForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """Multimodal (vision + text) LM wrapper that can instantiate LatentMoE blocks
    inside the language model stack.

    Module tree (inherited from `Qwen3_5MoeForConditionalGeneration`):
        self.model           : Qwen3_5MoeModel
          ├── visual         : Qwen3_5MoeVisionModel
          └── language_model : Qwen3_5MoeTextModel  (← SparseMoeBlocks live here)

    The recursive walker handles the extra nesting transparently; the MoE
    attributes are read from `config.text_config` rather than `config`.
    """
    config_class = Qwen3_5LatentMoeConfig

    def __init__(self, config):
        super().__init__(config)
        text_config = getattr(config, "text_config", config)
        use_latent = (
            getattr(text_config, "use_latent_moe", False) or getattr(config, "use_latent_moe", False)
        )
        if use_latent:
            self._replace_sparse_moe_blocks_with_latent()

    def _replace_sparse_moe_blocks_with_latent(self) -> None:
        # For ConditionalGeneration, MoE attrs live under config.text_config.
        text_config = getattr(self.config, "text_config", self.config)
        _replace_sparse_moe_blocks_with_latent(self, text_config)

    
    # def save_pretrained(self):
    #     transformers.auto.save_pretained(self)


# -----------------------------------------------------------------------------
# Model conversion helpers
# -----------------------------------------------------------------------------


def convert_qwen35_sparse_moe_block_inplace(
    moe_block: nn.Module,
    *,
    latent_dim: int,
    target_rank: Optional[int] = None,
    config=None,
) -> nn.Module:
    """Replace only moe_block with a latent Qwen3.5 MoE block."""
    latent_block = Qwen3_5LatentMoeSparseMoeBlock.from_qwen35_block(
        moe_block,
        latent_dim=latent_dim,
        target_rank=target_rank,
        config=config,
    )
    return latent_block


def convert_qwen35_model_to_latentmoe_inplace(
    model: nn.Module,
    *,
    latent_dim: int,
    target_rank: Optional[int] = None,
    moe_block_predicate=None,
) -> nn.Module:
    """Recursively replace Qwen3.5 sparse MoE blocks in a model."""

    def default_predicate(module: nn.Module) -> bool:
        has_experts = hasattr(module, "experts")
        has_gate = hasattr(module, "gate")
        has_shared_expert = hasattr(module, "shared_expert")
        has_shared_expert_gate = hasattr(module, "shared_expert_gate")
        experts = getattr(module, "experts", None)
        looks_like_qwen35_experts = hasattr(experts, "gate_up_proj") and hasattr(experts, "down_proj")
        return bool(has_experts and has_gate and has_shared_expert and has_shared_expert_gate and looks_like_qwen35_experts)

    predicate = moe_block_predicate or default_predicate

    model_config = getattr(model, "config", None)

    def _recurse(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if predicate(child):
                latent_block = convert_qwen35_sparse_moe_block_inplace(
                    child,
                    latent_dim=latent_dim,
                    target_rank=target_rank,
                    config=model_config,
                )
                setattr(parent, name, latent_block)
            else:
                _recurse(child)

    _recurse(model)
    return model


# -----------------------------------------------------------------------------
# Save / load helpers
# -----------------------------------------------------------------------------


def prepare_qwen35_latentmoe_config(config, *, latent_dim: int) -> object:
    """Mark a config so the saved checkpoint reloads the latent architecture."""
    hidden_size = int(config.hidden_size)
    if latent_dim <= 0:
        raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
    if hidden_size % latent_dim != 0:
        # Save the exact latent dimension; latent_factor may be non-integer.
        latent_factor = None
    else:
        latent_factor = hidden_size // latent_dim

    setattr(config, "use_latent_moe", True)
    setattr(config, "moe_latent_dim", int(latent_dim))
    setattr(config, "moe_latent_factor", latent_factor)
    setattr(config, "architectures", [Qwen3_5LatentMoeForCausalLM.__name__])
    setattr(
        config,
        "auto_map",
        {
            "AutoModelForCausalLM": f"{Path(__file__).stem}.{Qwen3_5LatentMoeForCausalLM.__name__}",
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
    """Save the converted model in Hugging Face format and copy this Python file.

    After saving, load with:
        AutoModelForCausalLM.from_pretrained(save_dir, trust_remote_code=True)
    """
    save_dir = Path(save_directory)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(model, "config"):
        raise AttributeError("model must have a config attribute.")

    latent_dim = getattr(model.config, "moe_latent_dim", None)
    if latent_dim is None:
        raise ValueError("model.config.moe_latent_dim must be set before saving.")
    prepare_qwen35_latentmoe_config(model.config, latent_dim=int(latent_dim))

    # Copy this module into the checkpoint directory so trust_remote_code can reload it.
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
    "Qwen3_5LatentMoeForConditionalGeneration",
    "convert_qwen35_sparse_moe_block_inplace",
    "convert_qwen35_model_to_latentmoe_inplace",
    "prepare_qwen35_latentmoe_config",
    "save_qwen35_latentmoe_pretrained",
    "qwen35_expert_reconstruction_error",
]
