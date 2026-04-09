# Copyright 2025 Huawei PCL. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qwen3.5-LatentMoE-MLA model.

Extends the standard Qwen3.5-MoE with:
  - **MLA (Multi-Latent Attention)**: Compressed QKV projection via low-rank
    bottleneck (DeepSeek-V2 style). Replaces the standard GQA attention on
    full-attention layers.
  - **LatentMoE**: Down/up projections around expert dispatch so that routed
    experts process tokens at a reduced hidden dimension (Nemotron-3 style).

Everything else (hybrid GatedDeltaNet linear attention, shared experts,
vision encoder, MTP, MRoPE, dynamic cache) is inherited from the upstream
Qwen3_5Moe classes unchanged.
"""

from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_experts_implementation
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPooling, ModelOutput
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging, torch_compilable_check
from transformers.utils.generic import is_flash_attention_requested, maybe_autocast, merge_with_config_defaults
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
from transformers.utils.output_capturing import OutputRecorder, capture_outputs

from .configuration_qwen3_5_latentmoe_mla import (
    Qwen3_5LatentMoeMLAConfig,
    Qwen3_5LatentMoeMLATextConfig,
    Qwen3_5LatentMoeMLAVisionConfig,
)

# Re-use the upstream Qwen3.5-MoE components that are NOT changed
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDynamicCache,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeMLP,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeVisionAttention,
    Qwen3_5MoeVisionBlock,
    Qwen3_5MoeVisionMLP,
    Qwen3_5MoeVisionModel,
    Qwen3_5MoeVisionPatchEmbed,
    Qwen3_5MoeVisionPatchMerger,
    Qwen3_5MoeVisionRotaryEmbedding,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    eager_attention_forward,
    load_balancing_loss_func,
    repeat_kv,
    rotate_half,
)

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None


logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Rotary embedding (identical logic, but references our config class)
# ---------------------------------------------------------------------------

class Qwen3_5LatentMoeMLATextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5LatentMoeMLATextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


# ---------------------------------------------------------------------------
# MLA Attention  (replaces Qwen3_5MoeAttention on full-attention layers)
# ---------------------------------------------------------------------------

class MLAAttention(nn.Module):
    """
    Multi-Latent Attention (DeepSeek-V2 style).

    Instead of standard Q/K/V projections, hidden states are compressed through
    a low-rank bottleneck:
      hidden -> linear_qkv -> [q_compressed, kv_compressed, k_rope_input]
      q_compressed -> q_norm -> linear_q_up_proj -> [q_nope, q_rope]
      kv_compressed -> kv_norm -> linear_kv_up_proj -> [k_nope, value]
      key = concat(k_nope, k_rope)   query = concat(q_nope, q_rope)
    """

    def __init__(self, config: Qwen3_5LatentMoeMLATextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_causal = True

        mla = config.mla_args
        self.q_lora_rank = mla["q_lora_rank"]
        self.kv_lora_rank = mla["kv_lora_rank"]
        self.qk_nope_head_dim = mla["qk_nope_head_dim"]
        self.qk_rope_head_dim = mla["qk_rope_head_dim"]
        self.v_head_dim = mla["v_head_dim"]
        self.num_heads = config.num_attention_heads
        self.attn_output_gate = config.attn_output_gate

        # Combined head dim for Q and K (nope + rope)
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** -0.5
        self.attention_dropout = config.attention_dropout

        # Compressed QKV projection (no TP — small bottleneck)
        qkv_out_dim = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        self.linear_qkv = nn.Linear(config.hidden_size, qkv_out_dim, bias=False)

        # Q up-projection: q_lora_rank -> num_heads * (qk_nope_head_dim + qk_rope_head_dim)
        q_up_out = self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        if self.attn_output_gate:
            # Double the output for gating (query + gate)
            q_up_out *= 2
        self.linear_q_up_proj = nn.Linear(self.q_lora_rank, q_up_out, bias=False)

        # KV up-projection: kv_lora_rank -> num_heads * (qk_nope_head_dim + v_head_dim)
        kv_up_out = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)
        self.linear_kv_up_proj = nn.Linear(self.kv_lora_rank, kv_up_out, bias=False)

        # Layer norms on compressed representations
        self.q_norm = Qwen3_5MoeRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.kv_norm = Qwen3_5MoeRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # Output projection: num_heads * v_head_dim -> hidden_size
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

        # For repeat_kv / attention interface compatibility
        self.num_key_value_groups = 1  # MLA: all heads share KV, but we expand to num_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]  # (batch, seq_len)
        bsz, seq_len = input_shape

        # --- Compressed QKV ---
        qkv = self.linear_qkv(hidden_states)  # (b, s, q_lora + kv_lora + rope_dim)
        q_compressed, kv_compressed, k_rope_input = torch.split(
            qkv, [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # --- Q path ---
        q_compressed = self.q_norm(q_compressed)
        q_up = self.linear_q_up_proj(q_compressed)  # (b, s, num_heads * (nope + rope) [* 2])

        if self.attn_output_gate:
            # Split into query states and gate
            q_up = q_up.view(bsz, seq_len, self.num_heads, self.q_head_dim * 2)
            query_states, gate = q_up[..., :self.q_head_dim], q_up[..., self.q_head_dim:]
            gate = gate.reshape(bsz, seq_len, -1)  # (b, s, num_heads * q_head_dim)
        else:
            query_states = q_up.view(bsz, seq_len, self.num_heads, self.q_head_dim)
            gate = None

        # Split query into nope and rope parts
        q_nope = query_states[..., :self.qk_nope_head_dim]
        q_rope = query_states[..., self.qk_nope_head_dim:]

        # --- KV path ---
        kv_compressed = self.kv_norm(kv_compressed)
        kv_up = self.linear_kv_up_proj(kv_compressed)  # (b, s, num_heads * (nope + v_dim))
        kv_up = kv_up.view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)

        k_nope = kv_up[..., :self.qk_nope_head_dim]
        value_states = kv_up[..., self.qk_nope_head_dim:]  # (b, s, num_heads, v_head_dim)

        # --- RoPE on rope parts ---
        # k_rope_input is shared across all heads: (b, s, rope_dim)
        k_rope = k_rope_input.unsqueeze(2).expand(-1, -1, self.num_heads, -1)  # (b, s, nh, rope_dim)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q_rope = q_rope.transpose(1, 2)  # (b, nh, s, rope_dim)
        k_rope = k_rope.transpose(1, 2)
        q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
        q_rope = q_rope.transpose(1, 2)  # back to (b, s, nh, rope_dim)
        k_rope = k_rope.transpose(1, 2)

        # --- Concatenate nope + rope ---
        query_states = torch.cat([q_nope, q_rope], dim=-1)  # (b, s, nh, q_head_dim)
        key_states = torch.cat([k_nope, k_rope], dim=-1)    # (b, s, nh, q_head_dim)

        # Transpose to (b, nh, s, dim) for attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # --- KV cache ---
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # --- Attention ---
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        # flash_attn requires equal Q/K/V head dims; pad V when they differ
        if query_states.shape[-1] != value_states.shape[-1]:
            value_states_padded = F.pad(value_states, (0, query_states.shape[-1] - value_states.shape[-1]))
        else:
            value_states_padded = value_states

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states_padded,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # Remove padding from attention output
        if query_states.shape[-1] != self.v_head_dim:
            attn_output = attn_output[..., :self.v_head_dim]

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # Gate
        if gate is not None:
            # gate has shape (b, s, num_heads * q_head_dim), but attn_output is (b, s, num_heads * v_head_dim)
            # We need to project gate to match attn_output dim
            # Actually, following the MindSpeed pattern: gate is applied per-head then projected
            # But here we simplify: sigmoid gate on the output
            # The gate from q_up covers num_heads * q_head_dim dimensions
            # We apply gate at the per-head v_head_dim level by reshaping
            attn_output_gated = attn_output.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            # Use only the nope part of gate for gating (matching v_head_dim per head)
            gate_per_head = gate.view(bsz, seq_len, self.num_heads, self.q_head_dim)
            gate_per_head = gate_per_head[..., :self.v_head_dim]
            attn_output = (attn_output_gated * torch.sigmoid(gate_per_head)).reshape(bsz, seq_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# LatentMoE Sparse Block  (replaces Qwen3_5MoeSparseMoeBlock)
# ---------------------------------------------------------------------------

class Qwen3_5LatentMoeMLATopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices


@use_experts_implementation
class Qwen3_5LatentMoeMLAExperts(nn.Module):
    """
    Expert weights that operate on the reduced (latent) dimension.
    gate_up_proj: (num_experts, 2 * moe_intermediate_size, latent_dim)
    down_proj:    (num_experts, latent_dim, moe_intermediate_size)
    """

    def __init__(self, config, latent_dim: int):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = latent_dim  # reduced dimension
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class LatentMoeSparseMoeBlock(nn.Module):
    """
    Sparse MoE block with latent dimension reduction.

    The router operates on the original hidden_size, but before dispatching to
    experts, hidden states are projected down by ``latent_moe_factor``. After
    expert processing, they are projected back up. The shared expert operates at
    the original dimension.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.latent_moe = getattr(config, "latent_moe", False)
        self.latent_moe_factor = getattr(config, "latent_moe_factor", 1)

        if self.latent_moe and self.latent_moe_factor > 1:
            self.latent_dim = self.hidden_size // self.latent_moe_factor
            self.down_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=False)
            self.up_proj = nn.Linear(self.latent_dim, self.hidden_size, bias=False)
        else:
            self.latent_dim = self.hidden_size

        self.gate = Qwen3_5LatentMoeMLATopKRouter(config)
        self.experts = Qwen3_5LatentMoeMLAExperts(config, latent_dim=self.latent_dim)
        self.shared_expert = Qwen3_5MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Shared expert (operates at original dimension)
        shared_expert_output = self.shared_expert(hidden_states_flat)

        # Router (operates at original dimension)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_flat)

        # LatentMoE: project down before expert dispatch
        if self.latent_moe and self.latent_moe_factor > 1:
            expert_input = self.down_proj(hidden_states_flat)
        else:
            expert_input = hidden_states_flat

        # Routed experts (at latent dimension)
        expert_output = self.experts(expert_input, selected_experts, routing_weights)

        # LatentMoE: project back up
        if self.latent_moe and self.latent_moe_factor > 1:
            expert_output = self.up_proj(expert_output)

        # Gated shared expert
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_flat)) * shared_expert_output

        output = expert_output + shared_expert_output
        output = output.reshape(batch_size, sequence_length, hidden_dim)
        return output, router_logits


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Qwen3_5LatentMoeMLADecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5LatentMoeMLATextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5MoeGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = MLAAttention(config, layer_idx)

        self.mlp = LatentMoeSparseMoeBlock(config)
        self.input_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, _ = hidden_states
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

class Qwen3_5LatentMoeMLAPreTrainedModel(PreTrainedModel):
    config_class = Qwen3_5LatentMoeMLAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5LatentMoeMLADecoderLayer", "Qwen3_5MoeVisionBlock"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range if hasattr(self.config, "initializer_range") else 0.02
        if isinstance(module, Qwen3_5MoeGatedDeltaNet):
            from transformers import initialization as init
            init.ones_(module.dt_bias)
            init.copy_(module.A_log, torch.empty_like(module.A_log).uniform_(0, 16).log_())
        elif isinstance(module, Qwen3_5MoeRMSNorm):
            module.weight.data.zero_()
        elif isinstance(module, (Qwen3_5LatentMoeMLAExperts,)):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)
        elif isinstance(module, LatentMoeSparseMoeBlock):
            module.gate.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# ---------------------------------------------------------------------------
# Text model
# ---------------------------------------------------------------------------

class Qwen3_5LatentMoeMLATextModel(Qwen3_5LatentMoeMLAPreTrainedModel):
    config_class = Qwen3_5LatentMoeMLATextConfig

    def __init__(self, config: Qwen3_5LatentMoeMLATextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5LatentMoeMLADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5LatentMoeMLATextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = Qwen3_5MoeDynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        from transformers.masking_utils import create_causal_mask
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if decoder_layer.layer_type == "linear_attention" else causal_mask

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModelOutputWithPast
        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

    def _update_linear_attn_mask(self, attention_mask, cache_position):
        linear_attn_mask = attention_mask
        if cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1)):
            linear_attn_mask = None
        return linear_attn_mask


# ---------------------------------------------------------------------------
# Multimodal model (vision + language)
# ---------------------------------------------------------------------------

class Qwen3_5LatentMoeMLAModel(Qwen3_5LatentMoeMLAPreTrainedModel):
    base_model_prefix = "model"
    accepts_loss_kwargs = False
    config_class = Qwen3_5LatentMoeMLAConfig

    def __init__(self, config: Qwen3_5LatentMoeMLAConfig):
        super().__init__(config)
        self.visual = Qwen3_5MoeVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5LatentMoeMLATextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # Delegate vision position and rope index to the upstream implementation pattern
    def get_vision_position_ids(self, start_position, grid_thw, temp_merge_size=1, spatial_merge_size=1, time_interval=1, device=None):
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(llm_grid_h * llm_grid_t)
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(llm_grid_w * llm_grid_t)
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long) * time_interval
        return torch.stack([position_temporal, position_height, position_width], dim=0)

    def get_rope_index(self, input_ids, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None, **kwargs):
        import itertools
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        mrope_position_deltas = []
        position_ids = torch.zeros(3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device)
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }
        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]
            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                input_type_group.append((key, group[0][0], group[-1][0] + 1))
            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device)
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    @can_return_tuple
    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.get_image_features(pixel_values_videos, video_grid_thw, **kwargs)

    @can_return_tuple
    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        vision_output.pooler_output = image_embeds
        return vision_output

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features=None, video_features=None):
        if input_ids is None:
            special_image_mask = (inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
            special_video_mask = (inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.video_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None:
            torch_compilable_check(
                inputs_embeds[special_image_mask].numel() == image_features.numel(),
                f"Image features and image tokens do not match: {n_image_tokens} vs {image_features.shape[0]}")

        n_video_tokens = special_video_mask.sum()
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None:
            torch_compilable_check(
                inputs_embeds[special_video_mask].numel() == video_features.numel(),
                f"Video features and video tokens do not match: {n_video_tokens} vs {video_features.shape[0]}")
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(self, input_ids, inputs_embeds, image_grid_thw=None, video_grid_thw=None, mm_token_type_ids=None, attention_mask=None, **kwargs):
        if mm_token_type_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids if input_ids is not None else torch.zeros_like(inputs_embeds[..., 0], dtype=torch.long),
                mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask)
            self.rope_deltas = rope_deltas
            return position_ids
        return None

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.get_image_features(pixel_values, image_grid_thw, **kwargs)
            if isinstance(image_embeds, BaseModelOutputWithPooling):
                image_embeds = image_embeds.pooler_output
            image_features = torch.cat(image_embeds, dim=0) if isinstance(image_embeds, (list, tuple)) else image_embeds
        else:
            image_features = None

        if pixel_values_videos is not None:
            video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)
            if isinstance(video_embeds, BaseModelOutputWithPooling):
                video_embeds = video_embeds.pooler_output
            video_features = torch.cat(video_embeds, dim=0) if isinstance(video_embeds, (list, tuple)) else video_embeds
        else:
            video_features = None

        if image_features is not None or video_features is not None:
            special_image_mask, special_video_mask = self.get_placeholder_mask(input_ids, inputs_embeds, image_features, video_features)
            if image_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
            if video_features is not None:
                inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids, inputs_embeds, image_grid_thw, video_grid_thw, mm_token_type_ids, attention_mask)

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModelOutputWithPast
        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            router_logits=outputs.router_logits,
        )


# ---------------------------------------------------------------------------
# For Conditional Generation (multimodal)
# ---------------------------------------------------------------------------

from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeCausalLMOutputWithPast


class Qwen3_5LatentMoeMLAForConditionalGeneration(Qwen3_5LatentMoeMLAPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    config_class = Qwen3_5LatentMoeMLAConfig

    def __init__(self, config: Qwen3_5LatentMoeMLAConfig):
        super().__init__(config)
        self.model = Qwen3_5LatentMoeMLAModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    @auto_docstring
    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.model.get_video_features(pixel_values_videos=pixel_values_videos, video_grid_thw=video_grid_thw, **kwargs)

    @auto_docstring
    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        return self.model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw, **kwargs)

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | Qwen3_5MoeCausalLMOutputWithPast:

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        aux_loss = None
        if kwargs.get("output_router_logits", False):
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.config.text_config.num_experts,
                self.config.text_config.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

        return Qwen3_5MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            router_logits=outputs.router_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)
        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            return text_positions[None, ...] + self.model.rope_deltas

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if (
            is_input_ids
            and model_kwargs.get("mm_token_type_ids") is not None
            and (model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None)
        ):
            mk = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
            vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **mk)
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device)

        text_positions = text_positions[None, ...]
        return torch.cat([text_positions, vision_positions], dim=0)


__all__ = [
    "Qwen3_5LatentMoeMLAForConditionalGeneration",
    "Qwen3_5LatentMoeMLAModel",
    "Qwen3_5LatentMoeMLATextModel",
    "Qwen3_5LatentMoeMLAPreTrainedModel",
]
