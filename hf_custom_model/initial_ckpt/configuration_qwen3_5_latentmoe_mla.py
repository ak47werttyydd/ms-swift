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
"""Configuration for Qwen3.5 LatentMoE + MLA model."""

from transformers.configuration_utils import PreTrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import RopeParameters


class Qwen3_5LatentMoeMLATextConfig(PreTrainedConfig):
    r"""
    Configuration for the text backbone of Qwen3.5-LatentMoE-MLA.

    Extends the standard Qwen3.5-MoE text config with:
      - **MLA** (Multi-Latent Attention): compressed QKV via low-rank bottleneck
      - **LatentMoE**: down/up projections around expert dispatch to reduce expert compute
    """

    model_type = "qwen3_5_latentmoe_mla_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        # MLA projections
        "layers.*.self_attn.linear_qkv": "replicated",
        "layers.*.self_attn.linear_q_up_proj": "colwise",
        "layers.*.self_attn.linear_kv_up_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.kv_norm": "replicated_with_grad_allreduce",
        # MoE experts
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=True,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias=False,
        attention_dropout=0.0,
        # --- Full attention head dim (for RoPE / MLA compatibility) ---
        head_dim=256,
        # --- MLA parameters ---
        mla=True,
        mla_args: dict | None = None,
        attn_output_gate=True,
        # --- Linear attention (GatedDeltaNet) ---
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=16,
        # --- MoE ---
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=128,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        # --- LatentMoE ---
        latent_moe=True,
        latent_moe_factor=2,
        # --- MTP ---
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        # --- Layer types ---
        layer_types=None,
        mlp_only_layers=None,
        # --- SSM dtype ---
        mamba_ssm_dtype="float32",
        # --- Misc ---
        intermediate_size=6144,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        kwargs["ignore_keys_at_rope_validation"] = {"mrope_section", "mrope_interleaved"}
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim
        self.rope_parameters = rope_parameters
        kwargs.setdefault("partial_rotary_factor", 0.25)

        # MLA
        self.mla = mla
        self.mla_args = mla_args or {
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
        }
        self.attn_output_gate = attn_output_gate

        # Layer types
        self.layer_types = layer_types
        if self.layer_types is None:
            interval_pattern = kwargs.get("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # Linear attention
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # LatentMoE
        self.latent_moe = latent_moe
        self.latent_moe_factor = latent_moe_factor

        # MTP
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

        # Misc
        self.mlp_only_layers = mlp_only_layers or []
        self.mamba_ssm_dtype = mamba_ssm_dtype

        super().__init__(**kwargs)


class Qwen3_5LatentMoeMLAVisionConfig(PreTrainedConfig):
    model_type = "qwen3_5_latentmoe_mla"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=24,
        hidden_size=1024,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4096,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2048,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range


class Qwen3_5LatentMoeMLAConfig(PreTrainedConfig):
    r"""
    Top-level configuration for Qwen3.5-LatentMoE-MLA (multimodal).

    Combines vision and text configs, with MLA attention and LatentMoE routing
    in the text backbone.
    """

    model_type = "qwen3_5_latentmoe_mla"
    sub_configs = {
        "vision_config": Qwen3_5LatentMoeMLAVisionConfig,
        "text_config": Qwen3_5LatentMoeMLATextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(**kwargs)


__all__ = [
    "Qwen3_5LatentMoeMLAConfig",
    "Qwen3_5LatentMoeMLATextConfig",
    "Qwen3_5LatentMoeMLAVisionConfig",
]
