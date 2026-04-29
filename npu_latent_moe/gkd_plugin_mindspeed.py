"""ms-swift external plugin: register Qwen3.5-LatentMoE for MindSpeed (Megatron)
NPU GKD training.

Usage:
    megatron rlhf --external_plugins npu_latent_moe/gkd_plugin_mindspeed.py \
                  --model_type qwen3_5_latentmoe \
                  --model npu_latent_moe/ckpt_latentmoe_40l ...

Differences vs gkd_plugin.py (DeepSpeed variant)
------------------------------------------------
* **No** `leaf_modules` / Zero3 hook — Megatron doesn't use DeepSpeed Zero3.
  The equivalent concern is solved by Megatron's EP (expert-parallel) layout,
  which is configured at the CLI (`--expert_model_parallel_size N`), not in
  the plugin.
* **Keep** the HF-side `register_model` call — even under Megatron, ms-swift
  still loads the HF tokenizer/processor and reads `config.json` (including
  `auto_map`) to resolve `architectures` and the chat template before handing
  weights to mcore-bridge for HF -> mcore conversion.
* **Add** a Megatron converter registration. For LatentMoE this is the
  hard part: mcore-bridge has no built-in rule for custom trust_remote_code
  architectures. The stubs below mark exactly what you must fill in.

What mcore-bridge needs (summary, read before implementing)
-----------------------------------------------------------
`mcore_bridge` (https://github.com/modelscope/mcore-bridge) maps an HF
`config.model_type` to a mcore `ModelConfig` + a GPTBridge that knows how to
shuffle `state_dict` keys between the two formats. Built-in bridges cover:
    qwen2 / qwen2_moe / qwen3 / qwen3_moe / deepseek_v3 / mixtral / ...
LatentMoE is NOT in that list. Two realistic paths:

  (A) Upstream-style: add a custom `GPTBridge` subclass via
      `mcore_bridge.register_model_bridge(model_type, bridge_cls)`. Requires:
        - `hf_to_mcore_config(hf_config)` override returning a mcore
          `TransformerConfig` with LatentMoE-specific fields wired
        - `convert_weights_hf_to_mcore(hf_state_dict) -> mcore_state_dict`
          and the reverse
        - A `model_provider(pre_process, post_process)` that builds a mcore
          GPTModel subclass with:
            * linear_attention layers (NOT standard DotProductAttention) —
              most work, since Megatron 0.15 has no built-in mamba/linear
              attention; port from `v5molae_qwen35_latentmoe.py`
            * LatentMoE block with latent down/up projections — subclass
              mcore MoELayer
            * MTP head (`mtp_num_hidden_layers=1`) — mcore 0.12+ has native
              MTP, param naming differs
      Weight converter map (HF -> mcore) for a standard dense/MoE Qwen3 baseline:
          embed_tokens.weight                      -> embedding.word_embeddings.weight
          layers.{i}.input_layernorm.weight        -> decoder.layers.{i}.input_layernorm.weight
          layers.{i}.self_attn.q_proj.weight       -> decoder.layers.{i}.self_attention.linear_q.weight (split from linear_qkv)
          layers.{i}.self_attn.k_proj.weight       -> decoder.layers.{i}.self_attention.linear_k.weight
          layers.{i}.self_attn.v_proj.weight       -> decoder.layers.{i}.self_attention.linear_v.weight
          layers.{i}.self_attn.o_proj.weight       -> decoder.layers.{i}.self_attention.linear_proj.weight
          layers.{i}.mlp.gate.weight               -> decoder.layers.{i}.mlp.router.weight
          layers.{i}.mlp.experts.gate_up_proj      -> decoder.layers.{i}.mlp.experts.linear_fc1.weight (per-expert slice)
          layers.{i}.mlp.experts.down_proj         -> decoder.layers.{i}.mlp.experts.linear_fc2.weight
          lm_head.weight                           -> output_layer.weight
      For LatentMoE, also: `moe_latent_{gate,up,down}_proj` (new keys, no mcore
      analogue — must be added to the custom MoELayer subclass).

  (B) Fallback-style: treat the student as plain Qwen3-MoE for Megatron,
      dropping the LatentMoE custom layers. This loses the whole point of
      LatentMoE (linear attention + latent experts). Only useful for smoke
      testing the pipeline end-to-end; do NOT use for real training.

Until (A) is done, running this plugin with `megatron rlhf` will fail at
config parsing with something like
    ValueError: model_type 'qwen3_5_latentmoe' not registered in mcore_bridge
which is exactly the signal to start implementing the bridge.
"""
from swift.llm import ModelMeta, ModelGroup, Model, register_model
from swift.llm.model.register import ModelLoader
from swift.llm.template import TemplateType

# ─── HF side: same as the DeepSpeed plugin MINUS the Zero3 leaf_modules ─────
# ms-swift still needs this for tokenizer/processor loading, chat template
# resolution, and `architectures` routing into mcore-bridge.

register_model(
    ModelMeta(
        'qwen3_5_latentmoe',
        [ModelGroup(
            [Model()],
            TemplateType.qwen3_5,
        )],
        ModelLoader,  # plain loader — no Zero3 leaf_modules injection
        architectures=['Qwen3_5LatentMoeForCausalLM'],
        requires=['transformers>=5.2.0'],
    )
)

# ─── Megatron side: mcore-bridge converter registration ─────────────────────
# This block is a SCAFFOLD. It runs lazily so importing the plugin on a host
# without mcore-bridge (e.g. eval machine) doesn't crash. Fill in the TODOs
# before expecting `megatron rlhf` to actually work for LatentMoE.

def _register_mindspeed_bridge() -> None:
    try:
        import mcore_bridge
    except ImportError:
        return  # plugin is being imported on a non-Megatron host

    # mcore-bridge's public surface has shifted between releases. Check
    # whichever is available in your install:
    #   >= 1.2:   mcore_bridge.registry.register_bridge(model_type, cls)
    #   1.0-1.1:  mcore_bridge.GPTBridge subclass + manual dict entry in
    #             mcore_bridge.models._BRIDGE_REGISTRY
    # Pick one and adapt the import + registration call below.

    # TODO(user): import the right base class for the installed mcore-bridge
    # from mcore_bridge import GPTBridge
    # from mcore_bridge.registry import register_bridge

    # TODO(user): implement the LatentMoE-specific bridge. Outline:
    #
    # class Qwen35LatentMoeBridge(GPTBridge):
    #     model_type = 'qwen3_5_latentmoe'
    #
    #     @staticmethod
    #     def hf_to_mcore_config(hf_config):
    #         from megatron.core.transformer import TransformerConfig
    #         # map LatentMoE hf_config fields -> TransformerConfig, including:
    #         #   num_moe_experts, moe_ffn_hidden_size, moe_router_topk,
    #         #   num_query_groups (from num_key_value_heads),
    #         #   kv_channels (from head_dim), rotary_base (from rope_theta),
    #         #   mtp_num_layers (from mtp_num_hidden_layers).
    #         # LatentMoE-specific extras (no mcore TransformerConfig field):
    #         #   moe_latent_dim, moe_latent_factor, layer_types
    #         # Stash these on a custom subclass so the model_provider can read
    #         # them when building layers.
    #         raise NotImplementedError
    #
    #     def convert_weights_hf_to_mcore(self, hf_state_dict):
    #         # See the weight mapping in this file's header docstring.
    #         # Linear-attention layers (layer_types[i] == 'linear_attention')
    #         # have no mcore analogue — their params flow into a custom
    #         # submodule your model_provider injects.
    #         raise NotImplementedError
    #
    #     def convert_weights_mcore_to_hf(self, mcore_state_dict):
    #         raise NotImplementedError
    #
    #     def model_provider(self, pre_process=True, post_process=True):
    #         # Build a mcore GPTModel subclass where:
    #         #   - full_attention layers use standard SelfAttention
    #         #   - linear_attention layers use a ported Qwen3_5LinearAttention
    #         #     module (copy from v5molae_qwen35_latentmoe.py and wrap with
    #         #     mcore's TensorParallel region + sequence_parallel handling)
    #         #   - MoE layers use a LatentMoE subclass of mcore MoELayer with
    #         #     the latent down/up projections on the expert forward path
    #         #   - NPU fused paths: torch_npu.npu_rms_norm (if
    #         #     use_npu_rmsnorm) and torch_npu.npu_grouped_matmul (if
    #         #     use_grouped_expert_matmul) replace the default ops
    #         raise NotImplementedError
    #
    # register_bridge('qwen3_5_latentmoe', Qwen35LatentMoeBridge)

    # For now, fail loudly with a pointer so the user sees the next step:
    import logging
    logging.getLogger(__name__).warning(
        'gkd_plugin_mindspeed: Qwen35LatentMoeBridge is not implemented. '
        'Running `megatron rlhf` with --model_type qwen3_5_latentmoe will fail '
        'inside mcore-bridge config resolution. Implement the bridge in '
        '_register_mindspeed_bridge() or fall back to the DeepSpeed path '
        '(gkd_plugin.py + gkd_latentmoe_vllm_npu.sh).'
    )


_register_mindspeed_bridge()
