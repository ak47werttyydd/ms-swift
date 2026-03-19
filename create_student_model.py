"""
Create a custom Qwen3.5-0.8B-A0.1B student model (MoE, ~0.8B total / ~0.1B active params).

Architecture design (language part only):
  hidden_size=1024, num_hidden_layers=16, head_dim=64
  num_experts=16, num_experts_per_tok=2
  moe_intermediate_size=512, shared_expert_intermediate_size=512

Rough param estimate:
  Attention per layer:  ~2.6M
  Shared expert:        ~1.6M
  2 active experts:     ~3.1M  (active per layer ~7.3M → 16 layers → ~0.12B active)
  16 total experts:     ~25.2M (total per layer ~29.4M → 16 layers → ~0.47B)
  Embedding (248320*1024): ~0.25B
  Grand total: ~0.72B ≈ 0.8B  /  Active LM: ~0.12B ≈ 0.1B

Vision config is copied unchanged from the base Qwen3.5-0.8B.
Weights are randomly initialised — distillation will train them from scratch.

Usage:
    python create_student_model.py \
        --base_model Qwen/Qwen3.5-0.8B \
        --output_dir ./Qwen3_5-0.8B-A0.1B-student
"""

import argparse
import os

from transformers import AutoProcessor, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='Qwen/Qwen3.5-0.8B',
                        help='Source model to borrow vision config + tokenizer from')
    parser.add_argument('--output_dir', default='./Qwen3_5-0.8B-A0.1B-student')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load base config & tokenizer ──────────────────────────────────────
    print(f'Loading base config from {args.base_model} ...')
    base_cfg = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    # ── 2. Build the MoE text config ─────────────────────────────────────────
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeConfig,
    )

    text_cfg = Qwen3_5MoeTextConfig(
        vocab_size=getattr(base_cfg, 'vocab_size', 248320),
        hidden_size=1024,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=64,
        hidden_act='silu',
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        # MoE
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=16,
        num_experts_per_tok=2,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
    )

    # ── 3. Reuse vision config from the base model ───────────────────────────
    vision_cfg = getattr(base_cfg, 'vision_config', None)
    if vision_cfg is None:
        raise RuntimeError('Could not find vision_config in base model config.')

    # ── 4. Assemble the full VLM config ──────────────────────────────────────
    moe_cfg = Qwen3_5MoeConfig(
        text_config=text_cfg.to_dict(),
        vision_config=vision_cfg.to_dict() if hasattr(vision_cfg, 'to_dict') else vision_cfg,
        image_token_id=getattr(base_cfg, 'image_token_id', 248056),
        video_token_id=getattr(base_cfg, 'video_token_id', 248057),
        vision_start_token_id=getattr(base_cfg, 'vision_start_token_id', 248053),
        vision_end_token_id=getattr(base_cfg, 'vision_end_token_id', 248054),
    )

    # ── 5. Randomly-initialised model ────────────────────────────────────────
    print('Building randomly-initialised student model ...')
    from transformers import Qwen3_5MoeForConditionalGeneration
    model = Qwen3_5MoeForConditionalGeneration(moe_cfg)

    total = sum(p.numel() for p in model.parameters()) / 1e9
    active = sum(
        p.numel() for name, p in model.named_parameters()
        if 'experts' not in name or any(f'experts.{i}.' in name for i in range(2))
    ) / 1e9
    print(f'Total params: {total:.3f}B')
    print(f'Approx active params (2/{text_cfg.num_experts} experts): {active:.3f}B')

    # ── 6. Save ───────────────────────────────────────────────────────────────
    print(f'Saving to {args.output_dir} ...')
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print('Done.')
    print(f'\nNow use in training:\n'
          f'  --model {args.output_dir} \\\n'
          f'  --model_type qwen3_5_moe')


if __name__ == '__main__':
    main()
