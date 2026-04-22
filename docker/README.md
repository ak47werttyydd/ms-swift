# GKD Training Docker Environment

Docker setup for Generalized Knowledge Distillation (GKD) of Qwen3.5 LatentMoE models using ms-swift.

## Prerequisites on target server

- 8x NVIDIA H100 80GB (or equivalent)
- NVIDIA Driver >= 590
- Docker with NVIDIA Container Toolkit (`nvidia-docker`)
- `/home` filesystem shared (model weights, source code, checkpoints)

## Quick start

```bash
# 1. Build image (~30 min, flash-attn CUDA compilation is slow)
cd /home/a84400789/ms-swift/docker
docker build -t daal/ms-swift:v0 .

# 2. Start container
docker compose up -d KD

# 3. Enter container
docker exec -it kd bash

# 4. Run GKD training
cd /home/a84400789/ms-swift

# Option A: both models sequentially (shared teacher, saves ~5 min)
STEPS_40L=100 STEPS_24L=200 bash gkd_latentmoe_both.sh

# Option B: individual scripts
bash gkd_latentmoe_vllm.sh      # 40-layer only
bash gkd_latentmoe_vllm_24l.sh  # 24-layer only
```

## Pinned versions

All versions are pinned to the working environment as of 2026-04-17:

| Package | Version | Notes |
|---------|---------|-------|
| Base image | `nvcr.io/nvidia/pytorch:26.02-py3` | torch 2.10.0, CUDA 12.8, Python 3.12 |
| transformers | 5.3.0 | Must be >=5.2.0 for Qwen3.5; installed AFTER vllm |
| vllm | 0.17.1 | Teacher serving; pins older transformers (overridden) |
| deepspeed | 0.18.8 | ZeRO-3 distributed training |
| flash-attn | 2.8.3 | CUDA kernel, ~20 min build |
| flash-linear-attention | 0.4.2 | Qwen3.5 linear attention |
| causal-conv1d | 1.6.1 | Gated Delta Net kernel |
| accelerate | 1.13.0 | Distributed training orchestration |
| peft | 0.18.1 | Parameter-efficient fine-tuning |
| liger-kernel | 0.7.0 | Fused JSD loss for GKD |
| safetensors | 0.7.0 | Checkpoint I/O |

## Install order matters

```
flash-attn → causal-conv1d → flash-linear-attention → vllm → deepspeed → transformers → rest
```

- `flash-attn` builds CUDA kernels against torch headers from the base image
- `vllm` pins an older `transformers`; we override it afterward
- `torchao` (installed by vllm) is incompatible with torch 2.10.0+cu128 and must be removed

## ms-swift

ms-swift is NOT baked into the image. It's installed editable at container startup
via `entrypoint.sh` from the host-mounted volume. This allows live code changes
without rebuilding the image.

To use a different ms-swift path, set `MS_SWIFT_DIR`:
```bash
MS_SWIFT_DIR=/home/other_user/ms-swift docker compose up -d KD
```

## GPU allocation

| GPUs | Role | Config |
|------|------|--------|
| 0-5 | Student training | DeepSpeed ZeRO-3, 6-way data parallel |
| 6-7 | Teacher server | vLLM TP=2 |

## Model checkpoints

Student checkpoints must be pre-packed (not legacy per-expert format).
Verify with:
```bash
python qwen35_latentmoe/repack_experts.py --check <checkpoint_dir>
python qwen35_latentmoe/check_weights.py <checkpoint_dir>
```

If legacy format detected, convert first:
```bash
python qwen35_latentmoe/repack_experts.py --convert <checkpoint_dir>
```

## Troubleshooting

**fabricmanager not running**: NVSwitch topology requires it.
```bash
sudo systemctl start nvidia-fabricmanager
```

**OOM during training**: Reduce `per_device_train_batch_size` and increase
`gradient_accumulation_steps` proportionally.

**Expert weights all zero**: Checkpoint was in legacy format. See "Model checkpoints" above.
