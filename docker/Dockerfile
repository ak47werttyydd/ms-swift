# ============================================================================
# GKD Training Image: Qwen3.5-35B-A3B (teacher) -> LatentMoE student
# Base: NVIDIA NGC PyTorch 26.02-py3 (torch 2.10.0, CUDA 12.8, Python 3.12)
#
# Build:
#   docker build -t daal/ms-swift:v0 .
#
# Key versions (pinned to match working environment 2026-04-17):
#   torch 2.10.0+cu128  (from base image, DO NOT pip install torch)
#   transformers 5.3.0  (Qwen3.5 support, installed AFTER vllm)
#   vllm 0.17.1         (teacher server)
#   deepspeed 0.18.8    (ZeRO-3 distributed training)
#   flash-attn 2.8.3    (fast attention kernels)
# ============================================================================
FROM nvcr.io/nvidia/pytorch:26.02-py3

LABEL maintainer="adrian" \
      description="ms-swift GKD with vLLM teacher for Qwen3.5 LatentMoE distillation"

ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=16

# ModelScope / HuggingFace cache on ramdisk for fast loading
ENV MODELSCOPE_CACHE="/dev/shm/ms_cache/"
ENV HF_HOME="/dev/shm/HF_HOME/"

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl htop tmux tree \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (ORDER MATTERS — heavy builds first, version overrides last) ─

# 1. flash-attn (long CUDA build, ~20 min; needs torch headers from base image)
RUN pip install "flash-attn==2.8.3" --no-build-isolation

# 2. causal-conv1d (CUDA kernel for Qwen3.5 Gated Delta Net linear attention)
RUN pip install "causal-conv1d==1.6.1" --no-build-isolation

# 3. flash-linear-attention (linear attention kernels for Qwen3.5)
RUN pip install "flash-linear-attention==0.4.2"

# 4. vLLM (teacher serving + student on-policy rollout)
#    Pins its own transformers version — we override in step 6.
RUN pip install "vllm==0.17.1"

# 5. deepspeed (ZeRO-3 for distributed training across 6-8 H100s)
RUN pip install "deepspeed==0.18.8"

# 6. transformers (MUST come AFTER vllm to override vllm's pinned version)
#    Qwen3.5 models require >=5.2.0; our custom LatentMoeConfig needs 5.3.0+
#    See: https://github.com/modelscope/ms-swift/issues/8188
RUN pip install "transformers==5.3.0"

# 7. Remaining deps
RUN pip install \
        "accelerate==1.13.0" \
        "peft==0.18.1" \
        "liger-kernel==0.7.0" \
        "safetensors==0.7.0" \
        "qwen_vl_utils>=0.0.14"

# 8. uv (fast package manager for ad-hoc installs)
RUN pip install uv

# 9. Remove torchao (vllm installs a dev build incompatible with torch 2.10.0+cu128)
#    torchao references torch.ops._c10d_functional._wrap_tensor_autograd which doesn't exist.
RUN pip uninstall -y torchao || true

# ── ms-swift ─────────────────────────────────────────────────────────────────
# Installed editable at container startup via entrypoint.sh.
# Mount your local source at runtime:
#   -v /home/a84400789/ms-swift:/home/a84400789/ms-swift

# ── Workspace ────────────────────────────────────────────────────────────────
WORKDIR /home

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
