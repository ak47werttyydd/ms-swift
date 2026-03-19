# ============================================================================
# GKD Training Image: Qwen3.5-35B-A3B (teacher) -> pruned 10.4B-A2.4B (student)
# Base: NVIDIA NGC PyTorch 26.02-py3
#
# Build:
#   docker build -t daal/ms-swift:v0 .
# ============================================================================
FROM nvcr.io/nvidia/pytorch:26.02-py3

LABEL maintainer="adrian" \
      description="ms-swift GKD with vLLM teacher for Qwen3.5 MoE distillation"

# For apt-get and pip to run non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=16

# ModelScope env
ENV MODELSCOPE_CACHE="/dev/shm/ms_cache/"
ENV HF_HOME="/dev/shm/HF_HOME/"

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl htop tmux tree \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (order: heavy builds first, light installs last) ─────────────

# 1. flash-attn (long build, pin version, needs torch headers)
RUN pip install "flash-attn==2.8.3" --no-build-isolation

# 2. causal-conv1d (build from source)
RUN pip install -U git+https://github.com/Dao-AILab/causal-conv1d --no-build-isolation

# 3. flash-linear-attention (fla main branch)
#    If training is slow, see: https://github.com/fla-org/flash-linear-attention/issues/758
RUN pip install -U git+https://github.com/fla-org/flash-linear-attention

# 4. vLLM (for teacher serving + student on-policy rollout)
RUN pip install -U "vllm>=0.17.0"

# 5. deepspeed
RUN pip install -U deepspeed

# 6. transformers (must come AFTER vllm to override vllm's pinned version)
#    See: https://github.com/modelscope/ms-swift/issues/8188
RUN pip install -U "transformers>=5.3.0"

# 7. Remaining light deps
RUN pip install -U \
        "qwen_vl_utils>=0.0.14" \
        peft \
        liger-kernel

# 8. Install UV
RUN pip install uv

# ── ms-swift ─────────────────────────────────────────────────────────────────
# Installed editable at container startup via entrypoint.sh.
# Mount your local source at runtime:
#   docker run -v /home/a84400789/ms-swift:/home/a84400789/ms-swift ...

# ── Workspace ────────────────────────────────────────────────────────────────
WORKDIR /home

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]