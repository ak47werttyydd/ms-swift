#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Conda env setup for Ascend NPU — MindSpeed (Megatron-LM) variant.
#
#   * Teacher: Qwen3.5-35B-A3B served by vllm-ascend releases/v0.18.0
#     (same as the DeepSpeed variant — vLLM teacher is backend-agnostic)
#   * Student: trained via `megatron rlhf` (MindSpeed + mcore-bridge),
#     using TP + EP + PP instead of DeepSpeed Zero3.
#
# Differences vs setup_npu_env.sh (DeepSpeed variant):
#   + Megatron-LM v0.15.3 cloned to $MEGATRON_LM_REPO
#   + MindSpeed core_r0.15.3 installed (Ascend kernel replacements for Megatron)
#   + mcore-bridge installed (HF <-> mcore converter, powers `megatron rlhf`)
#   + apex-ascend (fused optimizer/layernorm used by Megatron)
#   + env vars MEGATRON_LM_PATH + PYTHONPATH wired through activate.d/
#   * deepspeed is still installed as fallback but not required.
#
# Source of truth: ms-swift/docs/source_en/BestPractices/NPU-support.md
#   (Megatron-LM v0.15.3 + MindSpeed core_r0.15.3 — version-matched!)
#
# Run this ON the NPU host. Usage:
#   bash scripts/setup_npu_env_mindspeed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# --------------------------- User-tunable paths ------------------------------
ENV_NAME="${ENV_NAME:-ms-swift-mindspeed}"
PY_VERSION="${PY_VERSION:-3.11}"
CANN_SETENV="${CANN_SETENV:-/home/w00498690/gdn_post_train/CANN8.5.1/cann-8.5.1/set_env.sh}"
NNAL_SETENV="${NNAL_SETENV:-/home/w00498690/gdn_post_train/CANN8.5.1/nnal/atb/set_env.sh}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/home/w00498690/gdn_post_train/vllm-ascend}"
MS_SWIFT_DIR="${MS_SWIFT_DIR:-/home/w00498690/gdn_post_train/ms-swift}"
VLLM_DIR="${VLLM_DIR:-/home/w00498690/gdn_post_train/vllm}"
VLLM_ASCEND_URL="${VLLM_ASCEND_URL:-https://github.com/cosdt/vllm-ascend}"

# MindSpeed stack
MEGATRON_LM_REPO="${MEGATRON_LM_REPO:-/home/w00498690/gdn_post_train/Megatron-LM}"
MEGATRON_LM_TAG="${MEGATRON_LM_TAG:-v0.15.3}"
MINDSPEED_REPO="${MINDSPEED_REPO:-/home/w00498690/gdn_post_train/MindSpeed}"
MINDSPEED_BRANCH="${MINDSPEED_BRANCH:-core_r0.15.3}"
MCORE_BRIDGE_REPO="${MCORE_BRIDGE_REPO:-/home/w00498690/gdn_post_train/mcore-bridge}"

VLLM_ASCEND_BRANCH="${VLLM_ASCEND_BRANCH:-releases/v0.18.0}"
VLLM_TAG="${VLLM_TAG:-v0.18.0}"

# SOC_VERSION: required at runtime and by vllm-ascend source install.
# 910B1 = ascend910b1 (Atlas A2); 910B3/B4 = ascend910_9391 (Atlas A3).
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

USE_CN_MIRROR="${USE_CN_MIRROR:-0}"

# -----------------------------------------------------------------------------
log() { printf '\033[1;36m[setup-mindspeed]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup-mindspeed:error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Sanity: must run on Linux with NPU ---
[[ "$(uname -s)" == "Linux" ]] || die "Must run on the NPU host (Linux), not macOS."
command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found — install Ascend driver + CANN first."
[[ -f "$CANN_SETENV" ]] || die "CANN set_env.sh not found at $CANN_SETENV."
command -v conda >/dev/null 2>&1 || die "conda not found in PATH."
command -v git   >/dev/null 2>&1 || die "git not found in PATH."

log "Sourcing CANN toolkit: $CANN_SETENV"
# shellcheck disable=SC1090
source "$CANN_SETENV"
if [[ -f "$NNAL_SETENV" ]]; then
    set +u  # Huawei set_env.sh reads $ZSH_VERSION which is unset in bash
    # shellcheck disable=SC1090
    source "$NNAL_SETENV"
    set -u
fi
export SOC_VERSION
export GIT_SSL_NO_VERIFY=true

# --- conda env ---
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    log "Conda env '$ENV_NAME' already exists — reusing."
else
    log "Creating conda env '$ENV_NAME' (python=$PY_VERSION)"
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi
conda activate "$ENV_NAME"
python -V

# --- pip config ---
python -m pip install --upgrade pip setuptools wheel
if [[ "$USE_CN_MIRROR" == "1" ]]; then
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
fi

# --- torch + torch-npu (must precede everything that builds against torch) ---
# Install torch-npu from Ascend mirror; it pulls the aarch64-compatible torch 2.9.0.
# Do NOT install torch from pytorch.org — those wheels are x86_64 only.
log "Installing torch-npu==2.9.0 (pulls aarch64 torch 2.9.0)"
pip install "torch-npu==2.9.0" decorator \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

# --- apex-ascend (Megatron fused optimizer / fused layernorm) ---
# The Ascend fork of NVIDIA/apex. Install from Huawei's pypi mirror.
# If the wheel name changes across CANN versions, adjust or drop — Megatron
# will fall back to Python implementations (slower) when apex is missing.
log "Installing apex-ascend (optional but recommended for Megatron perf)"
pip install apex-ascend \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    || log "apex-ascend install failed — continuing without fused apex ops."

# --- vLLM (matched tag), teacher-side; NPU-skip target ---
if [[ ! -d "$VLLM_DIR/.git" ]]; then
    log "Cloning vllm @ $VLLM_TAG → $VLLM_DIR"
    git clone --branch "$VLLM_TAG" --single-branch --depth 1 https://github.com/vllm-project/vllm "$VLLM_DIR"
fi
(
    cd "$VLLM_DIR"
    if pip show vllm 2>/dev/null | grep -q "^Version: ${VLLM_TAG#v}"; then
        log "vllm ${VLLM_TAG#v} already installed — skipping pip install"
    else
        VLLM_TARGET_DEVICE=empty pip install -v -e .
    fi
)

# --- vllm-ascend (teacher inference backend) ---
if [[ ! -d "$VLLM_ASCEND_DIR/.git" ]]; then
    log "Cloning vllm-ascend @ $VLLM_ASCEND_BRANCH → $VLLM_ASCEND_DIR"
    git clone --branch "$VLLM_ASCEND_BRANCH" --single-branch --depth 1 "$VLLM_ASCEND_URL" "$VLLM_ASCEND_DIR"
fi
(
    cd "$VLLM_ASCEND_DIR"
    _vllm_ascend_ver="${VLLM_ASCEND_BRANCH##*v}"
    if pip show vllm-ascend 2>/dev/null | grep -q "^Version: ${_vllm_ascend_ver}"; then
        log "vllm-ascend ${_vllm_ascend_ver} already installed — skipping"
    else
        current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
        if [[ "$current_branch" != "$VLLM_ASCEND_BRANCH" ]]; then
            log "Checking out vllm-ascend branch $VLLM_ASCEND_BRANCH"
            git fetch --all --tags
            git checkout "$VLLM_ASCEND_BRANCH"
        fi
        git submodule update --init --recursive
        pip install -v -e .
    fi
)

# --- Megatron-LM (v0.15.3, matched with MindSpeed core_r0.15.3) ---
if [[ ! -d "$MEGATRON_LM_REPO/.git" ]]; then
    log "Cloning Megatron-LM @ $MEGATRON_LM_TAG → $MEGATRON_LM_REPO"
    git clone https://github.com/NVIDIA/Megatron-LM.git "$MEGATRON_LM_REPO"
fi
(
    cd "$MEGATRON_LM_REPO"
    git fetch --all --tags
    git checkout "$MEGATRON_LM_TAG"
    # Megatron-LM is imported via PYTHONPATH, NOT pip-installed —
    # mcore-bridge patches internals, so we need the source tree on disk.
)

# --- MindSpeed (core_r0.15.3 — Ascend kernel replacements for Megatron) ---
if [[ ! -d "$MINDSPEED_REPO/.git" ]]; then
    log "Cloning MindSpeed @ $MINDSPEED_BRANCH → $MINDSPEED_REPO"
    # Huawei's public gitcode mirror; gitee.com/ascend/MindSpeed is the other URL.
    git clone https://gitcode.com/Ascend/MindSpeed.git "$MINDSPEED_REPO"
fi
(
    cd "$MINDSPEED_REPO"
    git fetch --all --tags
    git checkout "$MINDSPEED_BRANCH"
    pip install -e .
)

# --- mcore-bridge (HF <-> mcore converter, drives `megatron rlhf`) ---
if [[ ! -d "$MCORE_BRIDGE_REPO/.git" ]]; then
    log "Cloning mcore-bridge → $MCORE_BRIDGE_REPO"
    git clone https://github.com/modelscope/mcore-bridge.git "$MCORE_BRIDGE_REPO"
fi
(
    cd "$MCORE_BRIDGE_REPO"
    pip install -e .
)

# --- ms-swift from local checkout (must come after mcore-bridge so the
#     megatron extras resolve correctly) ---
[[ -d "$MS_SWIFT_DIR/.git" ]] || die "ms-swift checkout not found at $MS_SWIFT_DIR"
(
    cd "$MS_SWIFT_DIR"
    log "Installing ms-swift from $MS_SWIFT_DIR"
    pip install -e .
    pip install -r requirements/megatron.txt
)

# --- Extras for GKD training ---
log "Installing training extras"
pip install "torch==2.9.0" \
            "deepspeed<0.19" "trl>=0.15,<0.30" "peft>=0.15,<0.19" \
            "transformers>=4.57.4,<5.6" "accelerate" \
            "tensorboard" "swanlab" \
            "modelscope>=1.23" "datasets>=3.0,<4.0" \
            "math_verify"
# Note: flash-attn (Dao-AILab) is NOT installed. Megatron on NPU routes
# `--attention_backend flash` to torch_npu.npu_fusion_attention via MindSpeed.

# --- Persist env vars into conda activate.d so every new shell has them ---
ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_D"
cat > "$ACTIVATE_D/mindspeed_env.sh" <<EOF
# Auto-generated by setup_npu_env_mindspeed.sh
export MEGATRON_LM_PATH="$MEGATRON_LM_REPO"
export PYTHONPATH="\$MEGATRON_LM_PATH:\${PYTHONPATH:-}"
# CANN needs to be sourced explicitly by the caller; we only export the path.
export CANN_SETENV="$CANN_SETENV"
EOF
log "Wrote $ACTIVATE_D/mindspeed_env.sh — MEGATRON_LM_PATH + PYTHONPATH on activate"

# Re-source to pick them up inside this shell too
# shellcheck disable=SC1090
source "$ACTIVATE_D/mindspeed_env.sh"

# --- Verification ---
log "Verifying NPU + Megatron + MindSpeed + mcore-bridge"
python - <<'PY'
import torch
from transformers.utils import is_torch_npu_available
print("torch:", torch.__version__)
print("torch_npu available:", is_torch_npu_available())
if is_torch_npu_available():
    import torch_npu  # noqa: F401
    print("NPU count:", torch.npu.device_count())

import mindspeed.megatron_adaptor  # noqa: F401
print("mindspeed.megatron_adaptor: OK")

import megatron.core as mcore
print("megatron.core:", mcore.__version__)

import mcore_bridge
print("mcore_bridge:", mcore_bridge.__version__)

from swift.megatron.init import init_megatron_env
init_megatron_env()
print("swift.megatron.init_megatron_env: OK")

import vllm, vllm_ascend, swift
print("vllm:", vllm.__version__)
print("vllm_ascend:", getattr(vllm_ascend, "__version__", "editable"))
print("ms-swift:", swift.__version__)
PY

cat <<'EOS'

[setup-mindspeed] Done.

Next steps (run in two shells on the NPU host):

  # Shell 1 — teacher (vLLM, same as the DeepSpeed path)
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  conda activate ms-swift-mindspeed
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  vllm serve Qwen/Qwen3.5-35B-A3B \
      --tensor-parallel-size 4 --port 8000 \
      --max-logprobs 64 --max-model-len 4610 \
      --dtype bfloat16 --enforce-eager --trust-remote-code

  # Shell 2 — student training via Megatron (NOT `swift rlhf`; use `megatron rlhf`)
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  conda activate ms-swift-mindspeed
  NPROC_PER_NODE=4 \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  megatron rlhf \
      --rlhf_type gkd \
      --model Qwen/Qwen3.5-4B \
      --teacher_model_server http://localhost:8000 \
      --gkd_logits_topk 64 \
      --dataset '/data/nemotron-cc' \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 1 \
      --expert_model_parallel_size 1 \
      --attention_backend flash \
      --torch_dtype bfloat16 \
      --micro_batch_size 2 --global_batch_size 32 \
      --train_iters 1000 --lr 1e-5 \
      --max_length 4607 --max_completion_length 1 \
      --finetune --no_save_optim --no_save_rng \
      --save output/qwen3_5-gkd-mindspeed

Notes:
  * `megatron` is the CLI entry-point installed by ms-swift; do NOT use
    `swift megatron rlhf` (that spelling does not exist).
  * Megatron uses step-driven training: --train_iters + --global_batch_size
    replace --num_train_epochs + per_device_batch_size + grad_accum.
  * For custom `trust_remote_code` models (e.g. Qwen3.5-LatentMoE),
    mcore-bridge has NO built-in converter — you need a plugin that either
    registers one via mcore-bridge's API or maps the arch to a supported
    Megatron model_type. See gkd_plugin_mindspeed.py for the scaffold.
EOS
