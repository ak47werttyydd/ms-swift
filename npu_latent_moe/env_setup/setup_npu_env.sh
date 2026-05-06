#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Conda env setup for Ascend NPU:
#   - Serve Qwen3.5-35B-A3B as GKD teacher via vllm-ascend (releases/v0.18.0)
#   - Train a smaller Qwen3.5 student with ms-swift GKD RLHF
#
# Run this ON the NPU host (NOT on macOS). It is written for Atlas 800/900
# A2 (910B/C) with CANN 8.5.1 already installed at /usr/local/Ascend.
#
# Usage:
#   bash scripts/setup_npu_env.sh
#
# Version matrix (reconciles ms-swift NPU docs + vllm-ascend releases/v0.18.0):
#   Python            3.11
#   CANN              8.5.1                  (required by both)
#   torch / torch-npu 2.9.0 / 2.9.0          (from vllm-ascend releases/v0.18.0
#                                             requirements.txt; overrides the
#                                             2.7.1 pin in ms-swift docs)
#   transformers      >= 4.57.4              (vllm-ascend) — compatible with
#                                             ms-swift's `<5.6` upper bound
#   vllm              v0.18.0 source         (matches vllm-ascend v0.18.0rc1)
#   vllm-ascend       releases/v0.18.0       (brings Qwen3.5 + MoE flashcomm
#                                             v1 + MTP shared-expert fix)
#   ms-swift          local source (4.2.0.dev0)
# -----------------------------------------------------------------------------
set -euo pipefail

# --------------------------- User-tunable paths ------------------------------
ENV_NAME="${ENV_NAME:-ms-swift}"
PY_VERSION="${PY_VERSION:-3.11}"
CANN_SETENV="${CANN_SETENV:-/home/w00498690/gdn_post_train/CANN8.5.1/cann-8.5.1/set_env.sh}"
NNAL_SETENV="${NNAL_SETENV:-/home/w00498690/gdn_post_train/CANN8.5.1/nnal/atb/set_env.sh}"

# Repo checkouts (override via env if you keep them elsewhere)
VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/home/w00498690/gdn_post_train/vllm-ascend}"
MS_SWIFT_DIR="${MS_SWIFT_DIR:-/home/w00498690/gdn_post_train/ms-swift}"
VLLM_DIR="${VLLM_DIR:-/home/w00498690/gdn_post_train/vllm}"
VLLM_ASCEND_URL="${VLLM_ASCEND_URL:-https://github.com/cosdt/vllm-ascend}"

VLLM_ASCEND_BRANCH="${VLLM_ASCEND_BRANCH:-releases/v0.18.0}"
VLLM_TAG="${VLLM_TAG:-v0.18.0}"   # matches vllm-ascend releases/v0.18.0 CI

# SOC_VERSION: required at runtime and by vllm-ascend source install.
# 910B1 = ascend910b1 (Atlas A2); 910B3/B4 = ascend910_9391 (Atlas A3).
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

# Optional: Chinese mirrors (set USE_CN_MIRROR=1 if you're in mainland China)
USE_CN_MIRROR="${USE_CN_MIRROR:-0}"

# -----------------------------------------------------------------------------
log() { printf '\033[1;36m[setup]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup:error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Sanity: must run on Linux with NPU ---
[[ "$(uname -s)" == "Linux" ]] || die "Must run on the NPU host (Linux), not macOS."
command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found — install Ascend driver + CANN first."
[[ -f "$CANN_SETENV" ]] || die "CANN set_env.sh not found at $CANN_SETENV (override via CANN_SETENV=...)."
command -v conda >/dev/null 2>&1 || die "conda not found in PATH."

log "Sourcing CANN toolkit: $CANN_SETENV"
# shellcheck disable=SC1090
source "$CANN_SETENV"
if [[ -f "$NNAL_SETENV" ]]; then
    # shellcheck disable=SC1090
    source "$NNAL_SETENV"
else
    log "NNAL set_env.sh not found at $NNAL_SETENV — libatb.so features may be unavailable."
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

# --- torch + torch-npu (must precede vllm / vllm-ascend install) ---
# Install torch-npu from Ascend mirror; it pulls the aarch64-compatible torch 2.9.0.
# Do NOT install torch from pytorch.org — those wheels are x86_64 only.
log "Installing torch-npu==2.9.0 (pulls aarch64 torch 2.9.0)"
pip install "torch-npu==2.9.0" decorator \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

# --- vllm (matched tag), source install with NPU-skip target ---
if [[ ! -d "$VLLM_DIR/.git" ]]; then
    log "Cloning vllm @ $VLLM_TAG → $VLLM_DIR"
    git clone --branch "$VLLM_TAG" --single-branch --depth 1 https://github.com/vllm-project/vllm "$VLLM_DIR"
else
    log "Using existing vllm checkout at $VLLM_DIR (expecting tag $VLLM_TAG)"
fi
(
    cd "$VLLM_DIR"
    if pip show vllm 2>/dev/null | grep -q "^Version: ${VLLM_TAG#v}"; then
        log "vllm ${VLLM_TAG#v} already installed — skipping pip install"
    else
        VLLM_TARGET_DEVICE=empty pip install -v -e .
    fi
)

# --- vllm-ascend ---
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

# --- ms-swift from local checkout (main, 4.2.0.dev0 or pinned tag) ---
[[ -d "$MS_SWIFT_DIR/.git" ]] || die "ms-swift checkout not found at $MS_SWIFT_DIR"
(
    cd "$MS_SWIFT_DIR"
    log "Installing ms-swift from $MS_SWIFT_DIR"
    pip install -e .
)

# --- Extras for training + GKD ---
log "Installing deepspeed and ms-swift extras"
pip install "torch==2.9.0" \
            "deepspeed<0.19" "trl>=0.15,<0.30" "peft>=0.11,<0.19" \
            "transformers>=4.57.4,<5.6" "accelerate" \
            "tensorboard" "swanlab" \
            "modelscope>=1.23" "datasets>=3.0,<4.0" \
            "math_verify"
# flash-attn is x86-CUDA-only; NPU uses --attn_impl sdpa via torch-npu.
# liger_kernel (CUDA-only) and nvitop (NVIDIA monitoring) are excluded.

# --- Verification ---
log "Verifying NPU + vllm + ms-swift install"
python - <<'PY'
import torch
from transformers.utils import is_torch_npu_available
print("torch:", torch.__version__)
print("torch_npu available:", is_torch_npu_available())
if is_torch_npu_available():
    import torch_npu  # noqa: F401
    print("NPU count:", torch.npu.device_count())
    print("test tensor:", torch.randn(3, device="npu:0"))
import vllm, vllm_ascend, swift
print("vllm:", vllm.__version__)
print("vllm_ascend:", getattr(vllm_ascend, "__version__", "editable"))
print("ms-swift:", swift.__version__)
PY

cat <<'EOS'

[setup] Done.

Next steps (run in two shells on the NPU host):

  # Shell 1 — teacher server (tensor parallel = 4 NPUs for 35B-A3B)
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  conda activate ms-swift
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
  vllm serve Qwen/Qwen3.5-35B-A3B \
      --tensor-parallel-size 4 \
      --port 8000 \
      --max-logprobs 64 \
      --gpu-memory-utilization 0.9 \
      --max-model-len 4096 \
      --trust-remote-code

  # Shell 2 — GKD student training against the teacher server
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  conda activate ms-swift
  NPROC_PER_NODE=4 \
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
  swift rlhf \
      --rlhf_type gkd \
      --model Qwen/Qwen3.5-4B \
      --teacher_model_server http://localhost:8000 \
      --gkd_logits_topk 64 \
      --dataset 'AI-ModelScope/alpaca-gpt4-data-en#2000' 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
      --split_dataset_ratio 0.01 \
      --lmbda 0.5 --seq_kd false --beta 0.5 \
      --torch_dtype bfloat16 \
      --num_train_epochs 1 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-5 \
      --max_length 2048 --max_completion_length 512 \
      --deepspeed zero3 \
      --attn_impl sdpa \
      --output_dir output/qwen3_5-gkd \
      --save_steps 100 --save_total_limit 2 \
      --logging_steps 5 \
      --report_to tensorboard

Notes:
  * The script pins torch 2.9.0 (required by vllm-ascend releases/v0.18.0),
    overriding the older 2.7.1 pin in ms-swift's NPU docs.
  * Use --attn_impl sdpa, NOT flash_attn — flash_attn is CUDA-only.
  * 35B-A3B is MoE; Qwen3.5 MoE flashcomm v1 + the #8004 MTP shared-expert
    shape fix are on releases/v0.18.0, which is why we pin this branch.
EOS
