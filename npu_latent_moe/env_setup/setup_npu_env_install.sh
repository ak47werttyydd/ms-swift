#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Install-only: conda env + Python packages for the Qwen3.5 GKD NPU workflow.
#
# Does NOT source CANN — all steps are pure pip/conda and work without CANN
# in the current shell. Run setup_npu_env_verify.sh afterwards (with CANN
# sourced) to confirm the NPU stack is functional.
#
# Usage:
#   bash npu_latent_moe/env_setup/setup_npu_env_install.sh
#
# Version matrix:
#   Python            3.11
#   torch / torch-npu 2.9.0 / 2.9.0  (vllm-ascend v0.18.0 hard requirement)
#   vllm              v0.18.0 source, VLLM_TARGET_DEVICE=empty
#   vllm-ascend       releases/v0.18.0
#   ms-swift          local source
# -----------------------------------------------------------------------------
set -euo pipefail

# --------------------------- User-tunable paths ------------------------------
ENV_NAME="${ENV_NAME:-ms-swift}"
PY_VERSION="${PY_VERSION:-3.11}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/home/w00498690/gdn_post_train/vllm-ascend}"
MS_SWIFT_DIR="${MS_SWIFT_DIR:-/home/w00498690/gdn_post_train/ms-swift}"
VLLM_DIR="${VLLM_DIR:-/home/w00498690/gdn_post_train/vllm}"
VLLM_ASCEND_URL="${VLLM_ASCEND_URL:-https://github.com/cosdt/vllm-ascend}"

VLLM_ASCEND_BRANCH="${VLLM_ASCEND_BRANCH:-releases/v0.18.0}"
VLLM_TAG="${VLLM_TAG:-v0.18.0}"

# SOC_VERSION: required by vllm-ascend source install (CPU-only/empty build path).
# 910B1 = ascend910b1 (Atlas A2); 910B3/B4 = ascend910_9391 (Atlas A3).
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

USE_CN_MIRROR="${USE_CN_MIRROR:-0}"

# -----------------------------------------------------------------------------
log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[install:error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Sanity checks (no CANN needed here) ---
[[ "$(uname -s)" == "Linux" ]] || die "Must run on the NPU host (Linux), not macOS."
command -v npu-smi >/dev/null 2>&1 || die "npu-smi not found — install Ascend driver first."
command -v conda   >/dev/null 2>&1 || die "conda not found in PATH."
command -v git     >/dev/null 2>&1 || die "git not found in PATH."
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

# --- torch + torch-npu ---
# torch-npu is on Huawei's Ascend mirror (not PyPI). Use --extra-index-url so
# pip still resolves other packages from PyPI; do NOT use -i (replaces PyPI entirely).
# torch 2.9.0 (aarch64) is pulled automatically as a torch-npu dependency.
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
        SOC_VERSION="$SOC_VERSION" pip install -v -e .
    fi
)

# --- ms-swift ---
[[ -d "$MS_SWIFT_DIR/.git" ]] || die "ms-swift checkout not found at $MS_SWIFT_DIR"
(
    cd "$MS_SWIFT_DIR"
    log "Installing ms-swift from $MS_SWIFT_DIR"
    pip install -e .
)

# --- Training extras ---
log "Installing deepspeed and ms-swift extras"
pip install "torch==2.9.0" \
            "deepspeed<0.19" "trl>=0.15,<0.30" "peft>=0.11,<0.19" \
            "transformers>=4.57.4,<5.6" "accelerate" \
            "tensorboard" "swanlab" \
            "modelscope>=1.23" "datasets>=3.0,<4.0" \
            "math_verify"
# flash-attn is x86-CUDA-only; NPU uses --attn_impl sdpa via torch-npu.
# liger_kernel (CUDA-only) and nvitop (NVIDIA monitoring) are excluded.

log "Install complete. Run setup_npu_env_verify.sh (with CANN sourced) to validate."
