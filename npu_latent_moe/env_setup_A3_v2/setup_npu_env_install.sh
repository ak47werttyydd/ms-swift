#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Install-only: conda env + Python packages for the Qwen3.5 GKD NPU workflow.
#
# Sources CANN only for the vllm-ascend step (custom ACLNN ops require
# bisheng from CANN; all other steps are pure pip/conda).
# Run setup_npu_env_verify.sh afterwards to confirm the full NPU stack.
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
CANN_SETENV="${CANN_SETENV:-/home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh}"
NNAL_SETENV="${NNAL_SETENV:-/home/canada_group_account/a84400789/CANN8.5.1/nnal/atb/set_env.sh}"

VLLM_ASCEND_DIR="${VLLM_ASCEND_DIR:-/home/canada_group_account/a84400789/vllm-ascend}"
MS_SWIFT_DIR="${MS_SWIFT_DIR:-/home/canada_group_account/a84400789/ms-swift}"
VLLM_DIR="${VLLM_DIR:-/home/canada_group_account/a84400789/vllm}"
VLLM_ASCEND_URL="${VLLM_ASCEND_URL:-https://github.com/vllm-project/vllm-ascend}"

VLLM_ASCEND_BRANCH="${VLLM_ASCEND_BRANCH:-releases/v0.18.0}"
VLLM_TAG="${VLLM_TAG:-v0.18.0}"

# SOC_VERSION: required by vllm-ascend source install (CPU-only/empty build path).
# 910B1 = ascend910b1 (Atlas A2); 910B3/B4 = ascend910_9391 (Atlas A3).
SOC_VERSION="${SOC_VERSION:-ascend910_9391}"

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

# --- Source CANN (required for vllm-ascend custom op compilation) ---
# vllm-ascend's csrc/build_aclnn.sh needs bisheng (Huawei C++ compiler) and
# ASCEND_HOME_PATH, both of which are only available after sourcing set_env.sh.
[[ -f "$CANN_SETENV" ]] || die "CANN set_env.sh not found at $CANN_SETENV — required for vllm-ascend build."
# shellcheck disable=SC1090
source "$CANN_SETENV"
if [[ -f "$NNAL_SETENV" ]]; then
    set +u  # Huawei set_env.sh reads $ZSH_VERSION which is unset in bash
    # shellcheck disable=SC1090
    source "$NNAL_SETENV"
    set -u
fi
export SOC_VERSION

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
            "deepspeed==0.18.8" "trl>=0.15,<0.30" "peft==0.18.1" \
            "transformers==5.3.0" "accelerate==1.13.0" \
            "safetensors==0.7.0" "qwen_vl_utils>=0.0.14" \
            "tensorboard" "swanlab" \
            "modelscope>=1.23" "datasets>=3.0,<4.0" \
            "math_verify"
# Versions mirror the GPU GKD docker (docker/Dockerfile in latentMoE branch).
# transformers==5.3.0 is required for Qwen3.5 LatentMoE (>=5.2 minimum); it is
# installed AFTER vllm/vllm-ascend so it overrides any older pin they bring.
# CUDA/Triton-only kernels are excluded on NPU: flash-attn, flash-linear-attention,
# causal-conv1d, liger_kernel; NPU uses --attn_impl sdpa via torch-npu.

# Remove torchao if vllm pulled an incompatible dev build (matches GPU docker
# step; the dev wheel references torch ops missing from torch 2.9.0+).
pip uninstall -y torchao 2>/dev/null || true

# --- setuptools downgrade ---
# setuptools >= 70 drops the bundled pkg_resources; many transitive deps
# (deepspeed, mindspeed, older trl helpers) still `import pkg_resources`.
# Pin to 68.x so import succeeds. Run last so nothing upgrades it back.
log "Pinning setuptools==68.0 to keep pkg_resources available"
pip install "setuptools==68.0"

log "Install complete. Run setup_npu_env_verify.sh (with CANN sourced) to validate."
