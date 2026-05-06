#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Verify the Qwen3.5 GKD NPU conda env installed by setup_npu_env_install.sh.
#
# Must be run AFTER sourcing CANN (torch_npu dlopen requires libascendcl.so):
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash npu_latent_moe/env_setup/setup_npu_env_verify.sh
# -----------------------------------------------------------------------------
set -euo pipefail

ENV_NAME="${ENV_NAME:-ms-swift}"
CANN_SETENV="${CANN_SETENV:-/home/w00498690/gdn_post_train/CANN8.5.1/cann-8.5.1/set_env.sh}"
NNAL_SETENV="${NNAL_SETENV:-/home/w00498690/gdn_post_train/CANN8.5.1/nnal/atb/set_env.sh}"
# 910B1 = ascend910b1 (Atlas A2); 910B3/B4 = ascend910_9391 (Atlas A3).
SOC_VERSION="${SOC_VERSION:-ascend910b1}"

log() { printf '\033[1;36m[verify]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[verify:error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Source CANN (required for torch_npu import) ---
[[ -f "$CANN_SETENV" ]] || die "CANN set_env.sh not found at $CANN_SETENV"
# shellcheck disable=SC1090
source "$CANN_SETENV"
if [[ -f "$NNAL_SETENV" ]]; then
    # shellcheck disable=SC1090
    source "$NNAL_SETENV"
else
    log "NNAL set_env.sh not found at $NNAL_SETENV — libatb.so features may be unavailable."
fi
export SOC_VERSION

# --- Activate env ---
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# --- Verify ---
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

log "Verification passed."
