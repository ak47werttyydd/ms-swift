#!/bin/bash
set -e

# ── Claude Code CLI ──────────────────────────────────────────────────────────
echo 'export PATH=$PATH:/home/ubuntu/.local/bin' >> ~/.bashrc

# ── Install ms-swift (editable, from host-mounted source) ────────────────────
MS_SWIFT_DIR="${MS_SWIFT_DIR:-/home/a84400789/ms-swift}"
if [[ -d "${MS_SWIFT_DIR}" ]]; then
    echo "=== Installing ms-swift (editable) from ${MS_SWIFT_DIR} ==="
    pip install -e "${MS_SWIFT_DIR}/"
else
    echo "WARNING: ms-swift dir not found at ${MS_SWIFT_DIR}, skipping editable install"
fi

# ── Override transformers (must come after ms-swift to override vLLM's pin) ──
# vLLM pins transformers to an older version; Qwen3.5 LatentMoE needs >=5.3.0
# Ref: https://github.com/modelscope/ms-swift/issues/8188
echo "=== Ensuring transformers==5.3.0 ==="
pip install "transformers==5.3.0"

# ── Remove incompatible torchao ──────────────────────────────────────────────
# vLLM installs torchao dev (0.16.0+git) which references
# torch.ops._c10d_functional._wrap_tensor_autograd — doesn't exist in
# torch 2.10.0+cu128. KD training doesn't need torchao.
pip uninstall -y torchao 2>/dev/null || true

# ── Pre-create triton cache dir (deepspeed/triton check at import time) ──────
mkdir -p /root/.triton/autotune

# ── Verify key packages ─────────────────────────────────────────────────────
echo "=== Package verification ==="
python -c "
import sys
packages = {
    'swift':        lambda: __import__('swift').__version__,
    'vllm':         lambda: __import__('vllm').__version__,
    'transformers': lambda: __import__('transformers').__version__,
    'deepspeed':    lambda: __import__('deepspeed').__version__,
    'flash_attn':   lambda: __import__('flash_attn').__version__,
    'torch':        lambda: __import__('torch').__version__,
}
ok = True
for name, get_ver in packages.items():
    try:
        ver = get_ver()
        print(f'  {name:<16} {ver}')
    except Exception as e:
        print(f'  {name:<16} FAILED: {e}')
        ok = False
if not ok:
    print('WARNING: some packages failed to import')
"

echo "=== Environment ready ==="
exec "$@"
