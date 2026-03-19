#!/bin/bash
set -e

# ── Install ms-swift (editable, from host-mounted source) ────────────────────
echo "=== Installing ms-swift (editable) ==="
uv pip install -e /home/a84400789/ms-swift/ --torch-backend=auto

# ── Verify key packages ───────────────────────────────────────────────────────
echo "=== Verifying packages ==="
python -c "import swift;        print(f'ms-swift      {swift.__version__}')"
python -c "import vllm;         print(f'vllm          {vllm.__version__}')"
python -c "import transformers; print(f'transformers  {transformers.__version__}')"
python -c "import deepspeed;    print(f'deepspeed     {deepspeed.__version__}')"
python -c "import flash_attn;   print(f'flash-attn    {flash_attn.__version__}')"

exec "$@"
