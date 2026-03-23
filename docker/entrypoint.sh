#!/bin/bash

# Claude environment
echo 'export PATH=$PATH:/home/ubuntu/.local/bin' >> ~/.bashrc

# ── Install ms-swift (editable, from host-mounted source) ────────────────────
echo "=== Installing ms-swift (editable) ==="
pip install -e /home/a84400789/ms-swift/

# 对于强化学习（RL）训练，需要覆盖 vLLM 的默认安装版本
# 训练报错参考这个issue: https://github.com/modelscope/ms-swift/issues/8188
echo "=== Overriding transformers (must come after ms-swift to override vLLM's pinned version) ==="
pip install -U "transformers>=5.3.0"

# vLLM 安装的 torchao dev 版本（0.16.0+git）与 torch 2.10.0+cu128 不兼容：
#   torchao/dtypes/nf4tensor.py 引用了 torch.ops._c10d_functional._wrap_tensor_autograd
#   该 op 在当前 torch 版本中不存在，导致 transformers/swift 的 import 全部失败。
# KD 训练不需要 torchao 量化，直接卸载即可。
echo "=== Removing incompatible torchao (vLLM dev build, broken with torch 2.10.0+cu128) ==="
pip uninstall -y torchao || true

# ── Pre-create triton cache dirs (deepspeed/triton checks these at import time) ─
mkdir -p /root/.triton/autotune

# ── Verify key packages ───────────────────────────────────────────────────────
echo "=== Verifying packages ==="
python -c "import swift;        print(f'ms-swift      {swift.__version__}')"
python -c "import vllm;         print(f'vllm          {vllm.__version__}')"          || echo "WARNING: vllm import failed"
python -c "import transformers; print(f'transformers  {transformers.__version__}')"
python -c "import deepspeed;    print(f'deepspeed     {deepspeed.__version__}')"     || echo "WARNING: deepspeed import failed"
python -c "import flash_attn;   print(f'flash-attn    {flash_attn.__version__}')"   || echo "WARNING: flash_attn import failed"

exec "$@"
