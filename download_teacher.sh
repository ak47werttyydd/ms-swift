#!/usr/bin/env bash
# Download Qwen3.5-35B-A3B from HuggingFace to HF_HOME before training
set -euo pipefail

MODEL_ID="Qwen/Qwen3.5-35B-A3B"
HF_HOME="${HF_HOME:-/dev/shm/HF_HOME}"
# HF_TOKEN=<fill in tokens>

echo "=== Downloading ${MODEL_ID} to ${HF_HOME} ==="

python3 - <<EOF
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id="${MODEL_ID}",
    cache_dir="${HF_HOME}",
    token="${HF_TOKEN}",
    local_dir=None,           # use standard HF cache layout
    ignore_patterns=["*.msgpack", "*.h5", "flax_*"],  # skip non-PyTorch weights
)
print("Download complete.")
EOF

echo "=== Done: ${MODEL_ID} cached at ${HF_HOME} ==="
