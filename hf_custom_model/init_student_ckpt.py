"""Initialize a language-model-only student checkpoint for GKD.

Generates randomly initialized LLM weights (sharded safetensors) into
initial_ckpt/. All other files (config.json, modeling code, tokenizer)
must already exist in initial_ckpt/ before running this script.

Vision encoder weights are excluded — ms-swift with --freeze_vit true
will randomly init them at load time (they stay frozen).
"""
import os
import sys
import json

import torch
from safetensors.torch import save_file

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(SCRIPT_DIR, "initial_ckpt")

# ── Step 1: Initialize model and save LLM-only weights ──────────────────────
print("=== Step 1: Initialize model from config ===")

# Add the checkpoint dir to sys.path so transformers can find the custom classes
sys.path.insert(0, CKPT_DIR)

from transformers import AutoConfig, AutoModelForCausalLM

model_config = AutoConfig.from_pretrained(CKPT_DIR, trust_remote_code=True)
print(f"  Config loaded: {model_config.architectures}")

# Initialize with random weights in bf16 on CPU
print("  Initializing model (bf16, CPU)... this may take a minute")
model = AutoModelForCausalLM.from_config(
    model_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params / 1e9:.2f}B")

# Separate LLM vs vision parameters
llm_state = {}
vision_state = {}
for name, param in model.state_dict().items():
    if "visual" in name:
        vision_state[name] = param
    else:
        llm_state[name] = param

# Always include lm_head.weight (copy of embed_tokens if tied).
# Transformers 5.3.0 has a bug where it untie weights when lm_head is absent,
# leaving lm_head on meta device. Including both makes both tie=True/False safe.
embed_key = "model.language_model.embed_tokens.weight"
lm_head_key = "lm_head.weight"
if lm_head_key not in llm_state and embed_key in llm_state:
    llm_state[lm_head_key] = llm_state[embed_key].clone()
    print(f"  Added {lm_head_key} (copy of embed_tokens, for tie_word_embeddings compat)")

llm_params = sum(p.numel() for p in llm_state.values())
vis_params = sum(p.numel() for p in vision_state.values())
print(f"  LLM parameters:    {llm_params / 1e9:.2f}B (saving)")
print(f"  Vision parameters: {vis_params / 1e6:.1f}M (skipping)")

# ── Step 2: Save LLM weights as sharded safetensors ─────────────────────────
print("=== Step 2: Save LLM weights ===")

# Shard into ~2GB files
MAX_SHARD_SIZE = 2 * 1024 * 1024 * 1024  # 2GB

shards = []
current_shard = {}
current_size = 0
weight_map = {}

for name, tensor in llm_state.items():
    tensor_size = tensor.numel() * tensor.element_size()
    if current_size + tensor_size > MAX_SHARD_SIZE and current_shard:
        shards.append(current_shard)
        current_shard = {}
        current_size = 0
    current_shard[name] = tensor
    current_size += tensor_size

if current_shard:
    shards.append(current_shard)

# Save each shard
total_size = 0
for i, shard in enumerate(shards):
    if len(shards) == 1:
        shard_name = "model.safetensors"
    else:
        shard_name = f"model-{i+1:05d}-of-{len(shards):05d}.safetensors"

    shard_path = os.path.join(CKPT_DIR, shard_name)
    save_file(shard, shard_path)

    shard_size = os.path.getsize(shard_path)
    total_size += shard_size
    print(f"  Saved {shard_name} ({shard_size / 1e9:.2f} GB, {len(shard)} tensors)")

    for name in shard:
        weight_map[name] = shard_name

# Save index file if sharded
if len(shards) > 1:
    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in llm_state.values())},
        "weight_map": weight_map,
    }
    index_path = os.path.join(CKPT_DIR, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
        f.write("\n")
    print(f"  Saved model.safetensors.index.json")

print(f"\n=== Done! Checkpoint at {CKPT_DIR} ===")
print(f"  Total weight size: {total_size / 1e9:.2f} GB")
print(f"  Files: {len(os.listdir(CKPT_DIR))}")
for f in sorted(os.listdir(CKPT_DIR)):
    size = os.path.getsize(os.path.join(CKPT_DIR, f))
    if size > 1e6:
        print(f"    {f:50s} {size/1e9:.2f} GB")
    else:
        print(f"    {f:50s} {size/1e3:.1f} KB")
