#!/usr/bin/env bash
# ============================================================================
# GKD – Teacher Pre-sampling: Shard 4
#
# GPU layout:
#   GPU 6,7  → Teacher inference (TP=2)
#
# Dataset: files[3] (4th parquet file)
#   000_00003.parquet
#
# Output:
#   output/teacher_presample_shard4.jsonl
#
# Usage:
#   bash gkd_presample_teacher_shard4.sh |& tee gkd_presample_teacher_shard4.log
# ============================================================================
set -euo pipefail

TEACHER_MODEL=/home/r00914194/models/Qwen3.5-35B-A3B
DATA_ROOT="/dev/shm/dataset/fineweb-edu-100BT/sample/100BT"
RESULT_PATH=output/teacher_presample_shard4.jsonl

mapfile -t _ALL < <(python3 -c "
import glob
for f in sorted(glob.glob('${DATA_ROOT}/*.parquet')):
    print(f)
")
# Shard 4: files[4] (4th parquet file)
SHARD_FILES=("${_ALL[@]:4:1}")

echo "=== Shard 4: ${#SHARD_FILES[@]} files → ${RESULT_PATH} ==="
echo "    First: ${SHARD_FILES[0]}"
echo "    Last:  ${SHARD_FILES[-1]}"
echo "    GPUs:  0,1  (TP=2)"

mkdir -p "$(dirname "${RESULT_PATH}")"

CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift infer \
    --model "${TEACHER_MODEL}" \
    --model_type qwen3_5_moe \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.85 \
    --vllm_tensor_parallel_size 2 \
    --val_dataset "${SHARD_FILES[@]}" \
    --columns '{"text":"query"}' \
    --vllm_max_model_len 8192 \
    --max_new_tokens 4096 \
    --max_length 4096 \
    --temperature 1.0 \
    --torch_dtype bfloat16 \
    --enable_thinking false \
    --dataset_num_proc 8 \
    --truncation_strategy right \
    --write_batch_size 100 \
    --result_path "${RESULT_PATH}"

echo "=== Shard 4 complete. Output: ${RESULT_PATH} ==="
echo "    Line count: $(wc -l < "${RESULT_PATH}")"
