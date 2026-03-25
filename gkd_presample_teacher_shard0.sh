#!/usr/bin/env bash
# ============================================================================
# GKD – Teacher Pre-sampling: Shard 0 re-run
#
# Shard 0 crashed previously. This script re-runs it independently.
#
# GPU layout:
#   GPU 0,1  → Teacher inference (TP=2)
#
# Dataset: files[0:25] of the first 49 parquet files
#   000_00000.parquet .. 002_00004.parquet
#
# Output:
#   output/teacher_presample_shard0.jsonl  (overwrites existing partial file)
#
# Usage:
#   bash gkd_presample_teacher_shard0.sh |& tee gkd_presample_teacher_shard0.log
# ============================================================================
set -euo pipefail

TEACHER_MODEL=/home/r00914194/models/Qwen3.5-35B-A3B
DATA_ROOT="/dev/shm/dataset/fineweb-edu-100BT/sample/100BT"
RESULT_PATH=output/teacher_presample_shard0_run3.jsonl

mapfile -t _ALL < <(python3 -c "
import glob
for f in sorted(glob.glob('${DATA_ROOT}/*.parquet')):
    print(f)
")
# Shard 0: files[0:25]  (mirrors DP=2, SHARD_SIZE=25 split in gkd_presample_teacher.sh)
SHARD_FILES=("${_ALL[@]:0:25}")

echo "=== Shard 0 re-run: ${#SHARD_FILES[@]} files → ${RESULT_PATH} ==="
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
    --vllm_gpu_memory_utilization 0.90 \
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

echo "=== Shard 0 complete. Output: ${RESULT_PATH} ==="
echo "    Line count: $(wc -l < "${RESULT_PATH}")"
