#!/usr/bin/env bash
# ============================================================================
# GKD Solution 2 – Step 1: Teacher Model Pre-sampling
#
# Pre-generates teacher completions for the entire fineweb-edu-100BT dataset
# offline, so the student can train on them without the teacher generating
# on-the-fly.  Covers all parquet files (not just the Phase 1 slice of 49).
#
# GPU layout (4× GPU):
#   GPU 0-3  → Teacher inference (TP=4)
#              Qwen3.5-35B-A3B in BF16 ≈ 70GB → ~17.5GB/GPU at TP=4,
#              leaving ~60GB/GPU for KV cache with util=0.90
#
# Output:
#   output/teacher_presample.jsonl   (query + teacher response per sample)
#
# Usage:
#   bash gkd_presample_teacher.sh |& tee gkd_presample_teacher.log
#
# Run this script BEFORE gkd_student_presample.sh.
# ============================================================================
set -euo pipefail

# ── Model paths ──────────────────────────────────────────────────────────────
TEACHER_MODEL=/home/r00914194/models/Qwen3.5-35B-A3B
RESULT_PATH=output/teacher_presample.jsonl

# ── GPU config ───────────────────────────────────────────────────────────────
TEACHER_GPUS="0,1,2,3"
TEACHER_DP=2
TEACHER_TP=2

# ── Dataset: first 49 parquet files (DATASET1 slice, mirrors gkd_qwen35_vllm.sh) ─
DATA_ROOT="/dev/shm/dataset/fineweb-edu-100BT/sample/100BT"

mapfile -t _ALL < <(python3 -c "
import glob
for f in sorted(glob.glob('${DATA_ROOT}/*.parquet')):
    print(f)
")
DATASET_CONSUMED_FILES=("${_ALL[@]:0:49}")

echo "=== Teacher pre-sampling: ${DATASET_CONSUMED_FILES[@]} files → ${RESULT_PATH} ==="
echo "    Teacher:  ${TEACHER_MODEL}"
echo "    GPUs:     ${TEACHER_GPUS} ,TP=${TEACHER_TP}, DP=${TEACHER_DP}"
# echo "    GPUs:     ${TEACHER_GPUS} (TP=${TEACHER_TP})"

mkdir -p "$(dirname "${RESULT_PATH}")"

# ── Run swift infer ───────────────────────────────────────────────────────────
# Column mapping: "text" → "query" so swift infer treats each document text
# as the prompt the teacher responds to (mirrors Phase 1 data flow where the
# "text" field drives the teacher generation in seq_kd mode).
#
# Adjust --columns if your dataset needs a different query/response split
# (e.g. use '{"prompt":"query","text":"response"}' for prompt-conditioned data).
CUDA_VISIBLE_DEVICES=${TEACHER_GPUS} \
NPROC_PER_NODE=${TEACHER_DP} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift infer \
    --model "${TEACHER_MODEL}" \
    --model_type qwen3_5_moe \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_num_seqs 32 \
    --vllm_tensor_parallel_size ${TEACHER_TP} \
    --val_dataset "${DATASET_CONSUMED_FILES[@]}" \
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

echo "=== Pre-sampling complete. Output: ${RESULT_PATH} ==="
echo "    Line count: $(wc -l < "${RESULT_PATH}")"

# --write_batch_size 1 \
# --vllm_max_model_len 8192 \
# --vllm_tensor_parallel_size ${TEACHER_TP} \
# --columns '{"text":"content"}' \