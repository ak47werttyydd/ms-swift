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
DATASET_CONSUMED_FILES=("${_ALL[@]:0:49}") # 35B tokens

# ── Split files across DP shards ────────────────────────────────────────────
NUM_FILES=${#DATASET_CONSUMED_FILES[@]}
SHARD_SIZE=$(( (NUM_FILES + TEACHER_DP - 1) / TEACHER_DP ))

echo "=== Teacher pre-sampling: ${NUM_FILES} files → ${RESULT_PATH} ==="
echo "    Teacher:  ${TEACHER_MODEL}"
echo "    DP=${TEACHER_DP}, TP=${TEACHER_TP} (2 separate processes, not torchrun)"

mkdir -p "$(dirname "${RESULT_PATH}")"

# ── Common swift infer args ─────────────────────────────────────────────────
COMMON_ARGS=(
    --model "${TEACHER_MODEL}"
    --model_type qwen3_5_moe
    --infer_backend vllm
    --vllm_gpu_memory_utilization 0.90
    --vllm_tensor_parallel_size ${TEACHER_TP}
    --columns '{"text":"query"}'
    --vllm_max_model_len 8192
    --max_new_tokens 4096
    --max_length 4096
    --temperature 1.0
    --torch_dtype bfloat16
    --enable_thinking false
    --dataset_num_proc 8
    --truncation_strategy right
    --write_batch_size 100
)

# ── Launch DP shards in parallel ────────────────────────────────────────────
# Each shard is a separate swift infer process with TP=2 on its own GPU pair.
# vLLM V1 (0.17.1) external_launcher doesn't support multiple TP groups in
# one torchrun, so we launch independent processes instead.
PIDS=()
for (( dp=0; dp<TEACHER_DP; dp++ )); do
    START=$(( dp * SHARD_SIZE ))
    END=$(( START + SHARD_SIZE ))
    if (( END > NUM_FILES )); then END=${NUM_FILES}; fi
    SHARD_FILES=("${DATASET_CONSUMED_FILES[@]:${START}:$(( END - START ))}")

    # Pick GPU pair for this DP shard: shard 0 → GPU 0,1; shard 1 → GPU 2,3
    GPU_START=$(( dp * TEACHER_TP ))
    GPU_IDS=""
    for (( g=0; g<TEACHER_TP; g++ )); do
        (( g > 0 )) && GPU_IDS+=","
        GPU_IDS+="$(( GPU_START + g ))"
    done

    SHARD_RESULT="output/teacher_presample_shard${dp}.jsonl"
    echo "  Shard ${dp}: GPUs=${GPU_IDS}, files[${START}:${END}] → ${SHARD_RESULT}"

    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift infer \
        "${COMMON_ARGS[@]}" \
        --val_dataset "${SHARD_FILES[@]}" \
        --result_path "${SHARD_RESULT}" &
    PIDS+=($!)
done

# ── Wait for all shards ─────────────────────────────────────────────────────
FAIL=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
done
if (( FAIL )); then
    echo "ERROR: one or more shards failed" >&2
    exit 1
fi

# ── Merge shard outputs ─────────────────────────────────────────────────────
> "${RESULT_PATH}"
for (( dp=0; dp<TEACHER_DP; dp++ )); do
    cat "output/teacher_presample_shard${dp}.jsonl" >> "${RESULT_PATH}"
done

echo "=== Pre-sampling complete. Output: ${RESULT_PATH} ==="
echo "    Line count: $(wc -l < "${RESULT_PATH}")"

# --vllm_max_num_seqs 32 \
# --write_batch_size 1 \
# --vllm_max_model_len 8192 \
# --vllm_tensor_parallel_size ${TEACHER_TP} \
# --columns '{"text":"content"}' \