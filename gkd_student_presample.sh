#!/usr/bin/env bash
# ============================================================================
# GKD Solution 2 – Step 2: Student Training on Pre-sampled Data
#
# Trains the student model using teacher completions that were pre-generated
# by gkd_presample_teacher.sh (Solution 2 from the GKD docs).
#
# Offline KD mode: teacher completions are already in the JSONL dataset.
# No teacher server is needed during training.
#
# GPU layout (8× total):
#   GPU 4-7  → Student training (ZeRO-3, 8 GPUs)
#
# Key settings:
#   • seq_kd false  → dataset already contains teacher completions; no
#                     on-the-fly generation by the teacher during training
#   • lmbda 0.0     → Mode 3 offline KD; student learns purely from the
#                     pre-sampled teacher responses in the JSONL
#
# Usage:
#   bash gkd_student_presample.sh |& tee gkd_student_presample.log
#
# Run AFTER gkd_presample_teacher.sh has finished.
# ============================================================================
set -euo pipefail

# ── Model paths ──────────────────────────────────────────────────────────────
STUDENT_MODEL=/home/r00914194/PruneMe/qwen35_exp80_layer_drop_082331
TEACHER_MODEL=/home/r00914194/models/Qwen3.5-35B-A3B
PRESAMPLE_DATA=output/presample_test.jsonl
OUTPUT_DIR=output/presample_student_test

# ── GPU config ───────────────────────────────────────────────────────────────
STUDENT_GPUS="4,5,6,7"
STUDENT_NPROC=4

# gkd top k
GKD_MAX_LOGPROBS=20

# ── Verify pre-sampled data exists ───────────────────────────────────────────
if [[ ! -f "${PRESAMPLE_DATA}" ]]; then
    echo "ERROR: Pre-sampled data not found: ${PRESAMPLE_DATA}"
    echo "       Run gkd_presample_teacher.sh first."
    exit 1
fi
echo "Using pre-sampled data: ${PRESAMPLE_DATA} ($(wc -l < "${PRESAMPLE_DATA}") lines)"

# ── Student training on pre-sampled data ─────────────────────────────────────
echo "=== Student training on pre-sampled data ==="
echo "    Student:  ${STUDENT_MODEL}"
echo "    Dataset:  ${PRESAMPLE_DATA}"
echo "    GPUs:     ${STUDENT_GPUS} (${STUDENT_NPROC} processes)"

run_phase() {
    local phase_name=$1
    local marker="${phase_name}/.done"
    shift
    if [[ -f "${marker}" ]]; then
        echo "=== ${phase_name} already completed, skipping ==="
        return 0
    fi
    "$@"
    touch "${marker}"
    echo "=== ${phase_name} completed successfully ==="
}

run_phase "${OUTPUT_DIR}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        --rlhf_type gkd \
        --model_type qwen3_5_moe \
        --model "${STUDENT_MODEL}" \
        --teacher_model "${TEACHER_MODEL}" \
        --dataset "${PRESAMPLE_DATA}" \
        --seq_kd false \
        --lmbda 0.0 \
        --beta 0.5 \
        --gkd_logits_topk ${GKD_MAX_LOGPROBS} \
        --train_type full \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length 8192 \
        --truncation_strategy right \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 42 \
        --learning_rate 1e-5 \
        --num_train_epochs 1 \
        --save_steps 10 \
        --save_only_model true \
        --deepspeed zero2 \
        --attn_impl flash_attn \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --enable_thinking false \
        --logging_steps 10 \
        --load_from_cache_file true \
        --padding_free true \
        --output_dir "${OUTPUT_DIR}"

echo "=== Done. Model saved to: ${OUTPUT_DIR} ==="
#--gradient_checkpointing true \
#--save_total_limit 2 \
#--max_steps 24000 \
#--max_completion_length 4096 \    #no inference under lmbda=0
#--max_length 8192 \  #because teacher query 4k + teacher response 4k = 8k total context, the student trains on the whole context
# --deepspeed zero3 \     # AssertionError: loss must be a scalar tensor. If you need to pass output gradients, backward() of output tensors