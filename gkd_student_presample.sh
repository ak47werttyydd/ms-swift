#!/usr/bin/env bash
# ============================================================================
# GKD Solution 2 – Step 2: Student Training on Pre-sampled Data
#
# Trains the student model using teacher completions that were pre-generated
# by gkd_presample_teacher.sh (Solution 2 from the GKD docs).
#
# The teacher model is still needed during training to compute the KD
# divergence loss (token-level logprob comparison).  We start it as a vLLM
# server on GPUs 0-3 so only its logprobs are requested — no on-the-fly
# generation — and train the student on GPUs 4-7.
#
# GPU layout (8× total):
#   GPU 0-3  → vLLM teacher server (TP=4) – logprobs only, no generation
#   GPU 4-7  → Student training (ZeRO-3, 4 GPUs)
#
# Key difference from Phase 1 of gkd_qwen35_vllm.sh:
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
TEACHER_MODEL=Qwen/Qwen3.5-35B-A3B
STUDENT_MODEL=/home/r00914194/PruneMe/qwen35_exp80_layer_drop_082331
PRESAMPLE_DATA=output/teacher_presample.jsonl
OUTPUT_DIR=output/gkd_presample_student

# ── GPU config ───────────────────────────────────────────────────────────────
TEACHER_GPUS="0,1,2,3"
TEACHER_TP=4
TEACHER_PORT=8000
TEACHER_GPU_MEM_UTIL=0.90
TEACHER_MAX_LOGPROBS=64
# max_model_len only needs to cover the training max_length (4096) + 1 for
# logprob queries (no generation), but keep it consistent with Phase 1.
TEACHER_MAX_MODEL_LEN=8193

STUDENT_GPUS="4,5,6,7"
STUDENT_NPROC=4

# ── Verify pre-sampled data exists ───────────────────────────────────────────
if [[ ! -f "${PRESAMPLE_DATA}" ]]; then
    echo "ERROR: Pre-sampled data not found: ${PRESAMPLE_DATA}"
    echo "       Run gkd_presample_teacher.sh first."
    exit 1
fi
echo "Using pre-sampled data: ${PRESAMPLE_DATA} ($(wc -l < "${PRESAMPLE_DATA}") lines)"

# ── Start teacher vLLM server (logprobs only) ─────────────────────────────────
start_teacher() {
    if ! pgrep -x nv-fabricmanager > /dev/null 2>&1; then
        echo "WARNING: nvidia-fabricmanager is not running."
        echo "  Fix: sudo systemctl start nvidia-fabricmanager"
        nvidia-smi -pm 1 2>/dev/null || true
    fi

    echo "=== Starting vLLM teacher server on GPUs ${TEACHER_GPUS} (TP=${TEACHER_TP}) ==="
    CUDA_VISIBLE_DEVICES=${TEACHER_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    vllm serve "${TEACHER_MODEL}" \
        --port ${TEACHER_PORT} \
        --tensor-parallel-size ${TEACHER_TP} \
        --max-model-len ${TEACHER_MAX_MODEL_LEN} \
        --gpu-memory-utilization ${TEACHER_GPU_MEM_UTIL} \
        --max-logprobs ${TEACHER_MAX_LOGPROBS} \
        --language-model-only \
        --dtype bfloat16 \
        --enforce-eager \
        &
    TEACHER_PID=$!
    echo "Teacher PID: ${TEACHER_PID}"

    echo "Waiting for teacher server to be ready..."
    for i in $(seq 1 300); do
        if curl -s "http://localhost:${TEACHER_PORT}/health" > /dev/null 2>&1; then
            echo "Teacher server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Teacher server did not become ready in 300s"
    kill ${TEACHER_PID} 2>/dev/null || true
    exit 1
}

stop_teacher() {
    echo "=== Stopping teacher server (PID ${TEACHER_PID}) ==="
    kill ${TEACHER_PID} 2>/dev/null || true
    wait ${TEACHER_PID} 2>/dev/null || true
}

trap 'stop_teacher; exit' EXIT INT TERM

start_teacher

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
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --dataset "${PRESAMPLE_DATA}" \
        --seq_kd false \
        --lmbda 0.0 \
        --beta 0.5 \
        --train_type full \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length 4096 \
        --max_completion_length 4096 \
        --truncation_strategy right \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 21 \
        --learning_rate 1e-4 \
        --max_steps 24000 \
        --save_steps 200 \
        --save_total_limit 2 \
        --save_only_model true \
        --deepspeed zero3 \
        --attn_impl flash_attn \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --gradient_checkpointing true \
        --enable_thinking false \
        --logging_steps 10 \
        --output_dir "${OUTPUT_DIR}"

stop_teacher
echo "=== Done. Model saved to: ${OUTPUT_DIR} ==="
