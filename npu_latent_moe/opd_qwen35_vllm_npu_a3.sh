#!/usr/bin/env bash
# =============================================================================
# Full On-Policy Distillation on Atlas A3 (16× NPU):
#   Qwen3.5-35B-A3B (teacher, vllm-ascend server)
#   → Qwen3.5-4B    (student, DeepSpeed ZeRO-3 + colocated vllm-ascend)
#
# Random-init checkpoints from npu_latent_moe/init_qwen3.5/.
#
# NPU layout (16× Atlas 910B3 64GB):
#   NPU 12-15 → vLLM teacher (TP=4 for Qwen3.5-35B-A3B MoE)
#   NPU  0-11 → student training (12 NPUs, deepspeed zero3) +
#               colocated student vLLM for on-policy rollouts.
#
# On-policy schedule (single phase, lmbda=1.0, beta=1.0):
#   Student samples each completion via its colocated vLLM; teacher returns
#   top-k logprobs over those tokens; student is trained to match.
#
# Attention backends (per user policy):
#   - HF student training : --attn_impl sdpa      (no flash_attn on NPU)
#   - vLLM (teacher+student colocated) : npu_fusion_attention via vllm-ascend
#     (selected with VLLM_ATTENTION_BACKEND=ASCEND).
#
# Dataset:
#   Alpaca parquet (instruction/input/output). On-policy GKD only consumes
#   the prompt; student samples its own completion, so `output` is unused.
#   We map instruction → query.
#
# Teacher max_model_len ≥ max_length + max_completion_length + 1.
#
# Usage:
#   bash opd_qwen35_vllm_npu_a3.sh |& tee opd_qwen35_vllm_npu_a3.log
# =============================================================================
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_MODEL="${TEACHER_MODEL:-${SCRIPT_DIR}/init_qwen3.5/qwen35_35B_A3B_init_ckpt}"
STUDENT_MODEL="${STUDENT_MODEL:-${SCRIPT_DIR}/init_qwen3.5/qwen35_4B_init_ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/opd_qwen35_35B_to_4B}"
mkdir -p "${OUTPUT_DIR}"

DATASET_PATH="${DATASET_PATH:-/home/canada_group_account/a84400789/dataset/alpaca/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet}"

# ─── Student training config ──────────────────────────────────────────────
STUDENT_NPUS="${STUDENT_NPUS:-0,1,2,3,4,5,6,7,8,9,10,11}"
STUDENT_NPROC="${STUDENT_NPROC:-12}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
STUDENT_VLLM_MAX_MODEL_LEN=$((MAX_LENGTH + MAX_COMPLETION_LENGTH))
MBS="${MBS:-4}"
GRAD_ACC="${GRAD_ACC:-15}"
SAVE_STEPS="${SAVE_STEPS:-50}"

# ─── Teacher vLLM server config ───────────────────────────────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-4}"
TEACHER_NPUS="${TEACHER_NPUS:-12,13,14,15}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-5200}"  # ≥ 4096+1024+1
TEACHER_NPU_MEM_UTIL="${TEACHER_NPU_MEM_UTIL:-0.85}"

# ─── Sanity checks ────────────────────────────────────────────────────────
command -v npu-smi >/dev/null 2>&1 || { echo "ERROR: npu-smi not found — CANN not sourced?" >&2; exit 1; }
[[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] || echo "WARN: ASCEND_TOOLKIT_HOME unset — did you source set_env.sh?"
[[ -d "${TEACHER_MODEL}" ]] || { echo "ERROR: teacher ckpt not found: ${TEACHER_MODEL}" >&2; exit 1; }
[[ -d "${STUDENT_MODEL}" ]] || { echo "ERROR: student ckpt not found: ${STUDENT_MODEL}" >&2; exit 1; }
[[ -f "${DATASET_PATH}"  ]] || { echo "ERROR: dataset not found: ${DATASET_PATH}" >&2; exit 1; }

# ─── Teacher lifecycle ────────────────────────────────────────────────────
start_teacher() {
    echo "=== Starting vLLM teacher on NPUs ${TEACHER_NPUS} (TP=${TEACHER_TP}) ==="
    echo "    max_model_len=${TEACHER_MAX_MODEL_LEN} (≥ ${MAX_LENGTH}+${MAX_COMPLETION_LENGTH}+1)"
    ASCEND_RT_VISIBLE_DEVICES=${TEACHER_NPUS} \
    VLLM_ATTENTION_BACKEND=ASCEND \
    vllm serve "${TEACHER_MODEL}" \
        --port "${TEACHER_PORT}" \
        --tensor-parallel-size "${TEACHER_TP}" \
        --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${TEACHER_NPU_MEM_UTIL}" \
        --max-logprobs "${TEACHER_MAX_LOGPROBS}" \
        --dtype bfloat16 \
        --enforce-eager \
        --trust-remote-code \
        &
    TEACHER_PID=$!
    echo "Teacher PID: ${TEACHER_PID}"

    echo "Waiting for teacher /health (up to 600s; 35B MoE cold start is slow) ..."
    for i in $(seq 1 600); do
        if curl -sf "http://localhost:${TEACHER_PORT}/health" >/dev/null 2>&1; then
            echo "Teacher ready after ${i}s"
            return 0
        fi
        if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
            echo "ERROR: Teacher process exited before becoming ready." >&2
            exit 1
        fi
        sleep 1
    done
    echo "ERROR: Teacher server did not become ready in 600s" >&2
    kill "${TEACHER_PID}" 2>/dev/null || true
    exit 1
}

stop_teacher() {
    [[ -n "${TEACHER_PID:-}" ]] || return 0
    echo "=== Stopping teacher server (PID ${TEACHER_PID}) ==="
    kill "${TEACHER_PID}" 2>/dev/null || true
    wait "${TEACHER_PID}" 2>/dev/null || true
}

# ─── Student lifecycle & watchdog ─────────────────────────────────────────
STUDENT_PID=0
WATCHDOG_PID=0

_global_cleanup() {
    [[ ${WATCHDOG_PID} -ne 0 ]] && kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    [[ ${STUDENT_PID}  -ne 0 ]] && kill -TERM -"${STUDENT_PID}"  2>/dev/null || true
    stop_teacher
}
trap '_global_cleanup; exit' EXIT INT TERM

check_teacher() {
    if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
        echo "ERROR: Teacher server (PID ${TEACHER_PID}) has died. Aborting." >&2
        exit 1
    fi
}

_watchdog_loop() {
    local student_pgid=$1
    while sleep 30; do
        if ! curl -sf "http://localhost:${TEACHER_PORT}/health" >/dev/null 2>&1; then
            echo "ERROR: Teacher unhealthy; killing student pgid ${student_pgid}." >&2
            kill -TERM -"${student_pgid}" 2>/dev/null || true
            exit 1
        fi
    done
}

run_training() {
    setsid "$@" &
    STUDENT_PID=$!

    _watchdog_loop "${STUDENT_PID}" &
    WATCHDOG_PID=$!

    if ! wait "${STUDENT_PID}"; then
        kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
        wait        "${WATCHDOG_PID}" 2>/dev/null || true
        STUDENT_PID=0; WATCHDOG_PID=0
        echo "ERROR: Student training failed." >&2
        exit 1
    fi

    kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    wait        "${WATCHDOG_PID}" 2>/dev/null || true
    STUDENT_PID=0; WATCHDOG_PID=0
    echo "=== Training completed successfully ==="
}

# ═══════════════════════════════════════════════════════════════════════════
start_teacher
check_teacher

echo "=== Full on-policy GKD: Qwen3.5-35B-A3B → Qwen3.5-4B (alpaca) ==="
run_training \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    ASCEND_RT_VISIBLE_DEVICES=${STUDENT_NPUS} \
    VLLM_ATTENTION_BACKEND=ASCEND \
    swift rlhf \
        --rlhf_type gkd \
        --model_type qwen3_5 \
        --model "${STUDENT_MODEL}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --dataset "${DATASET_PATH}" \
        --columns '{"instruction":"query","output":"response"}' \
        --seq_kd false \
        --lmbda 1.0 \
        --beta 1.0 \
        --train_type full \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length ${MAX_LENGTH} \
        --max_completion_length ${MAX_COMPLETION_LENGTH} \
        --truncation_strategy right \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size ${MBS} \
        --gradient_accumulation_steps ${GRAD_ACC} \
        --learning_rate 1e-5 \
        --max_steps 10000 \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit 5 \
        --save_only_model true \
        --deepspeed zero3 \
        --gradient_checkpointing true \
        --attn_impl sdpa \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --enable_thinking false \
        --logging_steps 10 \
        --load_from_cache_file true \
        --use_vllm true \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.65 \
        --vllm_tensor_parallel_size 1 \
        --vllm_max_model_len ${STUDENT_VLLM_MAX_MODEL_LEN} \
        --vllm_enforce_eager true \
        --sleep_level 2 \
        --offload_model true \
        --offload_optimizer true \
        --move_model_batches 24 \
        --output_dir "${OUTPUT_DIR}"

stop_teacher
echo "=== Done. Final model: ${OUTPUT_DIR} ==="
