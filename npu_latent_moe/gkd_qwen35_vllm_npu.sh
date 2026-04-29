#!/usr/bin/env bash
# =============================================================================
# Off-policy GKD on NPU: Qwen3.5-35B-A3B teacher → small Qwen3.5 student
#
# Single-phase, off-policy KD (lmbda=0.0, seq_kd=false):
#   - Teacher serves via vllm-ascend (releases/v0.18.0).
#   - Student trains against teacher logprobs over nemotron-CC.
#   - max_completion_length=1 means teacher only supplies per-token
#     distributions, not full generations → this is pure distillation on
#     pre-tokenized text, not sequence-level KD.
#
# NPU layout (8× Atlas 910B3 64GB):
#   NPU 4-7 → vLLM teacher (TP=4 for Qwen3.5-35B-A3B MoE)
#   NPU 0-3 → student training (deepspeed zero3)
#
# Ported from ak47werttyydd/ms-swift:latentMoE/gkd_qwen35_vllm.sh phase 1.
# CUDA→NPU diffs: CUDA_VISIBLE_DEVICES→ASCEND_RT_VISIBLE_DEVICES; dropped
# PYTORCH_CUDA_ALLOC_CONF and nvidia-fabricmanager probe; dropped
# --language-model-only (vllm-ascend doesn't honor it); health probe extended
# to 600s for MoE cold start.
#
# Usage:
#   bash gkd_qwen35_vllm_npu.sh |& tee gkd_qwen35_vllm_npu.log
# =============================================================================
set -euo pipefail

# ─── Models ────────────────────────────────────────────────────────────────
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-35B-A3B}"
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3.5-4B}"

OUTPUT_DIR="${OUTPUT_DIR:-output/gkd_qwen35_offpolicy}"

# ─── Teacher vLLM server ───────────────────────────────────────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-4}"
TEACHER_NPUS="${TEACHER_NPUS:-4,5,6,7}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"

# Student uses max_length=4607 (input) + max_completion_length=1 (teacher
# generates 1 token per position). vLLM enforces:
#     input_tokens + max_tokens  ≤  max_model_len
# Min = 4607 + 1 = 4608. Per the reference-repo bug note, leave +2 headroom
# so mid-stream prompts that include the extra BOS/assistant token don't
# blow past the budget and kill the server.
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-4610}"
TEACHER_NPU_MEM_UTIL="${TEACHER_NPU_MEM_UTIL:-0.85}"

# ─── Student training ──────────────────────────────────────────────────────
STUDENT_NPUS="${STUDENT_NPUS:-0,1,2,3}"
STUDENT_NPROC="${STUDENT_NPROC:-4}"

# ─── Dataset (nemotron-CC) ─────────────────────────────────────────────────
# TODO: replace with the real path on the NPU host. ms-swift will glob
# *.parquet / *.jsonl inside this folder (adjust if your format differs).
DATASET_DIR="${DATASET_DIR:-/data/nemotron-cc}"

if [[ ! -d "${DATASET_DIR}" ]]; then
    echo "WARN: DATASET_DIR '${DATASET_DIR}' does not exist — this is a placeholder." >&2
    echo "      Edit the script or export DATASET_DIR=/actual/path before running." >&2
fi

# ─── Sanity checks ─────────────────────────────────────────────────────────
command -v npu-smi >/dev/null 2>&1 || { echo "ERROR: npu-smi not found — CANN not sourced?" >&2; exit 1; }
[[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] || echo "WARN: ASCEND_TOOLKIT_HOME unset — did you source set_env.sh?"

# ─── Teacher lifecycle ─────────────────────────────────────────────────────
start_teacher() {
    echo "=== Starting vLLM teacher on NPUs ${TEACHER_NPUS} (TP=${TEACHER_TP}) ==="
    echo "    max_model_len=${TEACHER_MAX_MODEL_LEN} (student max_length=4607 + max_completion_length=1 + 2 headroom)"
    ASCEND_RT_VISIBLE_DEVICES=${TEACHER_NPUS} \
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

# ─── Student lifecycle & watchdog ──────────────────────────────────────────
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
    mkdir -p "${OUTPUT_DIR}"
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

echo "=== Off-policy GKD: Qwen3.5-35B-A3B → ${STUDENT_MODEL##*/} ==="
run_training \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    ASCEND_RT_VISIBLE_DEVICES=${STUDENT_NPUS} \
    swift rlhf \
        --rlhf_type gkd \
        --model_type qwen3_5_moe \
        --model "${STUDENT_MODEL}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --dataset "${DATASET_DIR}" \
        --seq_kd false \
        --lmbda 0.0 \
        --beta 0.5 \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --train_type full \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length 4607 \
        --max_completion_length 1 \
        --truncation_strategy left \
        --loss_scale all \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 10 \
        --learning_rate 1e-5 \
        --num_train_epochs 1 \
        --save_steps 100 \
        --save_total_limit 10 \
        --save_only_model true \
        --deepspeed zero3 \
        --attn_impl flash_attn \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --enable_thinking false \
        --logging_steps 10 \
        --load_from_cache_file true \
        --output_dir "${OUTPUT_DIR}"
