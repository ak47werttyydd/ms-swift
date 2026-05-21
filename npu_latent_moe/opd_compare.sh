#!/usr/bin/env bash
# =============================================================================
# OPD comparison run: forward-KL on GSM8K (question column).
#
# Layout (16× Atlas 910B3 64GB):
#   NPU 12-15 → vLLM teacher (TP=4 for Qwen3.5-35B-A3B MoE)  → 4 NPUs
#   NPU  0-11 → student training (DeepSpeed ZeRO-3) + colocated student vLLM
#               → 12 NPUs
#
# Differences vs opd_qwen35_vllm_npu_a3.sh:
#   - Loss: forward KL  (beta=0  → KL(teacher || student))
#   - Dataset: GSM8K parquet, question → query
#   - Teacher vLLM: graph mode (no --enforce-eager) — stable inference server,
#     shapes bucketed, graph compile amortizes over thousands of forwards.
#   - Student colocated vLLM: KEEP enforce-eager (--vllm_enforce_eager true).
#     sleep_level=2 releases weights+KV every step and training overwrites
#     weights every step → ACL graph capture cost can't be amortized and may
#     re-trigger on wake. Eager is more predictable here.
#   - gradient_accumulation_steps = 1
#   - micro batch size = 4
#   - max_steps = 100
#   - TEACHER_MAX_MODEL_LEN = 6224
#   - student MAX_COMPLETION_LENGTH = 2048
#   - student colocated vllm_max_model_len = 6144
#   - student model = /home/canada_group_account/a84400789/qwen3.5_0.8B
#
# Other parameter changes you should be aware of (not requested explicitly,
# but required to make the run actually launch):
#   * MAX_LENGTH lowered from 4096 → 4096 still fits 6144 student vLLM
#     (4096 + 2048 = 6144 exactly).  Teacher 6224 ≥ 4096+2048+1 ✓.
#   * --columns changed from instruction/output → just question→query.
#     GSM8K has no "instruction" / "output" fields, so the old mapping would
#     KeyError. On-policy GKD with seq_kd=false ignores the response column,
#     so omitting it is safe.
#   * --save_steps lowered to 50 (was 50) — kept; with only 100 steps you'll
#     get 2 checkpoints. Set SAVE_STEPS=100 if you want only the final one.
#   * Output dir renamed to ...opd_compare so it doesn't clobber the previous
#     alpaca run.
# =============================================================================
set -euo pipefail

# ─── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEACHER_MODEL="${TEACHER_MODEL:-/home/canada_group_account/a84400789/qwen3.5_35B_a3B}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/canada_group_account/a84400789/qwen3.5_0.8B}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/opd_compare}"
mkdir -p "${OUTPUT_DIR}"

DATASET_PATH="${DATASET_PATH:-/home/canada_group_account/a84400789/dataset/gsm8k/main/train-00000-of-00001.parquet}"

# ─── Student training config ──────────────────────────────────────────────
STUDENT_NPUS="${STUDENT_NPUS:-0,1,2,3,4,5,6,7,8,9,10,11}"
STUDENT_NPROC="${STUDENT_NPROC:-12}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
STUDENT_VLLM_MAX_MODEL_LEN="${STUDENT_VLLM_MAX_MODEL_LEN:-6144}"
MBS="${MBS:-4}"
GRAD_ACC="${GRAD_ACC:-1}"
MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-10}"

# ─── Teacher vLLM server config ───────────────────────────────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-4}"
TEACHER_NPUS="${TEACHER_NPUS:-12,13,14,15}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-6224}"  # ≥ 4096+2048+1
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
    VLLM_ASCEND_ENABLE_NZ=0 \
    vllm serve "${TEACHER_MODEL}" \
        --port "${TEACHER_PORT}" \
        --tensor-parallel-size "${TEACHER_TP}" \
        --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${TEACHER_NPU_MEM_UTIL}" \
        --max-logprobs "${TEACHER_MAX_LOGPROBS}" \
        --dtype bfloat16 \
        --trust-remote-code \
        &
    TEACHER_PID=$!
    echo "Teacher PID: ${TEACHER_PID}"

    echo "Waiting for teacher /health (up to 900s; 35B MoE cold start + graph compile) ..."
    for i in $(seq 1 900); do
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
    echo "ERROR: Teacher server did not become ready in 900s" >&2
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

echo "=== OPD compare: Qwen3.5-35B-A3B → qwen3.5_0.8B  (forward KL, gsm8k) ==="
run_training \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    ASCEND_RT_VISIBLE_DEVICES=${STUDENT_NPUS} \
    VLLM_ATTENTION_BACKEND=ASCEND \
    VLLM_ASCEND_ENABLE_NZ=0 \
    swift rlhf \
        --rlhf_type gkd \
        --model_type qwen3_5 \
        --model "${STUDENT_MODEL}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --dataset "${DATASET_PATH}" \
        --columns '{"question":"query"}' \
        --seq_kd false \
        --lmbda 1.0 \
        --beta 0.0 \
        --tuner_type full \
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
        --max_steps ${MAX_STEPS} \
        --save_steps ${SAVE_STEPS} \
        --save_total_limit 5 \
        --save_only_model true \
        --deepspeed zero3 \
        --gradient_checkpointing true \
        --attn_impl sdpa \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --enable_thinking false \
        --logging_steps 1 \
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
