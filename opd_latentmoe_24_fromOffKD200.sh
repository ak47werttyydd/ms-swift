#!/usr/bin/env bash
set -euo pipefail

TEACHER_MODEL="${TEACHER_MODEL:-/home/a84400789/local_models/qwen3_5_35b_a3b_teacher/Qwen3.5-35B-A3B}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/a84400789/local_models/gkd_rezaul_latentmoe_24l_16actexp/checkpoint-200}"

PHASE1_OUTPUT="${PHASE1_OUTPUT:-/home/a84400789/ms-swift/output/opd_latentmoe_24l_from_offpolicy1100}"

# ── Teacher vLLM server config ───────────────────────────────────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-2}"
TEACHER_GPUS="${TEACHER_GPUS:-6,7}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-5200}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.70}"

# ── Student training config ──────────────────────────────────────────────────
STUDENT_GPUS="${STUDENT_GPUS:-0,1,2,3,4,5}"
STUDENT_NPROC="${STUDENT_NPROC:-6}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
STUDENT_VLLM_MAX_MODEL_LEN=$((MAX_LENGTH + MAX_COMPLETION_LENGTH))

# ── Dataset ──────────────────────────────────────────────────────────────────
mapfile -t DATASET < <(ls -1 /dev/shm/dataset/fineweb-edu-100BT/sample/100BT/*.parquet | sort)

# ── Helper: start/stop teacher server ────────────────────────────────────────
start_teacher() {
    if ! pgrep -x nv-fabricmanager > /dev/null 2>&1; then
        echo "WARNING: nvidia-fabricmanager is not running."
        echo "  NVSwitch (NV18) topology requires it. Fix: sudo systemctl start nvidia-fabricmanager"
        nvidia-smi -pm 1 2>/dev/null || echo "  (persistence mode: no sudo)"
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
        if curl -s http://localhost:${TEACHER_PORT}/health > /dev/null 2>&1; then
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

# ── Watchdog & process-group management ──────────────────────────────────────
STUDENT_PID=0
WATCHDOG_PID=0

_global_cleanup() {
    [[ ${WATCHDOG_PID} -ne 0 ]] && kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    [[ ${STUDENT_PID}  -ne 0 ]] && kill -TERM -"${STUDENT_PID}"  2>/dev/null || true
    stop_teacher
}
trap '_global_cleanup; exit' EXIT INT TERM

check_teacher() {
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then
        echo "ERROR: Teacher server (PID ${TEACHER_PID}) has died. Aborting."
        exit 1
    fi
}

_watchdog_loop() {
    local student_pid=$1
    while sleep 30; do
        if ! curl -sf "http://localhost:${TEACHER_PORT}/health" >/dev/null 2>&1; then
            echo "ERROR: Teacher server unhealthy; killing student (pgid ${student_pid})." >&2
            kill -TERM -"${student_pid}" 2>/dev/null || true
            exit 1
        fi
    done
}

run_phase() {
    local phase_name=$1
    local marker="${phase_name}/.done"
    shift
    if [[ -f "${marker}" ]]; then
        echo "=== ${phase_name} already completed, skipping ==="
        return 0
    fi

    setsid "$@" &
    STUDENT_PID=$!

    _watchdog_loop "${STUDENT_PID}" &
    WATCHDOG_PID=$!

    if ! wait "${STUDENT_PID}"; then
        kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
        wait        "${WATCHDOG_PID}" 2>/dev/null || true
        STUDENT_PID=0; WATCHDOG_PID=0
        echo "ERROR: Phase ${phase_name} failed." >&2
        exit 1
    fi

    kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    wait        "${WATCHDOG_PID}" 2>/dev/null || true
    STUDENT_PID=0; WATCHDOG_PID=0

    touch "${marker}"
    echo "=== ${phase_name} completed successfully ==="
}

# ═══════════════════════════════════════════════════════════════════════════════
# START
# ═══════════════════════════════════════════════════════════════════════════════
start_teacher

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: On-Policy GKD (lmbda=1.0, student rollouts + teacher logprobs)
# ═══════════════════════════════════════════════════════════════════════════════
check_teacher
echo "=== Phase 1: On-Policy GKD (lmbda=1.0) — 24-layer student ==="
run_phase "${PHASE1_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        --rlhf_type gkd \
        --model "${STUDENT_MODEL}" \
        --model_type qwen3_5_latentmoe \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --dataset "${DATASET[@]}" \
        --streaming true \
        --columns '{"text":"query"}' \
        --seq_kd false \
        --lmbda 1.0 \
        --beta 0.5 \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --train_type full \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length ${MAX_LENGTH} \
        --max_completion_length ${MAX_COMPLETION_LENGTH} \
        --truncation_strategy right \
        --warmup_steps 30 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 10 \
        --learning_rate 1e-5 \
        --max_steps 10000 \
        --save_steps 50 \
        --save_total_limit 10 \
        --save_only_model true \
        --deepspeed zero3 \
        --attn_impl flash_attn \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --enable_thinking false \
        --logging_steps 10 \
        --load_from_cache_file true \
        --loss_scale all \
        --use_vllm true \
        --vllm_mode colocate \
        --vllm_enforce_eager true \
        --vllm_gpu_memory_utilization 0.65 \
        --vllm_tensor_parallel_size 1 \
        --vllm_max_model_len ${STUDENT_VLLM_MAX_MODEL_LEN} \
        --sleep_level 2 \
        --offload_model true \
        --offload_optimizer true \
        --move_model_batches 24 \
        --output_dir "${PHASE1_OUTPUT}"


#--gradient_checkpointing true  #--gradient_checkpointing true 

#--warmup_ratio 0.05 \
#--vllm_enforce_eager true \
#--per_device_train_batch_size 6 \
#--gradient_accumulation_steps 7 \

echo "=== GKD training complete ==="
