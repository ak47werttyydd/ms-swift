#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# GKD for both 40-layer (Sandeep) and 24-layer (Rezaul) LatentMoE models.
# Shares a single vLLM teacher server to avoid double startup overhead.
#
# Usage:
#   bash gkd_latentmoe_both.sh
#   STEPS_40L=200 STEPS_24L=0 bash gkd_latentmoe_both.sh   # 40L only, 200 steps
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Model paths ──────────────────────────────────────────────────────────────
TEACHER_MODEL="${TEACHER_MODEL:-/home/r00914194/models/Qwen3.5-35B-A3B}"
STUDENT_40L="${STUDENT_40L:-${SCRIPT_DIR}/qwen35_latentmoe/sandeep_latentmoe_40layers_original_ckpt}"
STUDENT_24L="${STUDENT_24L:-${SCRIPT_DIR}/qwen35_latentmoe/rezaul_latentmoe_24layers_original_ckpt}"
GKD_PLUGIN="${SCRIPT_DIR}/qwen35_latentmoe/gkd_plugin.py"

OUTPUT_40L="${OUTPUT_40L:-${SCRIPT_DIR}/output/gkd_sandeep_latentmoe_only}"
OUTPUT_24L="${OUTPUT_24L:-${SCRIPT_DIR}/output/gkd_rezaul_latentmoe_24l}"

# ── Steps (0 = skip) ────────────────────────────────────────────────────────
STEPS_40L="${STEPS_40L:-100}"
STEPS_24L="${STEPS_24L:-0}"

# ── Teacher vLLM server config ───────────────────────────────────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-2}"
TEACHER_GPUS="${TEACHER_GPUS:-6,7}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-5000}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.70}"

# ── Student training config ──────────────────────────────────────────────────
STUDENT_GPUS="${STUDENT_GPUS:-0,1,2,3,4,5}"
STUDENT_NPROC="${STUDENT_NPROC:-6}"

# ── Dataset ──────────────────────────────────────────────────────────────────
if [[ -z "${PRESAMPLE_DATA:-}" ]]; then
    PRESAMPLE_DATA_ARGS=(
        "/home/a84400789/ms-swift/output/teacher_presample_shard0.jsonl"
        "/home/a84400789/ms-swift/output/teacher_presample_shard1.jsonl"
        "/home/a84400789/ms-swift/output/teacher_presample_shard2.jsonl"
        "/home/a84400789/ms-swift/output/teacher_presample_shard3.jsonl"
        "/home/a84400789/ms-swift/output/teacher_presample_shard4.jsonl"
        "/home/a84400789/ms-swift/output/teacher_presample_shard0_run3.jsonl"
        "/home/a84400789/ms-swift/output/presample_test.jsonl"
    )
else
    read -ra PRESAMPLE_DATA_ARGS <<< "${PRESAMPLE_DATA}"
fi

# ── Helper: start/stop teacher server ────────────────────────────────────────
TEACHER_PID=0
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
    if [[ ${TEACHER_PID} -ne 0 ]]; then
        echo "=== Stopping teacher server (PID ${TEACHER_PID}) ==="
        kill ${TEACHER_PID} 2>/dev/null || true
        wait ${TEACHER_PID} 2>/dev/null || true
        TEACHER_PID=0
    fi
}

check_teacher() {
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then
        echo "ERROR: Teacher server (PID ${TEACHER_PID}) has died. Aborting."
        exit 1
    fi
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
    shift

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

    echo "=== ${phase_name} completed successfully ==="
}

# ═══════════════════════════════════════════════════════════════════════════════
# START — single teacher, two students
# ═══════════════════════════════════════════════════════════════════════════════
start_teacher

# ── 40-layer (Sandeep) ───────────────────────────────────────────────────────
if [[ ${STEPS_40L} -gt 0 ]]; then
    check_teacher
    echo "=== 40-layer GKD (max_steps=${STEPS_40L}) ==="
    run_phase "${OUTPUT_40L}" \
        env NPROC_PER_NODE=${STUDENT_NPROC} \
        CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
        swift rlhf \
            --rlhf_type gkd \
            --model "${STUDENT_40L}" \
            --model_type qwen3_5_latentmoe \
            --external_plugins "${GKD_PLUGIN}" \
            --teacher_model_server "http://localhost:${TEACHER_PORT}" \
            --dataset "${PRESAMPLE_DATA_ARGS[@]}" \
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
            --warmup_ratio 0.05 \
            --per_device_train_batch_size 3 \
            --gradient_accumulation_steps 13 \
            --learning_rate 1e-5 \
            --max_steps ${STEPS_40L} \
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
            --loss_scale all \
            --output_dir "${OUTPUT_40L}"
    echo "=== 40-layer GKD complete ==="
else
    echo "=== Skipping 40-layer (STEPS_40L=0) ==="
fi

# ── 24-layer (Rezaul) ───────────────────────────────────────────────────────
if [[ ${STEPS_24L} -gt 0 ]]; then
    check_teacher
    echo "=== 24-layer GKD (max_steps=${STEPS_24L}) ==="
    run_phase "${OUTPUT_24L}" \
        env NPROC_PER_NODE=${STUDENT_NPROC} \
        CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
        swift rlhf \
            --rlhf_type gkd \
            --model "${STUDENT_24L}" \
            --model_type qwen3_5_latentmoe \
            --external_plugins "${GKD_PLUGIN}" \
            --teacher_model_server "http://localhost:${TEACHER_PORT}" \
            --dataset "${PRESAMPLE_DATA_ARGS[@]}" \
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
            --warmup_ratio 0.05 \
            --per_device_train_batch_size 6 \
            --gradient_accumulation_steps 7 \
            --learning_rate 1e-5 \
            --max_steps ${STEPS_24L} \
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
            --loss_scale all \
            --output_dir "${OUTPUT_24L}"
    echo "=== 24-layer GKD complete ==="
else
    echo "=== Skipping 24-layer (STEPS_24L=0) ==="
fi

echo "=== All GKD training complete ==="
