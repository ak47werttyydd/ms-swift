#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# GKD Training Script for Qwen3.5-LatentMoE-MLA (custom architecture)
#
# Adapted from gkd_qwen35_layer_pruned_vllm_compact.sh
# Uses vLLM teacher server + ms-swift GKD with external plugin registration.
#
# Prerequisites:
#   pip install ms-swift[rlhf] flash-linear-attention causal-conv1d
#   Student checkpoint must contain: config.json (with auto_map),
#     configuration_qwen3_5_latentmoe_mla.py, modeling_qwen3_5_latentmoe_mla.py
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Model paths ──────────────────────────────────────────────────────────────
# STUDENT_MODEL: HF checkpoint dir with config.json + auto_map + weights
# Must contain: config.json, configuration_qwen3_5_latentmoe_mla.py,
#               modeling_qwen3_5_latentmoe_mla.py, *.safetensors, tokenizer files
TEACHER_MODEL="${TEACHER_MODEL:-/home/r00914194/models/Qwen3.5-35B-A3B}"
STUDENT_MODEL="${STUDENT_MODEL:-/home/a84400789/Huawei_PCL_MLLMs/hf_custom_model/initial_ckpt}"

PHASE1_OUTPUT="${PHASE1_OUTPUT:-/home/a84400789/ms-swift/output/gkd_hq_10ba1d5b}"

# ── Plugin path (auto-resolved relative to this script) ─────────────────────
GKD_PLUGIN="${SCRIPT_DIR}/gkd_plugin.py"

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
# Presample JSONL files with "response" field (teacher-generated completions)
# Override by setting PRESAMPLE_DATA as a space-separated string of paths.
if [[ -z "${PRESAMPLE_DATA:-}" ]]; then
    PRESAMPLE_DATA_ARGS=(
        # Add your presample data paths here, e.g.:
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
# PHASE 1: Offline GKD (lmbda=0.0, dataset responses with teacher logits)
# ═══════════════════════════════════════════════════════════════════════════════
check_teacher
echo "=== Phase 1: Offline GKD (lmbda=0.0) ==="
run_phase "${PHASE1_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        --rlhf_type gkd \
        --model "${STUDENT_MODEL}" \
        --model_type qwen3_5_latentmoe_mla \
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
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 5 \
        --learning_rate 1e-5 \
        --num_train_epochs 1 \
        --save_steps 50 \
        --save_total_limit 10 \
        --save_only_model true \
        --deepspeed zero3 \
        --attn_impl sdpa \
        --dataloader_num_workers 4 \
        --dataset_num_proc 8 \
        --enable_thinking false \
        --logging_steps 10 \
        --load_from_cache_file true \
        --loss_scale all \
        --output_dir "${PHASE1_OUTPUT}"

echo "=== GKD training complete ==="
