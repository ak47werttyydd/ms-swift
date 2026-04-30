#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# GKD 24-layer Rezaul LatentMoE — freeze200 then thaw200
#
# Phase 1 (200 steps): freeze attention + linear_attn, load from checkpoint-200
# Phase 2 (200 steps): resume from Phase-1 checkpoint, all weights unfrozen
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Model paths ──────────────────────────────────────────────────────────────
TEACHER_MODEL="${TEACHER_MODEL:-/home/r00914194/models/Qwen3.5-35B-A3B}"
STUDENT_24L="${STUDENT_24L:-/home/a84400789/ms-swift/output/gkd_rezaul_latentmoe_24l_from900_top16_aux_freezeattn/v0-20260423-213345/checkpoint-200}"
GKD_PLUGIN="${SCRIPT_DIR}/qwen35_latentmoe/gkd_plugin.py"

OUTPUT_24L="${OUTPUT_24L:-${SCRIPT_DIR}/output/gkd_rezaul_latentmoe_24l_freeze200_thaw200}"
OUTPUT_PHASE1="${OUTPUT_PHASE1:-${OUTPUT_24L}/phase1_freeze}"
OUTPUT_PHASE2="${OUTPUT_PHASE2:-${OUTPUT_24L}/phase2_thaw}"

# ── Phase step counts ────────────────────────────────────────────────────────
PHASE1_STEPS="${PHASE1_STEPS:-200}"   # frozen attn + linear_attn
PHASE2_STEPS="${PHASE2_STEPS:-2000}"   # all weights unfrozen

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
# Phase 1 sees only the "before-200step" slice of shard2 (~50,400 samples ≈
# 200 steps × 252 effective batch). Phase 2 gets the "after-200step" tail of
# shard2 plus all the other shards, guaranteeing no sample is re-used.
PHASE1_DATA_ARGS=(
    "/home/a84400789/ms-swift/output/teacher_presample_shard2_before200step.jsonl"
)
PHASE2_DATA_ARGS=(
    "/home/a84400789/ms-swift/output/teacher_presample_shard2_after200step.jsonl"
    "/home/a84400789/ms-swift/output/teacher_presample_shard3.jsonl"
    "/home/a84400789/ms-swift/output/teacher_presample_shard4.jsonl"
    "/home/a84400789/ms-swift/output/teacher_presample_shard0_run3.jsonl"
    "/home/a84400789/ms-swift/output/presample_test.jsonl"
    "/home/a84400789/ms-swift/output/teacher_presample_shard1.jsonl"
)

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
# START
# ═══════════════════════════════════════════════════════════════════════════════
start_teacher
check_teacher

# ── Phase 1: freeze attention + linear_attn ──────────────────────────────────
echo "=== Phase 1: freeze attn/linear_attn, ${PHASE1_STEPS} steps from $(basename ${STUDENT_24L}) ==="
run_phase "phase1-freeze" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        --rlhf_type gkd \
        --model "${STUDENT_24L}" \
        --model_type qwen3_5_latentmoe \
        --external_plugins "${GKD_PLUGIN}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --dataset "${PHASE1_DATA_ARGS[@]}" \
        --seq_kd false \
        --lmbda 0.0 \
        --beta 0.5 \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --train_type full \
        --freeze_vit true \
        --freeze_aligner true \
        --freeze_llm false \
        --freeze_parameters_regex '.*(linear_attn|self_attn|\.norm\.).*' \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length 4607 \
        --max_completion_length 1 \
        --truncation_strategy left \
        --warmup_ratio 0.05 \
        --per_device_train_batch_size 6 \
        --gradient_accumulation_steps 7 \
        --learning_rate 1e-5 \
        --max_steps ${PHASE1_STEPS} \
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
        --output_dir "${OUTPUT_PHASE1}"

#--lr_scheduler_kwargs '{"min_lr_rate": 0.1}'   # 衰到峰值的 10% should append it to avoid lr approaches 0


# ── Phase 2: unfreeze all, load weights from Phase-1 checkpoint ──────────────
# Load Phase-1 weights as a fresh training run (not --resume_from_checkpoint)
# because Phase 1 uses --save_only_model (no optimizer/dataloader state) and
# the Phase-2 dataset differs, so HF's batch-skip alignment would misbehave.
PHASE1_CKPT=$(ls -d "${OUTPUT_PHASE1}"/v*/checkpoint-${PHASE1_STEPS} 2>/dev/null | tail -n1)
if [[ -z "${PHASE1_CKPT}" || ! -d "${PHASE1_CKPT}" ]]; then
    echo "ERROR: could not locate Phase-1 checkpoint under ${OUTPUT_PHASE1}" >&2
    exit 1
fi

echo "=== Phase 2: all weights unfrozen, load from ${PHASE1_CKPT}, ${PHASE2_STEPS} steps ==="
run_phase "phase2-thaw" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        --rlhf_type gkd \
        --model "${PHASE1_CKPT}" \
        --model_type qwen3_5_latentmoe \
        --external_plugins "${GKD_PLUGIN}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --dataset "${PHASE2_DATA_ARGS[@]}" \
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
        --max_steps ${PHASE2_STEPS} \
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
        --output_dir "${OUTPUT_PHASE2}"

#--lr_scheduler_kwargs '{"min_lr_rate": 0.1}'   # 衰到峰值的 10% should append it to avoid lr approaches 0
echo "=== GKD training complete (freeze200 + thaw200) ==="
