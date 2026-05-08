#!/usr/bin/env bash
# ============================================================================
# Full On-Policy Distillation:
#   Qwen3.5-35B-A3B (teacher, vLLM server) -> pruned 10.4B-A2.4B (student)
#
# GPU layout (8× H100 80GB):
#   GPU 6-7  → vLLM teacher server (TP=2, ~35GB/GPU for 35B MoE weights)
#   GPU 0-5  → Student training (6 GPUs, DeepSpeed ZeRO-3) + colocated student
#              vLLM for on-policy rollouts.
#
# On-policy schedule (single phase, lmbda=1.0, beta=1.0):
#   For every batch the student samples a completion via its colocated vLLM;
#   the teacher (remote vLLM server) returns top-k logprobs for the sampled
#   tokens and the student is trained to match that distribution.
#
# Dataset:
#   /dev/shm/dataset/nemotron-cc-v2.1-high-quality/High-Quality/*.parquet
#   (all shards; each row's `text` is used as the query/prefix for student rollout)
#
# Teacher context:
#   TEACHER_MAX_MODEL_LEN must be ≥ max_length + max_completion_length + 1.
#
# Packing: Not supported for GKD (requires per-token logprob alignment).
# Temperature: T=1.0 — preserves teacher's natural distribution for student rollouts.
#
# Usage:
#   bash gkd_qwen35_vllm_opd.sh |& tee gkd_qwen35_vllm_opd.log
# ============================================================================
set -euo pipefail

# Set huggingface token (export so all child processes inherit it)
# export HF_TOKEN=<replace HF_TOKEN>

# ── Model paths ──────────────────────────────────────────────────────────────
TEACHER_MODEL=/home/a84400789/local_models/qwen3_5_35b_a3b_teacher/Qwen3.5-35B-A3B

#STUDENT_MODEL=/home/a84400789/local_models/our_1st_gkd_model/checkpoint-2300
#OPD_OUTPUT=output/opd_1st_gkd_ckpt_2300

STUDENT_MODEL=/home/a84400789/local_models/gkd_rezaul_latentmoe_24l_16actexp/checkpoint-200
OPD_OUTPUT=output/opd_lmoe_24l_16actexp_ckpt200

# ── Student training config ──────────────────────────────────────────────────
STUDENT_GPUS="0,1,2,3,4,5"
STUDENT_NPROC=6
MAX_LENGTH=4096
MAX_COMPLETION_LENGTH=1024
STUDENT_VLLM_MAX_MODEL_LEN=$((MAX_LENGTH + MAX_COMPLETION_LENGTH))
MBS=4
GRAD_ACC=30
SAVE_STEPS=50

# ── Teacher vLLM server config ───────────────────────────────────────────────
TEACHER_PORT=8000
TEACHER_TP=2                       # tensor-parallel across GPU 0,1
TEACHER_GPUS="6,7"
TEACHER_MAX_LOGPROBS=64            # top-k logprobs for GKD
TEACHER_MAX_MODEL_LEN=5200         # must be ≥ max_length + max_completion_length + 1
                                   # = 4096 + 1024 + 1 = 5121; 5200 satisfies this
TEACHER_GPU_MEM_UTIL=0.80          # --enforce-eager is set; CUDA graph capture disabled so
                                   # memory is not inflated by graph capture overhead.

# ── Dataset ──────────────────────────────────────────────────────────────────
# Full Nemotron-CC v2.1 high-quality corpus — use all parquet shards.
DATA_ROOT="/dev/shm/dataset/nemotron-cc-v2.1-high-quality/High-Quality"

mapfile -t DATASET < <(python3 -c "
import glob
for f in sorted(glob.glob('${DATA_ROOT}/*.parquet')):
    print(f)
")
echo "=== Found ${#DATASET[@]} parquet shards under ${DATA_ROOT} ==="

# ── Helper: start/stop teacher server ────────────────────────────────────────
start_teacher() {
    # Error 802 (cudaErrorSystemNotReady) on NVSwitch H100 requires nvidia-fabricmanager.
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
        --disable-custom-all-reduce \
        --enforce-eager \
        &
    TEACHER_PID=$!
    echo "Teacher PID: ${TEACHER_PID}"

    # Wait for server readiness
    echo "Waiting for teacher server to be ready..."
    for i in $(seq 1 600); do
        if curl -s http://localhost:${TEACHER_PORT}/health > /dev/null 2>&1; then
            echo "Teacher server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "ERROR: Teacher server did not become ready in 600s"
    kill ${TEACHER_PID} 2>/dev/null || true
    exit 1
}

stop_teacher() {
    echo "=== Stopping teacher server (PID ${TEACHER_PID}) ==="
    kill ${TEACHER_PID} 2>/dev/null || true
    wait ${TEACHER_PID} 2>/dev/null || true
}

# ── Watchdog & process-group management ──────────────────────────────────────

# Global PIDs so the EXIT trap can clean up whatever is currently running.
STUDENT_PID=0
WATCHDOG_PID=0

_global_cleanup() {
    [[ ${WATCHDOG_PID} -ne 0 ]] && kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    # setsid gave the student its own process group (PGID == PID), so -PID kills
    # the whole tree (torchrun + all GPU workers), not just the launcher.
    [[ ${STUDENT_PID}  -ne 0 ]] && kill -TERM -"${STUDENT_PID}"  2>/dev/null || true
    stop_teacher
}
trap '_global_cleanup; exit' EXIT INT TERM

# Helper: pre-flight check that teacher is still alive before starting a phase.
check_teacher() {
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then
        echo "ERROR: Teacher server (PID ${TEACHER_PID}) has died. Aborting."
        exit 1
    fi
}

# Background watchdog: poll /health every 30 s and kill the student process
# group immediately if the teacher becomes unhealthy.
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

# Helper: run a phase with completion marker, per-phase watchdog, and
# process-group kill on failure.
run_phase() {
    local phase_name=$1
    local marker="${phase_name}/.done"
    shift
    if [[ -f "${marker}" ]]; then
        echo "=== ${phase_name} already completed, skipping ==="
        return 0
    fi

    # Launch student in its own process group via setsid.
    # setsid makes the child the session leader → its PGID == its PID.
    # kill -TERM -$STUDENT_PID therefore reaches every torchrun worker,
    # not just the swift launcher.
    setsid "$@" &
    STUDENT_PID=$!

    # Start watchdog; pass student PID so it knows which group to kill.
    _watchdog_loop "${STUDENT_PID}" &
    WATCHDOG_PID=$!

    # Wait for student; propagate non-zero exit as a hard failure.
    if ! wait "${STUDENT_PID}"; then
        kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
        wait        "${WATCHDOG_PID}" 2>/dev/null || true
        STUDENT_PID=0; WATCHDOG_PID=0
        echo "ERROR: Phase ${phase_name} failed." >&2
        exit 1
    fi

    # Student finished cleanly — stop watchdog.
    kill -TERM "${WATCHDOG_PID}" 2>/dev/null || true
    wait        "${WATCHDOG_PID}" 2>/dev/null || true
    STUDENT_PID=0; WATCHDOG_PID=0

    touch "${marker}"
    echo "=== ${phase_name} completed successfully ==="
}

# start teacher inference by vllm
start_teacher

# ══════════════════════════════════════════════════════════════════════════════
# Full On-Policy Distillation (lmbda=1.0, beta=1.0)
#   - Student samples every completion via its colocated vLLM engine.
#   - Teacher (remote vLLM server) provides top-k logprobs on those completions.
#   - Dataset `text` is used as the query/prefix; truncation_strategy=right
#     keeps the prefix from the start of each document.
#   - bs=4, grad_accum=30 → 4×30×5120×6 ≈ 3.69M tokens/step
#     (max_length + max_completion_length = 4096 + 1024 = 5120)
# ══════════════════════════════════════════════════════════════════════════════
check_teacher
echo "=== Full on-policy distillation on nemotron-cc-v2.1-high-quality ==="
run_phase "${OPD_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        --rlhf_type gkd \
        --model_type qwen3_5_latentmoe \
        --model "${STUDENT_MODEL}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --dataset "${DATASET[@]}" \
        --streaming true \
        --columns '{"text":"query"}' \
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
        --attn_impl flash_attn \
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
        --output_dir "${OPD_OUTPUT}"

stop_teacher
echo "=== Training complete. Final model: ${OPD_OUTPUT} ==="
