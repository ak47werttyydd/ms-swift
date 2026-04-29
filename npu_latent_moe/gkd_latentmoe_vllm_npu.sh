#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# NPU port of gkd_latentmoe_vllm.sh.
#
# Two roles co-scheduled on a single 8-NPU host (Atlas 800/900 A2, 910B3/C):
#   * Teacher:  Qwen/Qwen3.5-35B-A3B served by vllm-ascend (releases/v0.18.0).
#   * Student:  Qwen3.5-LatentMoE under npu_latent_moe/ckpt_latentmoe_40l,
#               trained via `swift rlhf --rlhf_type gkd` against the teacher's
#               /v1/chat/completions endpoint.
#
# Differences vs. the CUDA reference script:
#   * CUDA_VISIBLE_DEVICES          → ASCEND_RT_VISIBLE_DEVICES
#   * nvidia-fabricmanager/nvidia-smi checks → npu-smi probe
#   * PYTORCH_CUDA_ALLOC_CONF dropped (no NPU analogue; torch-npu caches differently)
#   * flash_attn (Dao-AILab) is CUDA-only; on NPU, `--attn_impl flash_attn`
#     routes to HF's `flash_attention_2`, which torch-npu accelerates via
#     `use_npu_rmsnorm=True` + `use_grouped_expert_matmul=True` in config.json
#     (per npu_latentmoe_guide.docx).
#
# Prereqs (one-time setup):
#   bash /Users/adrianhwang/Code/vllm-ascend/scripts/setup_npu_env.sh
#
# Run:
#   bash npu_latent_moe/gkd_latentmoe_vllm_npu.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-35B-A3B}"
STUDENT_MODEL="${STUDENT_MODEL:-${SCRIPT_DIR}/ckpt_latentmoe_40l}"
GKD_PLUGIN="${GKD_PLUGIN:-${SCRIPT_DIR}/gkd_plugin.py}"

PHASE1_OUTPUT="${PHASE1_OUTPUT:-${SCRIPT_DIR}/output/gkd_phase1}"

# ─── Teacher vLLM server ────────────────────────────────────────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-4}"
TEACHER_NPUS="${TEACHER_NPUS:-4,5,6,7}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-5000}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.85}"

# ─── Student training ───────────────────────────────────────────────────────
STUDENT_NPUS="${STUDENT_NPUS:-0,1,2,3}"
STUDENT_NPROC="${STUDENT_NPROC:-4}"

# ─── Dataset: presampled JSONL with "response" field ───────────────────────
if [[ -z "${PRESAMPLE_DATA:-}" ]]; then
    PRESAMPLE_DATA_ARGS=(
        # Fill in your presampled teacher-generation shards, e.g.:
        # "${SCRIPT_DIR}/data/teacher_presample_shard0.jsonl"
        # "${SCRIPT_DIR}/data/teacher_presample_shard1.jsonl"
    )
else
    read -ra PRESAMPLE_DATA_ARGS <<< "${PRESAMPLE_DATA}"
fi

if [[ ${#PRESAMPLE_DATA_ARGS[@]} -eq 0 ]]; then
    echo "ERROR: no dataset paths set. Edit PRESAMPLE_DATA_ARGS or pass PRESAMPLE_DATA='path1 path2'." >&2
    exit 1
fi

# ─── Sanity: NPU + CANN ─────────────────────────────────────────────────────
command -v npu-smi >/dev/null 2>&1 || { echo "ERROR: npu-smi not found — CANN not sourced?" >&2; exit 1; }
npu-smi info -l >/dev/null 2>&1   || { echo "ERROR: npu-smi reports no NPUs." >&2; exit 1; }
[[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] || echo "WARN: ASCEND_TOOLKIT_HOME unset — did you source set_env.sh?"

# ─── Teacher lifecycle ──────────────────────────────────────────────────────
start_teacher() {
    echo "=== Starting vLLM teacher on NPUs ${TEACHER_NPUS} (TP=${TEACHER_TP}) ==="
    ASCEND_RT_VISIBLE_DEVICES=${TEACHER_NPUS} \
    vllm serve "${TEACHER_MODEL}" \
        --port "${TEACHER_PORT}" \
        --tensor-parallel-size "${TEACHER_TP}" \
        --max-model-len "${TEACHER_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${TEACHER_GPU_MEM_UTIL}" \
        --max-logprobs "${TEACHER_MAX_LOGPROBS}" \
        --dtype bfloat16 \
        --enforce-eager \
        --trust-remote-code \
        &
    TEACHER_PID=$!
    echo "Teacher PID: ${TEACHER_PID}"

    echo "Waiting for teacher /health (up to 600s; MoE load is slow) ..."
    for i in $(seq 1 600); do
        if curl -sf "http://localhost:${TEACHER_PORT}/health" >/dev/null 2>&1; then
            echo "Teacher server ready after ${i}s"
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

# ─── Student watchdog ──────────────────────────────────────────────────────
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
            echo "ERROR: Teacher server unhealthy; killing student (pgid ${student_pgid})." >&2
            kill -TERM -"${student_pgid}" 2>/dev/null || true
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
    mkdir -p "${phase_name}"

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

# ═══════════════════════════════════════════════════════════════════════════
start_teacher

# ─── Phase 1: Offline GKD (lmbda=0.0, precomputed teacher logits) ─────────
check_teacher
echo "=== Phase 1: Offline GKD (lmbda=0.0) ==="
run_phase "${PHASE1_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    ASCEND_RT_VISIBLE_DEVICES=${STUDENT_NPUS} \
    swift rlhf \
        --rlhf_type gkd \
        --model "${STUDENT_MODEL}" \
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
        --loss_scale all \
        --output_dir "${PHASE1_OUTPUT}"

echo "=== GKD training complete ==="
