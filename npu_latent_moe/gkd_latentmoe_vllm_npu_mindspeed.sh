#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MindSpeed (Megatron) variant of gkd_latentmoe_vllm_npu.sh.
#
# Same teacher role (vllm-ascend serving Qwen3.5-35B-A3B), different student:
# instead of `swift rlhf --deepspeed zero3`, the student is trained via
# `megatron rlhf` with MindSpeed + mcore-bridge. This uses tensor / expert /
# pipeline parallelism rather than ZeRO sharding.
#
# Changes vs gkd_latentmoe_vllm_npu.sh:
#   * Entry point:       `swift rlhf`  ->  `megatron rlhf`
#   * Batch sizing:      --per_device_train_batch_size + grad_accum
#                        ->  --micro_batch_size + --global_batch_size
#   * Epochs:            --num_train_epochs   ->  --train_iters
#   * Parallelism:       --deepspeed zero3
#                        ->  --tensor_model_parallel_size / expert_mp / pp
#   * Attention flag:    --attn_impl flash_attn  ->  --attention_backend flash
#   * Output dir:        --output_dir            ->  --save
#   * LR flag:           --learning_rate         ->  --lr
#   * Adds:              --finetune --no_save_optim --no_save_rng
#                        --recompute_granularity selective
#                        --sequence_parallel true
#   * Plugin:            gkd_plugin.py           ->  gkd_plugin_mindspeed.py
#
# Prereqs (one-time):
#   bash /Users/adrianhwang/Code/vllm-ascend/scripts/setup_npu_env_mindspeed.sh
#
# IMPORTANT: This script will NOT work for the LatentMoE student until
# `gkd_plugin_mindspeed.py`'s Qwen35LatentMoeBridge is implemented.
# mcore-bridge has no built-in converter for `qwen3_5_latentmoe`. See the
# docstring in gkd_plugin_mindspeed.py for what to write.
#
# Run:
#   bash npu_latent_moe/gkd_latentmoe_vllm_npu_mindspeed.sh
# -----------------------------------------------------------------------------
set -euo pipefail

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-35B-A3B}"
STUDENT_MODEL="${STUDENT_MODEL:-${SCRIPT_DIR}/ckpt_latentmoe_40l}"
GKD_PLUGIN="${GKD_PLUGIN:-${SCRIPT_DIR}/gkd_plugin_mindspeed.py}"

PHASE1_OUTPUT="${PHASE1_OUTPUT:-${SCRIPT_DIR}/output/gkd_phase1_mindspeed}"

# ─── Teacher vLLM server (unchanged vs DeepSpeed variant) ───────────────────
TEACHER_PORT="${TEACHER_PORT:-8000}"
TEACHER_TP="${TEACHER_TP:-4}"
TEACHER_NPUS="${TEACHER_NPUS:-4,5,6,7}"
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-5000}"
TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.85}"

# ─── Student training (Megatron) ────────────────────────────────────────────
STUDENT_NPUS="${STUDENT_NPUS:-0,1,2,3}"
STUDENT_NPROC="${STUDENT_NPROC:-4}"

# Parallelism layout for the student. With 4 NPUs the simplest useful split is
# TP=2, EP=2, PP=1 (world_size = TP * EP * PP must divide NPROC_PER_NODE).
# For LatentMoE with 256 experts, EP=2 is already tight; raise only if you
# scale to 8+ NPUs.
TP_SIZE="${TP_SIZE:-2}"
EP_SIZE="${EP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"

# Global batch determines optimizer step cadence. Keep total tokens comparable
# to the DeepSpeed config: per_device_bs=3 * grad_accum=13 * 4 NPUs = 156.
# With Megatron we express that as micro=3, global=156, which yields
# 156 / (3 * dp_world=1) = 52 accumulation steps/rank — match by tuning.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-156}"
TRAIN_ITERS="${TRAIN_ITERS:-1000}"
LR="${LR:-1e-5}"
LR_WARMUP_FRACTION="${LR_WARMUP_FRACTION:-0.05}"

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

# ─── Sanity: NPU + CANN + Megatron env ─────────────────────────────────────
command -v npu-smi >/dev/null 2>&1 || { echo "ERROR: npu-smi not found — CANN not sourced?" >&2; exit 1; }
npu-smi info -l >/dev/null 2>&1   || { echo "ERROR: npu-smi reports no NPUs." >&2; exit 1; }
[[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] || echo "WARN: ASCEND_TOOLKIT_HOME unset — did you source set_env.sh?"
[[ -n "${MEGATRON_LM_PATH:-}" ]] || { echo "ERROR: MEGATRON_LM_PATH unset. Activate the mindspeed conda env (it sets this in activate.d/)." >&2; exit 1; }
command -v megatron >/dev/null 2>&1 || { echo "ERROR: \`megatron\` CLI not found (provided by ms-swift + mcore-bridge)." >&2; exit 1; }

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

    echo "Waiting for teacher /health (up to 600s; MoE cold start is slow) ..."
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

# ─── Student watchdog ───────────────────────────────────────────────────────
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

# ─── Offline GKD (lmbda=0.0, precomputed teacher logits) — Megatron ────────
check_teacher
echo "=== Offline GKD via megatron rlhf (TP=${TP_SIZE} EP=${EP_SIZE} PP=${PP_SIZE}) ==="
run_phase "${PHASE1_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    ASCEND_RT_VISIBLE_DEVICES=${STUDENT_NPUS} \
    megatron rlhf \
        --rlhf_type gkd \
        --model "${STUDENT_MODEL}" \
        --model_type qwen3_5_latentmoe \
        --external_plugins "${GKD_PLUGIN}" \
        --teacher_model_server "http://localhost:${TEACHER_PORT}" \
        --gkd_logits_topk ${TEACHER_MAX_LOGPROBS} \
        --dataset "${PRESAMPLE_DATA_ARGS[@]}" \
        --seq_kd false \
        --lmbda 0.0 \
        --beta 0.5 \
        --tensor_model_parallel_size ${TP_SIZE} \
        --expert_model_parallel_size ${EP_SIZE} \
        --pipeline_model_parallel_size ${PP_SIZE} \
        --context_parallel_size ${CP_SIZE} \
        --sequence_parallel true \
        --attention_backend flash \
        --recompute_granularity selective \
        --torch_dtype bfloat16 \
        --temperature 1.0 \
        --max_length 4607 \
        --max_completion_length 1 \
        --truncation_strategy left \
        --micro_batch_size ${MICRO_BATCH_SIZE} \
        --global_batch_size ${GLOBAL_BATCH_SIZE} \
        --train_iters ${TRAIN_ITERS} \
        --lr ${LR} \
        --lr_warmup_fraction ${LR_WARMUP_FRACTION} \
        --save_steps 100 \
        --save_total_limit 10 \
        --logging_steps 10 \
        --enable_thinking false \
        --load_from_cache_file true \
        --loss_scale all \
        --finetune \
        --no_save_optim \
        --no_save_rng \
        --save "${PHASE1_OUTPUT}"

echo "=== GKD training complete (MindSpeed) ==="
