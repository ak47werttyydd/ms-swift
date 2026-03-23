#!/usr/bin/env bash
# ============================================================================
# GKD: Qwen3.5-35B-A3B (teacher, vLLM) -> pruned 10.4B-A2.4B (student, ZeRO-3)
#
# GPU layout (8× H100 80GB):
#   GPU 0-1  → vLLM teacher server (TP=2, ~35GB/GPU for 35B MoE weights)
#   GPU 2-7  → Student training (6 GPUs, DeepSpeed ZeRO-3)
#              Phase 1-2: seq_kd (teacher generates completions, lmbda=0)
#              Phase 3-4: + vLLM rollout for on-policy generation (lmbda>0)
#
# VRAM budget per student GPU (ZeRO-3, 10.4B model):
#   Weights:    20.8GB / 6 ≈  3.5 GB
#   Gradients:  20.8GB / 6 ≈  3.5 GB
#   Optimizer:  10.4B × 12B / 6 ≈ 20.8 GB  (FP32 copy + Adam states)
#   Activations (grad ckpt, bs=4): ~20-40 GB
#   Total Phase 1-2 (bs=4):        ~48-68 GB / 81.5 GB  ← ~60-84%, target 90%
#   Total Phase 3-4 (bs=2):        ~38-57 GB + vLLM 30% non-overlapping
#
# Why 2 GPUs for teacher?
#   Qwen3.5-35B-A3B has 35B total params (all experts loaded).
#   BF16: 35B × 2B = ~70GB → won't fit in 1× H100 80GB with KV cache.
#   TP=2: ~35GB/GPU → ~45GB free per GPU for KV cache + overhead.
#
# Four-phase schedule:
#   Phase 1 (seq_kd, lmbda=0.0, ~70B tokens): Mode 2 sequential KD — teacher
#            generates completions; student learns from teacher-sampled sequences.
#            First half of off-policy data (49 files). Avoids Arrow cache bloat.
#   Phase 2 (lmbda=0.0, ~70B tokens): Mode 3 offline KD — dataset text as
#            response, teacher provides logprob supervision. Second half (49 files).
#   Phase 3 (lmbda=0.3, ~40B tokens): mixed on/off-policy
#   Phase 4 (lmbda=1.0, ~30B tokens): full on-policy, reduce exposure bias
#
# Token budget:
#   Phase 1-2: 6 GPUs, bs=4, grad_accum=21, avg ~4k tokens/sample
#              → 4×21×4096×6 ≈ 2.06M tokens/step
#   Phase 3-4: 6 GPUs, bs=2, grad_accum=42, avg ~4k tokens/sample
#              → 2×42×4096×6 ≈ 2.06M tokens/step
#
# Eval: Not added (streaming mode, no natural eval split in fineweb-edu).
#       Add --val_dataset if you want Phase 3/4 monitoring.
#
# Packing: Not supported for GKD (requires per-token logprob alignment).
#
# Temperature: T=1.0 throughout — preserves teacher's natural distribution
#              for seq_kd (Phase 1-2) and on-policy student generation (Phase 3-4).
#
# Usage:
#   bash gkd_qwen35_vllm.sh |& tee gkd_qwen35_vllm.log
# ============================================================================
set -euo pipefail

# Set huggingface token (export so all child processes inherit it)
# export HF_TOKEN=<replace HF_TOKEN>

# ── Model paths ──────────────────────────────────────────────────────────────
TEACHER_MODEL=Qwen/Qwen3.5-35B-A3B
STUDENT_MODEL=/home/r00914194/PruneMe/qwen35_exp80_layer_drop_082331

PHASE1_OUTPUT=output/gkd_9b_phase1
PHASE2_OUTPUT=output/gkd_9b_phase2
PHASE3_OUTPUT=output/gkd_9b_phase3
PHASE4_OUTPUT=output/gkd_9b_phase4

# ── Teacher vLLM server config ───────────────────────────────────────────────
TEACHER_PORT=8000
TEACHER_TP=2                       # tensor-parallel across GPU 0,1
TEACHER_GPUS="0,1"
TEACHER_MAX_LOGPROBS=64            # top-k logprobs for GKD
TEACHER_MAX_MODEL_LEN=8193         # must be > max_length + max_completion_length of students across all phases
                                   # vLLM requires input_tokens + max_tokens(1) <= max_model_len
                                   # Phase 2: 4096+4096=8192 → need 8193; Phase 3/4: 4096+2048=6144
TEACHER_GPU_MEM_UTIL=0.70          # 0.90 leaves only ~8.5GB headroom; CUDA graph capture for
                                   # max_model_len=8192 peaks higher than 4608 → OOM at 0.90
                                   ## --enforce-eager disables CUDA graph capture

# ── Student training config ──────────────────────────────────────────────────
STUDENT_GPUS="2,3,4,5,6,7"
STUDENT_NPROC=6
STUDENT_VLLM_MAX_MODEL_LEN=6144  # max_length(4096) + max_completion_length(2048) for Phase 3/4

# ── Dataset splits ───────────────────────────────────────────────────────────
# 140 parquet files: 70% (98) → Phase 1-2, 20% (28) → Phase 3, 10% (14) → Phase 4
# Phase 1-2 each get 49 files to control Arrow cache size
DATA_ROOT="/dev/shm/dataset/fineweb-edu-100BT/sample/100BT"

mapfile -t _ALL < <(python3 -c "
import glob
for f in sorted(glob.glob('${DATA_ROOT}/*.parquet')):
    print(f)
")
_N=${#_ALL[@]}
_N1=$(python3 -c "print(int($_N * 0.70))")   # 98
_N2=$(python3 -c "print(int($_N * 0.20))")   # 28

DATASET1=("${_ALL[@]:0:49}")
DATASET2=("${_ALL[@]:49:49}")
DATASET3=("${_ALL[@]:$_N1:$_N2}")
DATASET4=("${_ALL[@]:$((_N1+_N2))}")

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
        --enforce-eager \
        &
    TEACHER_PID=$!
    echo "Teacher PID: ${TEACHER_PID}"

    # Wait for server readiness
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

trap 'stop_teacher; exit' EXIT INT TERM

# Helper: check teacher is still alive, abort if crashed
check_teacher() {
    if ! kill -0 ${TEACHER_PID} 2>/dev/null; then
        echo "ERROR: Teacher server (PID ${TEACHER_PID}) has died. Aborting."
        exit 1
    fi
}

# Helper: run a phase with completion marker (skip if already done)
run_phase() {
    local phase_name=$1
    local marker="${phase_name}/.done"
    shift
    if [[ -f "${marker}" ]]; then
        echo "=== ${phase_name} already completed, skipping ==="
        return 0
    fi
    "$@"
    touch "${marker}"
    echo "=== ${phase_name} completed successfully ==="
}

# start teacher inference by vllm
start_teacher

# ── Common training args ─────────────────────────────────────────────────────
# Note: --teacher_model_server replaces --teacher_model + --teacher_deepspeed
#       --gkd_logits_topk activates top-k KD (required for server mode)
#       --temperature applies to all generation (teacher seq_kd + student on-policy)
COMMON_ARGS=(
    --rlhf_type gkd
    --model_type qwen3_5_moe
    --teacher_model_server "http://localhost:${TEACHER_PORT}"
    --gkd_logits_topk ${TEACHER_MAX_LOGPROBS}
    --train_type full
    --columns '{"text":"response"}'
    --freeze_vit true
    --freeze_aligner true
    --freeze_llm false
    --torch_dtype bfloat16
    --temperature 1.0
    --warmup_ratio 0.05
    --max_length 4096
    --truncation_strategy right
    --save_steps 200
    --save_total_limit 2
    --save_only_model true
    --deepspeed zero3
    --attn_impl flash_attn
    --dataloader_num_workers 4
    --dataset_num_proc 8
    --streaming true
    --gradient_checkpointing true
    --enable_thinking false
    --logging_steps 10
)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: seq_kd=True, lmbda=0.0 – Mode 2 Sequential KD (files 1-49)
#   Teacher generates completions on-the-fly; student learns from teacher
#   sampled sequences instead of raw dataset text.
#   bs=4, grad_accum=21 → 4×21×4096×6 ≈ 2.06M tokens/step
# ══════════════════════════════════════════════════════════════════════════════

# check_teacher
# echo "=== Phase 1: seq_kd, lmbda=0.0 (files 1-49) ==="
# run_phase "${PHASE1_OUTPUT}" \
#     env NPROC_PER_NODE=${STUDENT_NPROC} \
#     CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
#     PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
#     swift rlhf \
#         "${COMMON_ARGS[@]}" \
#         --model "${STUDENT_MODEL}" \
#         --dataset "${DATASET1[@]}" \
#         --seq_kd true \
#         --lmbda 0.0 \
#         --max_completion_length 4096 \
#         --per_device_train_batch_size 2 \
#         --gradient_accumulation_steps 21 \
#         --learning_rate 1e-4 \
#         --max_steps 24000 \
#         --output_dir "${PHASE1_OUTPUT}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: lmbda=0.0 – Mode 3 Offline KD (files 50-98)
#   Dataset text used as response; teacher provides logprob supervision.
#   Faster than Phase 1 (no teacher generation), broader data coverage.
#   bs=2, grad_accum=21 → 2×21×8192×6 ≈ 2.07M tokens/step  (total_length=4096+4096=8192)
# ══════════════════════════════════════════════════════════════════════════════
check_teacher
echo "=== Phase 2: offline KD, lmbda=0.0 (files 50-98) ==="
run_phase "${PHASE2_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        "${COMMON_ARGS[@]}" \
        --model "${STUDENT_MODEL}" \
        --dataset "${DATASET2[@]}" \
        --lmbda 0.0 \
        --seq_kd false \
        --max_completion_length 4096 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 21 \
        --learning_rate 1e-4 \
        --max_steps 24000 \
        --output_dir "${PHASE2_OUTPUT}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: lmbda=0.3 – Mixed on/off-policy
#   Student needs vLLM for on-policy generation (30% of samples).
#   vllm_mode=colocate + sleep_level=1: student vLLM shares GPUs with training.
#   bs=2, grad_accum=35 → 2×35×6144×6 ≈ 2.58M tokens/step  (total_length=4096+2048=6144)
# ══════════════════════════════════════════════════════════════════════════════
check_teacher
echo "=== Phase 3: lmbda=0.3 ==="
run_phase "${PHASE3_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        "${COMMON_ARGS[@]}" \
        --model "${PHASE2_OUTPUT}" \
        --dataset "${DATASET3[@]}" \
        --lmbda 0.3 \
        --max_completion_length 2048 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 35 \
        --learning_rate 5e-5 \
        --max_steps 13000 \
        --use_vllm true \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.3 \
        --vllm_tensor_parallel_size 1 \
        --vllm_max_model_len ${STUDENT_VLLM_MAX_MODEL_LEN} \
        --sleep_level 1 \
        --offload_model true \
        --offload_optimizer true \
        --output_dir "${PHASE3_OUTPUT}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: lmbda=1.0 – Full on-policy KD
#   Every sample uses student-generated output → vLLM critical for speed.
#   bs=2, grad_accum=35 → 2×35×6144×6 ≈ 2.58M tokens/step  (total_length=4096+2048=6144)
# ══════════════════════════════════════════════════════════════════════════════
check_teacher
echo "=== Phase 4: lmbda=1.0 ==="
run_phase "${PHASE4_OUTPUT}" \
    env NPROC_PER_NODE=${STUDENT_NPROC} \
    CUDA_VISIBLE_DEVICES=${STUDENT_GPUS} \
    PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
    swift rlhf \
        "${COMMON_ARGS[@]}" \
        --model "${PHASE3_OUTPUT}" \
        --dataset "${DATASET4[@]}" \
        --lmbda 1.0 \
        --max_completion_length 2048 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 35 \
        --learning_rate 2e-5 \
        --max_steps 7000 \
        --use_vllm true \
        --vllm_mode colocate \
        --vllm_gpu_memory_utilization 0.3 \
        --vllm_tensor_parallel_size 1 \
        --vllm_max_model_len ${STUDENT_VLLM_MAX_MODEL_LEN} \
        --sleep_level 1 \
        --offload_model true \
        --offload_optimizer true \
        --output_dir "${PHASE4_OUTPUT}"

stop_teacher
echo "=== Training complete. Final model: ${PHASE4_OUTPUT} ==="
