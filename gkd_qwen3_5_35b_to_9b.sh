# GKD: Qwen3.5-35B-A3B (pruned to 9B-A2B, student) <- Qwen3.5-4B (teacher)
#
# Step 2 – run this script:
#   bash gkd_qwen3_5_35b_to_9b.sh |& tee gkd_qwen3_5_35b_to_9b.log
#
# Three-phase schedule:
#   Phase 1 (lmbda=0.0, ~70B tokens): stable recovery from pruning via off-policy KD
#   Phase 2 (lmbda=0.3, ~20B tokens): gradually introduce on-policy signal
#   Phase 3 (lmbda=1.0, ~10B tokens): full on-policy KD to reduce exposure bias
#
# Token budget estimate (8 GPUs, batch=4, grad_accum=16, avg ~4k tokens/sample):
#   ~2M tokens/step → Phase1: 70k steps, Phase2: 20k steps, Phase3: 10k steps

STUDENT_MODEL=/home/r00914194/PruneMe/qwen35_exp80_layer_drop_082331
# TEACHER_MODEL=Qwen/Qwen3.5-35B-A3B
TEACHER_MODEL=Qwen/Qwen3.5-4B

PHASE1_OUTPUT=output/gkd_9b_phase1
PHASE2_OUTPUT=output/gkd_9b_phase2
PHASE3_OUTPUT=output/gkd_9b_phase3

# All 140 files split across phases: 70%(98) / 20%(28) / 10%(14)
# Phase 1 further split into 1a(49) + 1b(49) to avoid filling disk with Arrow cache
# Use bash arrays — each file becomes a separate argument to --dataset
mapfile -t _ALL < <(python3 -c "
import glob
for f in sorted(glob.glob('/dev/shm/dataset/fineweb-edu-100BT/sample/100BT/*.parquet')):
    print(f)
")
_N=${#_ALL[@]}
_N1=$(python3 -c "print(int($_N * 0.70))")   # 98
_N2=$(python3 -c "print(int($_N * 0.20))")   # 28
DATASET1A=("${_ALL[@]:0:49}")
DATASET1B=("${_ALL[@]:49:49}")
DATASET2=("${_ALL[@]:$_N1:$_N2}")
DATASET3=("${_ALL[@]:$((_N1+_N2))}")

# ARROW_CACHE=/dev/shm/ms_cache/datasets/parquet

COMMON_ARGS=(
    --rlhf_type gkd
    --model_type qwen3_5_moe
    --teacher_model "$TEACHER_MODEL"
    --teacher_deepspeed zero2
    --tuner_type full
    --columns '{"text":"response"}'
    --freeze_vit true
    --freeze_aligner true
    --freeze_llm false
    --torch_dtype bfloat16
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 64
    --warmup_ratio 0.05
    --max_length 4096
    --save_steps 2000
    --save_total_limit 2
    --save_only_model true
    --deepspeed zero2
    --attn_impl flash_attn
    --dataloader_num_workers 4
    --dataset_num_proc 8
    --streaming true
    --gradient_checkpointing true
)
# --logging_steps 10
#--offload_teacher_model true

# ── Phase 1a: lmbda=0.0 – files 1-49 ────────────────────────────────────────
echo "=== Phase 1a: lmbda=0.0 (files 1-49) ==="
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    "${COMMON_ARGS[@]}" \
    --model "$STUDENT_MODEL" \
    --dataset "${DATASET1A[@]}" \
    --lmbda 0.0 \
    --learning_rate 1e-4 \
    --max_steps 35000 \
    --output_dir "$PHASE1_OUTPUT"

# echo "=== Clearing Arrow cache after Phase 1a ==="
# rm -rf "$ARROW_CACHE"/default-*

# ── Phase 1b: lmbda=0.0 – files 50-98 ───────────────────────────────────────
echo "=== Phase 1b: lmbda=0.0 (files 50-98) ==="
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    "${COMMON_ARGS[@]}" \
    --model "$PHASE1_OUTPUT" \
    --dataset "${DATASET1B[@]}" \
    --lmbda 0.0 \
    --learning_rate 1e-4 \
    --max_steps 35000 \
    --output_dir "$PHASE1_OUTPUT"

# echo "=== Clearing Arrow cache after Phase 1b ==="
# rm -rf "$ARROW_CACHE"/default-*

# # ── Phase 2: lmbda=0.3 – mixed on/off-policy ─────────────────────────────────
# echo "=== Phase 2: lmbda=0.3 ==="
# NPROC_PER_NODE=8 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# swift rlhf \
#     "${COMMON_ARGS[@]}" \
#     --model "$PHASE1_OUTPUT" \
#     --dataset "${DATASET2[@]}" \
#     --lmbda 0.3 \
#     --max_completion_length 1024 \
#     --learning_rate 5e-5 \
#     --max_steps 20000 \
#     --output_dir "$PHASE2_OUTPUT"
# echo "=== Clearing Arrow cache after Phase 2 ==="
# rm -rf "$ARROW_CACHE"/default-*

# # ── Phase 3: lmbda=1.0 – full on-policy KD, reduce exposure bias ─────────────
# echo "=== Phase 3: lmbda=1.0 ==="
# NPROC_PER_NODE=8 \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
# swift rlhf \
#     "${COMMON_ARGS[@]}" \
#     --model "$PHASE2_OUTPUT" \
#     --dataset "${DATASET3[@]}" \
#     --lmbda 1.0 \
#     --max_completion_length 1024 \
#     --learning_rate 2e-5 \
#     --max_steps 10000 \
#     --output_dir "$PHASE3_OUTPUT"
# echo "=== Clearing Arrow cache after Phase 3 ==="
# rm -rf "$ARROW_CACHE"/default-*

# echo "=== Training complete. Final model: $PHASE3_OUTPUT ==="
