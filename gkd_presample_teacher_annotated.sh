#!/usr/bin/env bash
# ============================================================================
# GKD Solution 2 – Step 1: Teacher Model Pre-sampling  (annotated version)
#
# WHAT THIS SCRIPT DOES
# ---------------------
# In the standard GKD seq_kd flow (Phase 1 of gkd_qwen35_vllm.sh), the
# teacher model generates completions *live* during each training step, which
# is the main bottleneck.  Solution 2 ("Teacher Model Pre-sampling") breaks
# this into two independent jobs:
#
#   [this script]  Teacher runs swift infer offline → saves JSONL
#   [next script]  Student trains on that JSONL      → no live teacher gen
#
# The student still needs the teacher for KD logprob supervision during
# training, but getting logprobs is much cheaper than generating text, so the
# teacher server on 4 GPUs handles it easily in parallel with student training.
#
# RELATIONSHIP TO gkd_qwen35_vllm.sh Phase 1
# -------------------------------------------
#   Phase 1 uses: seq_kd=true, lmbda=0.0, files 1-49 (49 parquet files)
#   This script:  same teacher + same 49 files, but generates completions once
#                 offline so gkd_student_presample.sh can use seq_kd=false.
#
# GPU LAYOUT  (4 GPUs total for this script)
# ------------------------------------------
#   GPU 0-3  → vLLM tensor-parallel inference (TP=4)
#
#   Qwen3.5-35B-A3B memory budget (BF16, TP=4):
#     Weights:   35B × 2B / 4 ≈ 17.5 GB/GPU
#     KV cache:  (1.0 - 0.17) × 80GB  ≈ 50 GB/GPU  (util=0.90 leaves ~8GB)
#     → Well within 80 GB H100 limits; util=0.90 is safe here.
#
# OUTPUT
#   output/teacher_presample.jsonl
#   Each line: {"query": "...", "response": "..."}
#   The "response" is the teacher's generated completion for that document.
#   gkd_student_presample.sh passes this file directly as --dataset.
#
# USAGE
#   bash gkd_presample_teacher_annotated.sh |& tee gkd_presample_teacher.log
#   (Run BEFORE gkd_student_presample.sh)
# ============================================================================

# Exit immediately on error, treat unset variables as errors, propagate pipe
# failures.  This prevents silent failures mid-run from corrupting the output.
set -euo pipefail

# ── Model & output paths ──────────────────────────────────────────────────────

# Teacher model: same as gkd_qwen35_vllm.sh.  Can be a ModelScope/HuggingFace
# ID or a local path.  The model is downloaded to MODELSCOPE_CACHE /
# HF_HOME on first use.
TEACHER_MODEL=Qwen/Qwen3.5-35B-A3B

# Where to write the pre-sampled completions.  One JSON object per line.
# gkd_student_presample.sh reads this path via --dataset.
RESULT_PATH=output/teacher_presample.jsonl

# ── GPU / parallelism config ──────────────────────────────────────────────────

# Use all 4 GPUs assigned to the teacher.  Adjust if you want to leave GPUs
# free for other work running concurrently.
TEACHER_GPUS="0,1,2,3"

# Tensor-parallel degree: must equal the number of GPUs in TEACHER_GPUS.
# TP=4 shards each weight matrix across 4 GPUs, reducing per-GPU memory from
# ~70 GB (TP=1) to ~17.5 GB, leaving ample room for KV cache.
TEACHER_TP=4

# ── Dataset: Phase 1 slice (first 49 parquet files) ──────────────────────────

# fineweb-edu-100BT lives in RAM (/dev/shm) to eliminate storage I/O overhead
# when iterating over 49 × ~1 GB parquet files.
DATA_ROOT="/dev/shm/dataset/fineweb-edu-100BT/sample/100BT"

# Collect all parquet files in sorted order so the slice is deterministic
# across runs.  Uses Python glob to avoid shell glob ordering surprises.
mapfile -t _ALL < <(python3 -c "
import glob
for f in sorted(glob.glob('${DATA_ROOT}/*.parquet')):
    print(f)
")

# Take the first 49 files — identical slice to Phase 1 of gkd_qwen35_vllm.sh
# so the pre-sampled data covers exactly the same documents.
DATASET1=("${_ALL[@]:0:49}")

# ── Pre-flight checks & setup ─────────────────────────────────────────────────
echo "=== Teacher pre-sampling ==="
echo "    Teacher model : ${TEACHER_MODEL}"
echo "    GPUs          : ${TEACHER_GPUS}  (TP=${TEACHER_TP})"
echo "    Dataset files : ${#DATASET1[@]}"
echo "    Output        : ${RESULT_PATH}"

# Create the output directory if it doesn't exist yet.
mkdir -p "$(dirname "${RESULT_PATH}")"

# ── swift infer ───────────────────────────────────────────────────────────────
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lets PyTorch's allocator
# grow CUDA segments on demand, which reduces OOM risk during the initial model
# load and KV cache allocation without wasting reserved headroom.
CUDA_VISIBLE_DEVICES=${TEACHER_GPUS} \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift infer \
    \
    # ── Model ──────────────────────────────────────────────────────────────
    --model "${TEACHER_MODEL}" \
    \
    # Explicit model_type avoids auto-detection overhead and ensures the
    # correct MoE architecture class is loaded for Qwen3.5-35B-A3B.
    --model_type qwen3_5_moe \
    \
    # Use vLLM as the inference backend.  vLLM's continuous batching and
    # PagedAttention give far higher throughput than the Transformers backend
    # for bulk generation over thousands of samples.
    --infer_backend vllm \
    \
    # Must match TEACHER_TP (number of GPUs visible to this process).
    --vllm_tensor_parallel_size ${TEACHER_TP} \
    \
    # Reserve 90 % of GPU VRAM for model weights + KV cache.
    # At TP=4 with ~17.5 GB weights/GPU and 80 GB H100s, this gives
    # ~(80×0.9 - 17.5) = 54.5 GB/GPU for KV cache → very generous.
    --vllm_gpu_memory_utilization 0.90 \
    \
    # Maximum context length (prompt + generated tokens) vLLM will allocate
    # KV blocks for.  8192 matches Phase 1's max_length(4096) +
    # max_completion_length(4096).  Longer sequences are silently truncated
    # by vLLM if they exceed this limit.
    --vllm_max_model_len 8192 \
    \
    # ── Dataset ────────────────────────────────────────────────────────────
    --dataset "${DATASET1[@]}" \
    \
    # Column mapping: the fineweb-edu parquet files have a "text" column
    # containing raw web document text.  We map it to "query" so swift infer
    # treats each document as the prompt for the teacher.
    #
    # Why "query" and not "response"?
    #   In training (seq_kd=true), the "text" column feeds the model's input
    #   context and the teacher generates a *new* completion for it.
    #   Here we replicate that: "text" becomes the query the teacher sees,
    #   and swift infer writes the teacher's output as the "response" field
    #   in the output JSONL.
    #
    # If your data has a separate prompt field, change to e.g.:
    #   '{"prompt":"query"}' or '{"prompt":"query","text":"response"}'
    --columns '{"text":"query"}' \
    \
    # ── Generation settings ────────────────────────────────────────────────
    # Generate up to 4096 tokens per sample — same as Phase 1's
    # max_completion_length, so the student training lengths are consistent.
    --max_new_tokens 4096 \
    \
    # T=1.0 preserves the teacher's natural output distribution.
    # Lowering temperature sharpens the distribution (less diverse outputs);
    # raising it softens it.  T=1.0 is the GKD paper default and matches
    # Phase 1 of gkd_qwen35_vllm.sh.
    --temperature 1.0 \
    \
    # BF16: same dtype as training.  Using FP16 here while training in BF16
    # can introduce minor numerical mismatches in logprob supervision.
    --torch_dtype bfloat16 \
    \
    # Disable Qwen3's built-in chain-of-thought thinking tokens so the
    # teacher outputs plain responses (matches Phase 1 setting).
    --enable_thinking false \
    \
    # ── Output settings ────────────────────────────────────────────────────
    # Flush results to disk every 1000 samples.  This makes the output file
    # readable before inference finishes and limits data loss if the job is
    # interrupted (resume by filtering already-processed queries).
    --write_batch_size 1000 \
    \
    # Path for the output JSONL.  Each line is a JSON object:
    #   {"query": "<document text>", "response": "<teacher completion>"}
    # gkd_student_presample.sh passes this file directly as --dataset.
    --result_path "${RESULT_PATH}"

echo ""
echo "=== Pre-sampling complete ==="
echo "    Output : ${RESULT_PATH}"
echo "    Lines  : $(wc -l < "${RESULT_PATH}")"
echo ""
echo "Next step: bash gkd_student_presample.sh"
