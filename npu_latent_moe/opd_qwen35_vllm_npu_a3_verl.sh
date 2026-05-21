#!/usr/bin/env bash
# =============================================================================
# verl port of opd_qwen35_vllm_npu_a3.sh
#
# Full On-Policy Distillation on Atlas A3 (16x NPU):
#   Qwen3.5-35B-A3B (teacher, vllm-ascend)
#   -> Qwen3.5-0.8B (student, FSDP2 + colocated vllm-ascend rollout)
#
# Mapping notes:
#   - The original ms-swift script launches a *separate* `vllm serve` teacher
#     and points `--teacher_model_server` at it. verl owns the teacher
#     lifecycle: it spins up the teacher inference replicas itself inside a
#     dedicated Ray resource pool (distillation.{n_gpus_per_node,nnodes}).
#     There is therefore no start_teacher / stop_teacher / watchdog logic
#     here (the user explicitly asked to drop the watchdog).
#   - The ms-swift NPU layout (NPU 12-15 = teacher, NPU 0-11 = student) is
#     replaced by two Ray resource pools of sizes 4 + 12 that share the same
#     16-NPU node. Ray decides the physical NPU assignment; you cannot pin
#     "teacher to 12-15" explicitly, so all 16 NPUs must be visible.
#   - GKD with lmbda=1.0 + beta=1.0 (always sample from student, pure
#     forward-KL) maps onto verl's GKD OPD recipe:
#       distillation.distillation_loss.loss_mode=forward_kl_topk
#       distillation.distillation_loss.use_policy_gradient=False
#     See docs/algo/opd.md "GKD OPD" section.
#
# Dataset:
#   The raw HF gsm8k parquet (`question` + `answer` columns) is NOT what verl
#   reads. verl only reads `data.prompt_key` (default `"prompt"`), which must
#   be a chat-format list of {"role","content"} messages. You must first
#   preprocess the raw parquet, e.g. via verl's own helper:
#
#     python examples/data_preprocess/gsm8k.py \
#         --local_dataset_path path/to/raw/gsm8k \
#         --local_save_dir     path/to/gsm8k
#
#   That produces train.parquet/test.parquet with a `prompt` column derived
#   from `question`. The `answer` column is unused on the OPD path because
#   distillation.distillation_loss.use_task_rewards=False below: the student
#   generates its own response from the prompt, the teacher scores logprobs
#   on (prompt + student response), and the student is trained to match the
#   teacher distribution. The dataset's `answer` is never read.
# =============================================================================
set -xeuo pipefail

# ---- Paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# original: TEACHER_MODEL=${SCRIPT_DIR}/init_qwen3.5/qwen35_35B_A3B_init_ckpt
TEACHER_MODEL="${TEACHER_MODEL:-${SCRIPT_DIR}/init_qwen3.5/qwen35_35B_A3B_init_ckpt}"
# original: STUDENT_MODEL=${SCRIPT_DIR}/init_qwen3.5/qwen35_4B_init_ckpt
# user override: student is now Qwen3.5-0.8B.
STUDENT_MODEL="${STUDENT_MODEL:-path/to/qwen3.5-0.8B}"
# original: OUTPUT_DIR=${SCRIPT_DIR}/output/opd_qwen35_35B_to_4B
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/opd_qwen35_35B_to_0_8B}"
mkdir -p "${OUTPUT_DIR}"

# original: DATASET_PATH=.../alpaca/.../parquet
# Placeholder for the *raw* HF gsm8k parquet. Override DATASET_PATH at run
# time with the path of a *verl-preprocessed* parquet (see header comment).
# If you point this at the raw HF parquet, the run will crash inside
# tokenizer.apply_chat_template because `question` is a string, not a chat
# message list.
DATASET_PATH="${DATASET_PATH:-path/to/gsm8k/train-00000-of-00001.parquet}"
# gsm8k has a separate test split (test-00000-of-00001.parquet) on HF; the
# original script had no validation set, so we reuse the train path as a
# placeholder. Override DATASET_VAL_PATH if you actually want eval.
DATASET_VAL_PATH="${DATASET_VAL_PATH:-${DATASET_PATH}}"

# ---- Student training config -----------------------------------------------
# original: STUDENT_NPROC=12 (NPROC_PER_NODE for the student torchrun)
STUDENT_NPROC="${STUDENT_NPROC:-12}"
# original: MAX_LENGTH=4096       -> data.max_prompt_length
MAX_LENGTH="${MAX_LENGTH:-4096}"
# original: MAX_COMPLETION_LENGTH=1024 -> data.max_response_length
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
# original: STUDENT_VLLM_MAX_MODEL_LEN = MAX_LENGTH + MAX_COMPLETION_LENGTH
STUDENT_VLLM_MAX_MODEL_LEN=$(( MAX_LENGTH + MAX_COMPLETION_LENGTH + 1 ))
# original: MBS=4    -> per-device micro batch
MBS="${MBS:-4}"
# original: GRAD_ACC=15
GRAD_ACC="${GRAD_ACC:-15}"
# Effective batch = MBS * GRAD_ACC * STUDENT_NPROC = 4 * 15 * 12 = 720.
# verl computes grad-acc as ppo_mini_batch_size / (n_gpus * ppo_micro_batch_size_per_gpu).
# Setting train_batch_size == ppo_mini_batch_size = 720 yields one mini-batch per iter
# with 15 grad-acc micro-steps per update, matching the original semantics.
TRAIN_BATCH_SIZE=$(( MBS * GRAD_ACC * STUDENT_NPROC ))
PPO_MINI_BATCH_SIZE=${TRAIN_BATCH_SIZE}
# original: SAVE_STEPS=50 -> trainer.save_freq
SAVE_STEPS="${SAVE_STEPS:-50}"

# ---- Teacher (vLLM) pool config --------------------------------------------
# original: TEACHER_TP=4
TEACHER_TP="${TEACHER_TP:-4}"
# original: TEACHER_NPUS=12,13,14,15 (4 NPUs)
# In verl: dedicated teacher pool sized n_gpus_per_node * nnodes = 4 * 1.
TEACHER_NGPUS_PER_NODE="${TEACHER_NGPUS_PER_NODE:-4}"
TEACHER_NNODES="${TEACHER_NNODES:-1}"
# original: TEACHER_MAX_LOGPROBS=64 -> distillation.distillation_loss.topk
TEACHER_MAX_LOGPROBS="${TEACHER_MAX_LOGPROBS:-64}"
# original: TEACHER_MAX_MODEL_LEN=5200 -> teacher inference.max_model_len
TEACHER_MAX_MODEL_LEN="${TEACHER_MAX_MODEL_LEN:-5200}"
# original: TEACHER_NPU_MEM_UTIL=0.85 -> teacher inference.gpu_memory_utilization
TEACHER_NPU_MEM_UTIL="${TEACHER_NPU_MEM_UTIL:-0.85}"

# ---- NPU env ----------------------------------------------------------------
# original: VLLM_ATTENTION_BACKEND=ASCEND
export VLLM_ATTENTION_BACKEND=ASCEND
# original: VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_ENABLE_NZ=0
# verl-specific: tell Ray not to clobber ASCEND_RT_VISIBLE_DEVICES so worker
# processes inherit the right device list. No direct ms-swift counterpart.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
# original split TEACHER_NPUS / STUDENT_NPUS across two processes; here Ray
# assigns the two pools out of the full 16-NPU set, so expose all of them.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
# verl-specific debugging aid; no direct ms-swift counterpart.
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1

# ---- Sanity checks ---------------------------------------------------------
# original block 1:1 with this one. NOTE: these will trip on the placeholder
# `path/to/...` strings above; override TEACHER_MODEL/STUDENT_MODEL/DATASET_PATH
# in your environment before invoking the script.
command -v npu-smi >/dev/null 2>&1 || { echo "ERROR: npu-smi not found - CANN not sourced?" >&2; exit 1; }
[[ -n "${ASCEND_TOOLKIT_HOME:-}" ]] || echo "WARN: ASCEND_TOOLKIT_HOME unset - did you source set_env.sh?"
[[ -d "${TEACHER_MODEL}" ]] || { echo "ERROR: teacher ckpt not found: ${TEACHER_MODEL}" >&2; exit 1; }
[[ -d "${STUDENT_MODEL}" ]] || { echo "ERROR: student ckpt not found: ${STUDENT_MODEL}" >&2; exit 1; }
[[ -f "${DATASET_PATH}"  ]] || { echo "ERROR: dataset not found: ${DATASET_PATH}" >&2; exit 1; }

# ============================================================================
# verl parameter arrays
# Each block annotates its ms-swift counterpart (or notes the lack of one).
# ============================================================================

DATA=(
    # original: algorithm not exposed - GKD has no task reward.
    # verl requires *some* adv_estimator even when use_task_rewards=False; grpo
    # is the convention used by the upstream OPD examples.
    algorithm.adv_estimator=grpo
    # original: no ref-policy KL term (--beta_kl absent). Disable both verl
    # KL knobs so the only training signal is the teacher-vs-student KL.
    algorithm.use_kl_in_reward=False

    # original: --dataset "${DATASET_PATH}" (single parquet, train only)
    data.train_files="['${DATASET_PATH}']"
    data.val_files="['${DATASET_VAL_PATH}']"

    # original: --columns '{"instruction":"query","output":"response"}'
    # No direct verl equivalent. verl always reads `data.prompt_key` (default
    # `prompt`) as a chat-format list of message dicts. gsm8k's raw `question`
    # column is a plain string, so it must be preprocessed into `prompt` first.
    # `answer` is unused on this code path (use_task_rewards=False below).
    data.prompt_key=prompt

    # original: derived from MBS * GRAD_ACC * STUDENT_NPROC
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    # original: --max_length 4096
    data.max_prompt_length=${MAX_LENGTH}
    # original: --max_completion_length 1024
    data.max_response_length=${MAX_COMPLETION_LENGTH}
    # original: --truncation_strategy right
    data.truncation='right'
    # original: implicit (ms-swift defaults to shuffling the train split)
    data.shuffle=True
    # original: no ms-swift equivalent. Skip overlong prompts so they don't crash
    # the rollout when verl's vLLM hits its max_model_len cap.
    data.filter_overlong_prompts=True
)

MODEL=(
    # original: --model "${STUDENT_MODEL}"
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    # original: no ms-swift counterpart. verl-specific perf flag (removes pad
    # tokens before the forward pass); safe default for OPD.
    actor_rollout_ref.model.use_remove_padding=True
    # original: --gradient_checkpointing true
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # original: --torch_dtype bfloat16
    actor_rollout_ref.model.dtype=bfloat16
    # original: --model_type qwen3_5  -> verl autodetects from HF config, no
    # explicit knob needed.
)

ACTOR=(
    # original: --deepspeed zero3 - FSDP2 is the closest verl-supported analog
    # (full param + grad + optimizer sharding). DeepSpeed ZeRO-3 itself is not
    # supported as a verl training strategy.
    actor_rollout_ref.actor.strategy=fsdp2
    # original: --learning_rate 1e-5
    actor_rollout_ref.actor.optim.lr=1e-5
    # original: --warmup_ratio 0.05
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05
    actor_rollout_ref.actor.optim.warmup_style=cosine

    # original: PPO_MINI_BATCH_SIZE == TRAIN_BATCH_SIZE so each rollout batch
    # is consumed in a single PPO mini-batch (one optimizer step per iter),
    # matching the plain SFT-like loop ms-swift uses for GKD.
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    # original: --per_device_train_batch_size 4
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MBS}
    # original: --gradient_accumulation_steps 15 - in verl this is derived
    # implicitly as ppo_mini_batch_size / (n_gpus * ppo_micro_batch_size_per_gpu)
    # = 720 / (12 * 4) = 15. No direct knob.

    # original: no ms-swift counterpart - GKD has no ref-policy KL, so turn
    # off verl's KL term too (otherwise the student is regularized toward both
    # the ref-policy AND the teacher, mixing two signals).
    actor_rollout_ref.actor.use_kl_loss=False

    # original: --offload_model true
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    # original: --offload_optimizer true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

    # original: --attn_impl sdpa - on NPU, FSDP forward uses HF attn impl;
    # set sdpa explicitly to match the original (flash_attn unavailable on NPU).
    actor_rollout_ref.model.attn_implementation=sdpa

    # original: --dataloader_num_workers 4 / --dataset_num_proc 8 / --load_from_cache_file true
    # No direct verl equivalents: verl loads pre-processed parquets via Ray and
    # does not expose HF DataLoader / datasets caching knobs.

    # original: --tuner_type full (full-parameter SFT) / --freeze_llm false /
    # --freeze_vit true / --freeze_aligner true
    # No verl knob needed: verl always full-trains the actor (no built-in
    # LoRA gating on this path); the Qwen3.5-0.8B student has no vision tower
    # to freeze.

    # original: --enable_thinking false - controlled via chat-template kwargs
    # in tokenizer apply; no verl-side equivalent flag. Bake the right
    # template into the preprocessed parquet if needed.

    # original: --logging_steps 10 / --save_total_limit 5 - logging cadence and
    # ckpt rotation are partially exposed via trainer.* below; verl logs every
    # step and has no `logging_steps` equivalent.
)

ROLLOUT=(
    # original: --use_vllm true
    actor_rollout_ref.rollout.name=vllm
    # original: --vllm_tensor_parallel_size 1
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    # original: --vllm_gpu_memory_utilization 0.65
    actor_rollout_ref.rollout.gpu_memory_utilization=0.65
    # original: --vllm_max_model_len 5120 (= MAX_LENGTH + MAX_COMPLETION_LENGTH)
    actor_rollout_ref.rollout.max_model_len=${STUDENT_VLLM_MAX_MODEL_LEN}
    # original: --vllm_enforce_eager true
    actor_rollout_ref.rollout.enforce_eager=True
    # original: --temperature 1.0
    actor_rollout_ref.rollout.temperature=1.0
    # original: implicit - GKD samples one completion per prompt
    actor_rollout_ref.rollout.n=1
    # original: --vllm_mode colocate - verl always colocates rollout with actor
    # in the same pool, so this is the implicit (and only) layout. No knob.
    # original: --sleep_level 2 - verl manages the engine sleep/wake cycle via
    # `free_cache_engine`; sleep_level is not user-tunable.
    actor_rollout_ref.rollout.free_cache_engine=True
    # original: --move_model_batches 24 - verl uses its checkpoint engine for
    # train->infer weight syncing; bucket size is tuned separately and not a
    # 1:1 mapping of move_model_batches.
)

TRAINER=(
    # original: implicit - one wandb-style logger; mirror with verl's logger list.
    trainer.logger='["console"]'
    trainer.project_name='opd_qwen35'
    trainer.experiment_name='qwen35_35b_to_0_8b_gkd_gsm8k_vllm_npu_a3'
    # original: STUDENT_NPROC=12 (--nproc_per_node) -> trainer pool size
    trainer.n_gpus_per_node=${STUDENT_NPROC}
    trainer.nnodes=1
    # original: no ms-swift counterpart - verl-specific flag to balance the
    # sequence load across DP ranks for FSDP. Safe default.
    trainer.balance_batch=True
    # original: no eval split / no val_before_train option in ms-swift GKD.
    trainer.val_before_train=False
    trainer.test_freq=-1
    # original: --save_steps 50
    trainer.save_freq=${SAVE_STEPS}
    # original: --save_total_limit 5
    trainer.max_actor_ckpt_to_keep=5
    # original: --max_steps 10000
    trainer.total_training_steps=10000
    # original: --output_dir "${OUTPUT_DIR}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    # original: --save_only_model true
    actor_rollout_ref.actor.checkpoint.save_contents='[model]'
)

# Single teacher; the OPD doc warns that the default key `teacher_model` is
# silently popped when extra named teachers are added, so keep this one name.
EXTRA=(
    # original: --rlhf_type gkd -> verl OPD enabled
    distillation.enabled=True
    # original: TEACHER_NPUS=12,13,14,15 (4 NPUs on 1 node)
    distillation.n_gpus_per_node=${TEACHER_NGPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}

    # original: TEACHER_MODEL path
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    # original: --tensor-parallel-size ${TEACHER_TP} (vllm serve flag)
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    # original: vllm serve (vllm-ascend on NPU)
    distillation.teacher_models.teacher_model.inference.name=vllm
    # original: --gpu-memory-utilization ${TEACHER_NPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_NPU_MEM_UTIL}
    # original: --max-model-len ${TEACHER_MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.max_model_len=${TEACHER_MAX_MODEL_LEN}
    # original: --enforce-eager
    distillation.teacher_models.teacher_model.inference.enforce_eager=True
    # original: --max-logprobs ${TEACHER_MAX_LOGPROBS} - verl auto-bumps this
    # to >= topk; pin it for clarity.
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs=${TEACHER_MAX_LOGPROBS}
    # original: --dtype bfloat16 (vllm serve flag)
    distillation.teacher_models.teacher_model.inference.dtype=bfloat16
    # original: --trust-remote-code - no separate verl knob; not needed for
    # qwen3_5 (shipped in transformers / vllm).
    # original: --port 8000 - not applicable, verl talks to the teacher
    # directly over Ray, no HTTP server.

    # original: --gkd_logits_topk 64
    distillation.distillation_loss.topk=${TEACHER_MAX_LOGPROBS}
    # original: --rlhf_type gkd + --lmbda 1.0 + --beta 1.0
    # -> verl's GKD OPD recipe: forward-KL on teacher top-k logits, no PG.
    # (lmbda=1 maps to "always sample from student" = the OPD default;
    #  beta=1 maps to "pure forward KL with the teacher" = forward_kl_topk
    #  with use_policy_gradient=False, see docs/algo/opd.md "GKD OPD".)
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.use_policy_gradient=False
    # original: --seq_kd false - OPD is on-policy by construction; this matches.
    # original: no --use_task_rewards equivalent; GKD has no task reward term.
    # Keep this False so gsm8k's `answer` column is genuinely ignored - the
    # only training signal is teacher-vs-student forward KL on tokens the
    # student itself sampled.
    distillation.distillation_loss.use_task_rewards=False
    # original: no ms-swift counterpart. Stability guards used by every upstream
    # OPD example; keep them on.
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

############################ launch ##########################################
# Original script had: start_teacher; check_teacher; run_training; stop_teacher
# verl collapses all of that into a single main_ppo invocation (the teacher
# pool is materialized by main_ppo when distillation.enabled=True).
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"

echo "=== Done. Final model: ${OUTPUT_DIR} ==="
