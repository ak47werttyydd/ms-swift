#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | FSDP training | NVIDIA GPUs

# ray stop --force
# ps -aux | grep "VLLM" | grep -v grep| awk '{print $2}' | xargs kill -9 | pkill -9 python

source /home/m00845822/cann-851/ascend-toolkit/set_env.sh
source /home/m00845822/cann-851/nnal/atb/set_env.sh

# set -xeuo pipefail
export HYDRA_FULL_ERROR=1
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1

export PYTHONPATH="/home/m00845822/MindSpeed-MM:/home/b00585163/MindSpeed-0304:$PYTHONPATH"

# Mindspeed-MM fsdp config
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export OMP_NUM_THREADS=1

# ---- NPU env ----------------------------------------------------------------
# original: VLLM_ATTENTION_BACKEND=ASCEND
export VLLM_ATTENTION_BACKEND=ASCEND
# original: VLLM_ASCEND_ENABLE_NZ=0
export VLLM_ASCEND_ENABLE_NZ=0


rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.65}
teacher_tp=${TEACHER_TP:-4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.85}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-50}
test_freq=${TEST_FREQ:-1}

project_name=${PROJECT_NAME:-verl_distill_gsm8k_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_from_qwen3_32b_vllm_fsdp}
# ---- end user-adjustable ----

gsm8k_train=./data/train.parquet
gsm8k_test=./data/test.parquet

train_files="['$gsm8k_train']"
val_files="['$gsm8k_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='right'
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.tokenizer_path="$STUDENT_MODEL"
    actor_rollout_ref.model.trust_remote_code=True

    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.plugin='[mindspeed_mm/fsdp/models/qwen3_5, mindspeed_mm/fsdp/data/datasets/huggingface]'
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.micro_batch_size=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.mindspeed.fsdp_kwargs.training.gradient_accumulation_steps=${GRAD_ACCU_STEPS}
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.model_id=qwen3_5
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.use_triton_gdn=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.model.freeze='[model.visual]'
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.fsdp_plan.apply_modules="['model.visual.blocks.{*}', \
    'model.visual', 'model.language_model.layers.{*}', 'model.language_model.embed_tokens', 'model.language_model', 'lm_head']"
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.recompute=True
    +actor_rollout_ref.actor.mindspeed.fsdp_kwargs.parallel.recompute_plan.apply_modules="['model.language_model.layers.{*}']"
    actor_rollout_ref.actor.mindspeed.ulysses_sequence_parallel_size=$sp_size
    actor_rollout_ref.ref.mindspeed.ulysses_sequence_parallel_size=$sp_size
    actor_rollout_ref.actor.mindspeed.param_offload=True
    actor_rollout_ref.actor.mindspeed.optimizer_offload=True
    actor_rollout_ref.actor.mindspeed.offload_policy=True
    actor_rollout_ref.ref.mindspeed.param_offload=True
    actor_rollout_ref.ref.mindspeed.offload_policy=True
    actor_rollout_ref.actor.optim.optimizer=adamw
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.total_training_steps=100
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distil
# profiling NPU options
SAVE_PATH="./profile_data/0518"
LEVEL="level0"
CONTENTS=['npu','cpu','stack']
ANALYSIS=True

PROFILER=(
    global_profiler.tool=npu
    global_profiler.steps=$PROFILE_STEPS
    global_profiler.save_path=$SAVE_PATH
    actor_rollout_ref.actor.profiler.enable=True
    actor_rollout_ref.actor.profiler.ranks=$PROFILE_RANKS
    actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL
    actor_rollout_ref.actor.profiler.tool_config.npu.discrete=$DISCRETE
    actor_rollout_ref.actor.profiler.tool_config.npu.contents=$CONTENTS
    actor_rollout_ref.actor.profiler.tool_config.npu.level=$LEVEL
    actor_rollout_ref.actor.profiler.tool_config.npu.analysis=$ANALYSIS
    actor_rollout_ref.ref.profiler.enable=True
    actor_rollout_ref.ref.profiler.ranks=$PROFILE_RANKS
    actor_rollout_ref.ref.profiler.all_ranks=$PROFILE_RANKS_ALL
    actor_rollout_ref.ref.profiler.tool_config.npu.discrete=$DISCRETE
    actor_rollout_ref.ref.profiler.tool_config.npu.contents=$CONTENTS
    actor_rollout_ref.ref.profiler.tool_config.npu.level=$LEVEL
    actor_rollout_ref.ref.profiler.tool_config.npu.analysis=$ANALYSIS
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name='ppo_trainer.yaml' \
    model_engine=mindspeed \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${MINDSPEED_CONFIG[@]}" \
    "$@"

# "${PROFILER[@]}" \
