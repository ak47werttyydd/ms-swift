# GKD (Generalized Knowledge Distillation): Qwen3.5-0.8B-A0.1B (student) <- Qwen3.5-0.8B (teacher)
# Student: custom MoE model, ~0.8B total / ~0.1B active, randomly initialised
# Teacher: pretrained Qwen3.5-0.8B
# GPU 2,3 (NPROC_PER_NODE=2), ZeRO-3, full tuner, only LM backbone trained
#
# Step 1 – create the student model (run once):
#   python create_student_model.py --base_model Qwen/Qwen3.5-0.8B \
#                                  --output_dir ./Qwen3_5-0.8B-A0.1B-student
# Step 2 – run this script:
#   bash gkd_qwen3_5_0.8b.sh |& tee gkd_qwen3_5_0.8b.log 

STUDENT_MODEL=./Qwen3_5-0.8B-A0.1B-student

NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type gkd \
    --model "$STUDENT_MODEL" \
    --model_type qwen3_5_moe \
    --teacher_model Qwen/Qwen3.5-0.8B \
    --teacher_deepspeed zero3 \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --dataset '/dev/shm/dataset/fineweb-edu-100BT/sample/100BT/000_00000.parquet#50000' \
    --columns '{"text": "response"}' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_completion_length 512 \
    --lmbda 0.0 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --save_only_model true \
    --output_dir output/gkd_qwen3_5_0.8b_a0.1b \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
