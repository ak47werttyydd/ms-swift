# Qwen3.5-LatentMoE-MLA: Custom HuggingFace Model & GKD Setup

## What Was Done

We created a custom HuggingFace model class for a hybrid architecture that
combines features not present in the standard `Qwen3_5MoeForConditionalGeneration`:

| Feature | Standard Qwen3.5 MoE | Our Custom Model |
|---------|----------------------|------------------|
| Full attention | GQA (q/k/v projections) | **MLA** (compressed QKV via low-rank bottleneck) |
| MoE routing | Standard dispatch at full hidden dim | **LatentMoE** (down-proj -> experts -> up-proj) |
| Linear attention | GatedDeltaNet | GatedDeltaNet (inherited) |
| MTP | Supported | Supported (inherited) |
| Vision | Qwen3.5 ViT | Qwen3.5 ViT (inherited) |

### Files Created

```
hf_custom_model/
├── __init__.py                                  # Package exports
├── config.json                                  # Model config with auto_map
├── configuration_qwen3_5_latentmoe_mla.py       # Config classes
└── modeling_qwen3_5_latentmoe_mla.py            # Model classes
```

### Architecture Details

**Model type**: `qwen3_5_latentmoe_mla`
**Architecture class**: `Qwen3_5LatentMoeMLAForConditionalGeneration`

#### MLA Attention (replaces GQA on full-attention layers)

```
hidden_states [b, s, 2048]
    |
    v  linear_qkv (no TP): 2048 -> 2112 (= q_lora:1536 + kv_lora:512 + rope:64)
    |
    +-- q_compressed -> q_norm -> linear_q_up_proj: 1536 -> 5120 (8 heads * 320 * 2 with gate)
    |     split -> q_nope [8, 256] + q_rope [8, 64] + gate [8, 256]
    |
    +-- kv_compressed -> kv_norm -> linear_kv_up_proj: 512 -> 4096 (8 heads * 512)
    |     split -> k_nope [8, 256] + value [8, 256]
    |
    +-- k_rope_input -> expand to 8 heads -> apply RoPE
    |
    v  concat(q_nope, q_rope) = query [8, 320]
    v  concat(k_nope, k_rope) = key   [8, 320]
    v  attention(query, key, value) -> output [8, 256]
    v  output * sigmoid(gate) -> o_proj: 2048 -> 2048
```

#### LatentMoE (wraps standard MoE experts)

```
hidden_states [b, s, 2048]
    |
    +-- router(hidden_states) -> top-8 of 128 experts (at full dim)
    |
    +-- down_proj: 2048 -> 1024 (factor=2)
    |     v  dispatch to experts (at 1024 dim)
    |     v  expert FFN: 1024 -> 512 -> 1024
    |     v  unpermute
    +-- up_proj: 1024 -> 2048
    |
    +-- shared_expert(hidden_states) * sigmoid(gate)
    |
    v  routed_output + shared_output
```

### Verification Results

| Test | Result |
|------|--------|
| Config loading via `AutoConfig` | Pass |
| Model instantiation (6.34B params) | Pass |
| MLA forward pass (CPU) | Pass |
| LatentMoE forward pass (CPU) | Pass |
| Full decoder layer forward (CPU) | Pass |
| Tiny model forward + backward (CUDA bf16) | Pass |

---

## How to Set Up GKD with ms-swift

### Prerequisites

```bash
# ms-swift installed at /home/a84400789/ms-swift
pip install ms-swift[rlhf]  # or already installed
pip install flash-linear-attention causal-conv1d  # for GatedDeltaNet
```

### Step 1: Prepare Model Files

Copy the custom model files alongside your converted checkpoint:

```bash
# Your checkpoint directory should look like:
my_student_model/
├── config.json                                  # from hf_custom_model/config.json
├── configuration_qwen3_5_latentmoe_mla.py       # custom config class
├── modeling_qwen3_5_latentmoe_mla.py            # custom modeling class
├── __init__.py                                  # package init
├── model-00001-of-XXXXX.safetensors             # converted weights
├── model-00002-of-XXXXX.safetensors
├── ...
├── model.safetensors.index.json
├── tokenizer.json                               # from Qwen3.5 tokenizer
├── tokenizer_config.json
└── special_tokens_map.json
```

The `auto_map` field in `config.json` ensures `AutoModelForCausalLM.from_pretrained()`
loads the correct custom classes:

```json
{
    "auto_map": {
        "AutoConfig": "configuration_qwen3_5_latentmoe_mla.Qwen3_5LatentMoeMLAConfig",
        "AutoModelForCausalLM": "modeling_qwen3_5_latentmoe_mla.Qwen3_5LatentMoeMLAForConditionalGeneration"
    }
}
```

### Step 2: Create ms-swift External Plugin

Create a file `gkd_plugin.py`:

```python
from swift.model import ModelMeta, ModelGroup, Model, register_model
from swift.template import TemplateType

register_model(
    ModelMeta(
        'qwen3_5_latentmoe_mla',
        [ModelGroup(
            [Model(path='./my_student_model')],
            TemplateType.qwen3_5,
        )],
        architectures=['Qwen3_5LatentMoeMLAForConditionalGeneration'],
        requires=['transformers>=4.57'],
        tags=['vision', 'video'],
    )
)
```

### Step 3: Prepare Dataset

GKD requires a dataset with a `response` field (or mapped via `--columns`).

**Option A**: Use a pre-existing text dataset:
```bash
# Example: FineWeb-Edu subset
DATASET='/path/to/fineweb-edu/sample.parquet#50000'
COLUMNS='{"text": "response"}'
```

**Option B**: Pre-sample teacher responses (recommended for large teachers):
```bash
# First, run teacher inference to generate responses
swift infer \
    --model Qwen/Qwen3.5-4B \
    --infer_backend vllm \
    --columns '{"text": "query"}' \
    --val_dataset /path/to/raw_data.parquet \
    --result_path presample_teacher.jsonl
```

### Step 4: Run GKD Training

#### Option A: Local Teacher (simpler, requires more GPU memory)

```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
swift rlhf \
    --rlhf_type gkd \
    --model ./my_student_model \
    --model_type qwen3_5_latentmoe_mla \
    --trust_remote_code true \
    --external_plugins gkd_plugin.py \
    --teacher_model Qwen/Qwen3.5-4B \
    --teacher_deepspeed zero3 \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --dataset '/path/to/data.parquet#50000' \
    --columns '{"text": "response"}' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_completion_length 512 \
    --beta 0.5 \
    --lmbda 0.0 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --save_only_model true \
    --output_dir output/gkd_latentmoe_mla \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
```

#### Option B: vLLM Teacher Server (recommended for large teachers)

**Terminal 1** - Start teacher vLLM server:
```bash
CUDA_VISIBLE_DEVICES=0,1 \
swift deploy \
    --model Qwen/Qwen3.5-4B \
    --infer_backend vllm \
    --vllm_tensor_parallel_size 2 \
    --served_model_name teacher \
    --port 8000
```

**Terminal 2** - Run GKD with server teacher:
```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
swift rlhf \
    --rlhf_type gkd \
    --model ./my_student_model \
    --model_type qwen3_5_latentmoe_mla \
    --trust_remote_code true \
    --external_plugins gkd_plugin.py \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --tuner_type full \
    --freeze_vit true \
    --freeze_aligner true \
    --freeze_llm false \
    --dataset '/path/to/data.parquet#50000' \
    --columns '{"text": "response"}' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --max_completion_length 512 \
    --beta 0.5 \
    --lmbda 0.0 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --save_only_model true \
    --output_dir output/gkd_latentmoe_mla \
    --deepspeed zero3 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
```

### Step 5: Key GKD Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beta` | 0.5 | JSD interpolation: 0=forward KL, 1=reverse KL, 0.5=symmetric |
| `--lmbda` | 0.0 | On-policy probability: 0=offline (dataset), 1=full on-policy (student generates) |
| `--temperature` | 1.0 | Temperature for teacher/student logits |
| `--seq_kd` | false | Teacher generates completions (requires vLLM) |
| `--sft_alpha` | 0.0 | Weight of SFT loss added to GKD loss |
| `--gkd_logits_topk` | None | Top-k logits for KL (required with server mode, saves bandwidth) |
| `--offload_teacher_model` | false | Offload teacher to CPU between forward passes |
| `--max_completion_length` | 512 | Max tokens for on-policy generation |

### Training Phases (Recommended)

**Phase 1 - Offline KD** (`lmbda=0.0`):
Train student on dataset responses with teacher logits supervision.

**Phase 2 - On-Policy KD** (`lmbda=0.3`):
Mix dataset responses with student-generated responses.
Requires `--use_vllm true --vllm_mode colocate`.

### Troubleshooting

1. **"trust_remote_code" error**: Add `--trust_remote_code true`
2. **OOM on teacher**: Use `--teacher_deepspeed zero3` or `--offload_teacher_model true`, or switch to vLLM server mode
3. **Slow convergence**: Increase `--sft_alpha 0.1` to add supervised loss
4. **Model not recognized**: Ensure `--external_plugins gkd_plugin.py` is set and the plugin file is accessible
5. **causal_conv1d error**: Install `pip install causal-conv1d>=1.2.0` (CUDA required)
