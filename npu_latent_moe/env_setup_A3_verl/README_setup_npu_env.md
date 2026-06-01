# setup_npu_env.sh

在 Ascend NPU 主机上为 Qwen3.5 GKD 蒸馏工作流搭建 conda 环境。

Teacher（vllm-ascend）和 Student（ms-swift）运行在同一个 `ms-swift` conda env 内，分属两个进程/Shell。

---

## 硬件与系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | openEuler（linux-aarch64） |
| CPU 架构 | aarch64 |
| NPU | Ascend 910C / `ascend910_9391`（Atlas 900 A3 SuperPoD，PCI 0xD803，16 NPU/节点） |
| NPU 显存 | 910C 每 die 64 GB；Qwen3.5-35B-A3B teacher 需 TP4 |
| CANN | **8.5.1**（必须，不能用 8.5.0 或 8.5.2） |
| conda | 已安装，在 PATH 中可用（脚本不自行安装 conda） |

---

## 版本矩阵

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11 | |
| CANN | 8.5.1 | 由 CANN `.run` 包预先安装 |
| torch / torch-npu | 2.9.0 / 2.9.0 | vllm-ascend v0.18.0 硬依赖；覆盖 ms-swift 文档的 2.7.1 pin |
| vllm | v0.18.0 | source install，`VLLM_TARGET_DEVICE=empty`（跳过 CUDA 编译） |
| vllm-ascend | releases/v0.18.0 | 包含 Qwen3.5 MoE flashcomm v1 + MTP shared-expert fix (#8004) |
| ms-swift | 本地源码 | 4.2.0.dev0 或更新 |
| transformers | >=4.57.4, <5.6 | |
| deepspeed | <0.19 | student 训练用 ZeRO-3 |

---

## 前置条件

1. CANN 8.5.1 已通过三个 `.run` 包安装（toolkit + 910b-ops + nnal），并能 source `set_env.sh`：
   ```bash
   /home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh
   /home/canada_group_account/a84400789/CANN8.5.1/nnal/atb/set_env.sh   # 可选，libatb.so
   ```
2. `npu-smi` 在 PATH 中可用。
3. 以下本地 git 仓库已 clone（路径可通过环境变量覆盖）：
   - `~/Code/vllm-ascend`（本仓库，需在 `releases/v0.18.0` 分支）
   - `~/Code/ms-swift`
   - `~/Code/vllm`（脚本会自动 clone，如已有则复用）

---

## 用法

```bash
# 默认配置
bash scripts/setup_npu_env.sh

# 自定义路径或开启国内镜像
VLLM_ASCEND_DIR=/data/vllm-ascend \
MS_SWIFT_DIR=/data/ms-swift \
USE_CN_MIRROR=1 \
bash scripts/setup_npu_env.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENV_NAME` | `ms-swift` | conda 环境名 |
| `PY_VERSION` | `3.11` | Python 版本 |
| `CANN_SETENV` | `/home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh` | CANN 环境脚本路径 |
| `NNAL_SETENV` | `/home/canada_group_account/a84400789/CANN8.5.1/nnal/atb/set_env.sh` | NNAL 环境脚本路径（可选） |
| `VLLM_ASCEND_DIR` | `~/Code/vllm-ascend` | vllm-ascend 本地仓库路径 |
| `MS_SWIFT_DIR` | `~/Code/ms-swift` | ms-swift 本地仓库路径 |
| `VLLM_DIR` | `~/Code/vllm` | vllm 本地仓库路径 |
| `VLLM_ASCEND_URL` | `https://github.com/ak47werttyydd/vllm-ascend` | vllm-ascend fork（含 mem_get_info 等 ms-swift 补丁） |
| `VLLM_ASCEND_BRANCH` | `v0.18.0_ms_swift` | vllm-ascend 分支（基于 releases/v0.18.0） |
| `VLLM_TAG` | `v0.18.0` | vllm tag |
| `USE_CN_MIRROR` | `0` | 设为 `1` 切换 pip 到阿里云镜像 |

---

## 安装完成后的使用方式

在两个 Shell 中分别运行 teacher 和 student：

**Shell 1 — Teacher 推理服务（Qwen3.5-35B-A3B，TP4）**

```bash
source /home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh
conda activate ms-swift
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 4 \
    --port 8000 \
    --max-logprobs 64 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code
```

**Shell 2 — Student GKD 训练（Qwen3.5-4B，DeepSpeed ZeRO-3）**

```bash
source /home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh
conda activate ms-swift
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-4B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#2000' 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --split_dataset_ratio 0.01 \
    --lmbda 0.5 --seq_kd false --beta 0.5 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --max_length 2048 --max_completion_length 512 \
    --deepspeed zero3 \
    --attn_impl sdpa \
    --output_dir output/qwen3_5-gkd \
    --save_steps 100 --save_total_limit 2 \
    --logging_steps 5 \
    --report_to tensorboard
```

---

## 注意事项

- **torch 安装**：torch-npu 从华为镜像安装，会自动拉取 aarch64 兼容的 torch 2.9.0。不从 pytorch.org 安装（那里的 wheel 是 x86_64）。
- **flash-attn 不安装**：flash-attn 仅支持 x86 CUDA。NPU 上用 `--attn_impl sdpa`，由 torch-npu 的 `npu_fusion_attention` 承接。
- **QLoRA / 量化不支持**：ms-swift NPU 文档明确标注，不安装 auto-gptq / bitsandbytes。
- **vllm 编译**：`VLLM_TARGET_DEVICE=empty` 跳过所有 CUDA kernel 编译，aarch64 上无 x86 专属扩展依赖，但纯 C++ 部分编译时间较长。
- **CANN 版本必须精确**：8.5.0 缺少 Qwen3.5 所需 patch，8.5.2 未经验证，混版本有 torch-npu 算子不兼容风险。
