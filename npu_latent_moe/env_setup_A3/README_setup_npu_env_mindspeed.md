# setup_npu_env_mindspeed.sh

在 Ascend NPU 主机上为 Qwen3.5 GKD 蒸馏工作流搭建 conda 环境（MindSpeed / Megatron 路径）。

Teacher（vllm-ascend）和 Student（`megatron rlhf`）运行在同一个 `ms-swift-mindspeed` conda env 内，分属两个进程/Shell。

与 `setup_npu_env.sh`（DeepSpeed 路径）的区别：

| 项目 | DeepSpeed 路径 | MindSpeed 路径（本脚本） |
|------|---------------|------------------------|
| Student 入口 | `swift rlhf` + DeepSpeed ZeRO-3 | `megatron rlhf` + TP/EP/PP |
| 额外依赖 | 无 | Megatron-LM、MindSpeed、mcore-bridge、apex-ascend |
| 权重格式 | HF 原生 | mcore-bridge 在线转换 |
| 适用规模 | ≤ 8 卡 | 32+ 卡，TP/EP 重度场景 |

---

## 硬件与系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | openEuler（linux-aarch64） |
| CPU 架构 | aarch64 |
| NPU | Ascend 910C / `ascend910_9391`（Atlas 900 A3 SuperPoD，PCI 0xD803，16 NPU/节点） |
| NPU 显存 | 910C 每 die 64 GB；teacher TP4 需 4 卡，student 视 TP/EP 设置而定 |
| CANN | **8.5.1**（必须，不能用 8.5.0 或 8.5.2） |
| conda | 已安装，在 PATH 中可用（脚本不自行安装 conda） |

---

## 版本矩阵

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.11 | |
| CANN | 8.5.1 | 由 CANN `.run` 包预先安装 |
| torch / torch-npu | 2.9.0 / 2.9.0 | vllm-ascend v0.18.0 硬依赖；覆盖 ms-swift 文档的 2.7.1 pin |
| vllm | v0.18.0 | source install，`VLLM_TARGET_DEVICE=empty` |
| vllm-ascend | releases/v0.18.0 | teacher 推理后端 |
| Megatron-LM | v0.15.3 | 通过 PYTHONPATH 引入，不 pip install |
| MindSpeed | core_r0.15.3 | Ascend kernel 替换层，**必须与 Megatron-LM 版本匹配** |
| mcore-bridge | >= 1.0.2 (latest) | HF ↔ mcore 权重转换，驱动 `megatron rlhf` |
| apex-ascend | latest | Megatron fused optimizer/layernorm（可选，失败时自动跳过） |
| ms-swift | 本地源码 | 包含 `requirements/megatron.txt` |
| transformers | >=4.57.4, <5.6 | |
| deepspeed | <0.19 | 作为 fallback 安装，Megatron 路径本身不需要 |

---

## 前置条件

1. CANN 8.5.1 已通过三个 `.run` 包安装，并能 source `set_env.sh`：
   ```bash
   /home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh
   /home/canada_group_account/a84400789/CANN8.5.1/nnal/atb/set_env.sh   # 可选，libatb.so
   ```
2. `npu-smi` 和 `git` 在 PATH 中可用。
3. 以下本地 git 仓库已 clone（路径可通过环境变量覆盖）：
   - `~/Code/vllm-ascend`（本仓库，需在 `releases/v0.18.0` 分支）
   - `~/Code/ms-swift`
   - `~/Code/vllm`（脚本会自动 clone，如已有则复用）
   - `~/Code/Megatron-LM`（脚本会自动 clone，如已有则复用）
   - `~/Code/MindSpeed`（脚本会自动 clone，如已有则复用）
   - `~/Code/mcore-bridge`（脚本会自动 clone，如已有则复用）

---

## 用法

```bash
# 默认配置
bash scripts/setup_npu_env_mindspeed.sh

# 自定义路径或开启国内镜像
VLLM_ASCEND_DIR=/data/vllm-ascend \
MS_SWIFT_DIR=/data/ms-swift \
MEGATRON_LM_REPO=/data/Megatron-LM \
MINDSPEED_REPO=/data/MindSpeed \
MCORE_BRIDGE_REPO=/data/mcore-bridge \
USE_CN_MIRROR=1 \
bash scripts/setup_npu_env_mindspeed.sh
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENV_NAME` | `ms-swift-mindspeed` | conda 环境名 |
| `PY_VERSION` | `3.11` | Python 版本 |
| `CANN_SETENV` | `/home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh` | CANN 环境脚本路径 |
| `NNAL_SETENV` | `/home/canada_group_account/a84400789/CANN8.5.1/nnal/atb/set_env.sh` | NNAL 环境脚本路径（可选） |
| `VLLM_ASCEND_DIR` | `~/Code/vllm-ascend` | vllm-ascend 本地仓库路径 |
| `MS_SWIFT_DIR` | `~/Code/ms-swift` | ms-swift 本地仓库路径 |
| `VLLM_DIR` | `~/Code/vllm` | vllm 本地仓库路径 |
| `MEGATRON_LM_REPO` | `~/Code/Megatron-LM` | Megatron-LM 本地仓库路径 |
| `MEGATRON_LM_TAG` | `v0.15.3` | Megatron-LM tag |
| `MINDSPEED_REPO` | `~/Code/MindSpeed` | MindSpeed 本地仓库路径 |
| `MINDSPEED_BRANCH` | `core_r0.15.3` | MindSpeed 分支 |
| `MCORE_BRIDGE_REPO` | `~/Code/mcore-bridge` | mcore-bridge 本地仓库路径 |
| `VLLM_ASCEND_BRANCH` | `releases/v0.18.0` | vllm-ascend 分支 |
| `VLLM_TAG` | `v0.18.0` | vllm tag |
| `USE_CN_MIRROR` | `0` | 设为 `1` 切换 pip 到阿里云镜像 |

---

## 安装完成后的使用方式

**Shell 1 — Teacher 推理服务（Qwen3.5-35B-A3B，TP4）**

```bash
source /home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh
conda activate ms-swift-mindspeed
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
vllm serve Qwen/Qwen3.5-35B-A3B \
    --tensor-parallel-size 4 --port 8000 \
    --max-logprobs 64 --max-model-len 4610 \
    --dtype bfloat16 --enforce-eager --trust-remote-code
```

**Shell 2 — Student GKD 训练（Qwen3.5-4B，Megatron TP2）**

```bash
source /home/canada_group_account/a84400789/CANN8.5.1/cann-8.5.1/set_env.sh
conda activate ms-swift-mindspeed
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3.5-4B \
    --teacher_model_server http://localhost:8000 \
    --gkd_logits_topk 64 \
    --dataset '/data/nemotron-cc' \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --attention_backend flash \
    --torch_dtype bfloat16 \
    --micro_batch_size 2 --global_batch_size 32 \
    --train_iters 1000 --lr 1e-5 \
    --max_length 4607 --max_completion_length 1 \
    --finetune --no_save_optim --no_save_rng \
    --save output/qwen3_5-gkd-mindspeed
```

---

## 注意事项

- **CLI 是 `megatron`，不是 `swift megatron rlhf`**：`megatron` 是 ms-swift 安装的独立入口，拼写错误会导致找不到命令。
- **Megatron-LM 不 pip install**：通过 `MEGATRON_LM_PATH` 加入 `PYTHONPATH`（conda activate.d 自动注入）。mcore-bridge 需要 patch Megatron 内部，因此必须保留源码树。
- **版本必须三对齐**：Megatron-LM v0.15.3 + MindSpeed core_r0.15.3 + mcore-bridge >= 1.0.2，任意一个版本不匹配会导致导入失败。
- **torch 安装**：torch-npu 从华为镜像安装，自动拉取 aarch64 兼容的 torch 2.9.0。不从 pytorch.org 安装（那里的 wheel 是 x86_64）。
- **flash-attn 不安装**：Megatron 上 `--attention_backend flash` 由 MindSpeed 接管，路由到 `torch_npu.npu_fusion_attention`，不需要 Dao-AILab 的 flash-attn。
- **LatentMoE 不支持 Megatron 路径**：mcore-bridge 无内置 `qwen3_5_latentmoe` 转换器，需手写 plugin。如需 LatentMoE，改用 DeepSpeed 路径（`setup_npu_env.sh`）。
- **CANN 版本必须精确**：8.5.0 缺少 Qwen3.5 所需 patch，8.5.2 未经验证，混版本有 torch-npu 算子不兼容风险。
