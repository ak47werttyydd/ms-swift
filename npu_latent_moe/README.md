# NPU LatentMoE — GKD Distillation Workspace

在 Ascend NPU(Atlas 800/900 A2,8× 910B3 64GB)上用 vLLM 托管 **Qwen3.5-35B-A3B 教师**,
通过 ms-swift GKD 蒸馏到更小的学生模型。

本目录覆盖两种学生:
- **标准 Qwen3.5**(`Qwen/Qwen3.5-4B` 等) —— 走 HF + DeepSpeed Zero3 路径,即开即跑。
- **Qwen3.5-LatentMoE**(40 层自定义架构) —— DeepSpeed Zero3 路径已打通;MindSpeed/Megatron 路径留有脚手架,需补 mcore-bridge converter。

---

## 1. 硬件 & 系统前提

| 项 | 要求 |
|---|---|
| 机型 | Atlas 800/900 A2,8× 910B3 或 910C 64GB |
| OS | Linux(aarch64 或 x86_64) |
| CANN | 8.5.1 已安装,`/usr/local/Ascend/ascend-toolkit/set_env.sh` 存在 |
| 驱动 | `npu-smi info` 能正常输出 8 张卡 |
| Conda | miniconda/anaconda 任一,`conda` 在 PATH |
| 磁盘 | ≥ 200 GB(Megatron-LM + MindSpeed + mcore-bridge + 权重缓存) |

**每次新开 shell 都要先 source CANN**,否则所有 torch-npu 调用会失败:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 如果装了 NNAL/ATB:
source /usr/local/Ascend/nnal/atb/set_env.sh
```

---

## 2. 环境安装 —— 选一条路径

两条路径**不冲突**,可以并存为两个独立 conda env。教师服务(vLLM)两边通用,只在**学生训练后端**上有区别。

### 路径 A:DeepSpeed Zero3(推荐,覆盖绝大多数场景)

```bash
bash /Users/adrianhwang/Code/vllm-ascend/scripts/setup_npu_env.sh
```
装出来的 env 叫 `qwen35-gkd-npu`。装好后验证:
```bash
conda activate qwen35-gkd-npu
python -c "import torch, torch_npu, vllm_ascend, swift; \
           print('torch:', torch.__version__, 'NPUs:', torch.npu.device_count())"
```

### 路径 B:MindSpeed(Megatron-LM + TP/EP/PP)

只有当学生规模 ≥ 10B 或者卡数 ≥ 32 时才值得走这条路:
```bash
bash /Users/adrianhwang/Code/vllm-ascend/scripts/setup_npu_env_mindspeed.sh
```
装出来的 env 叫 `qwen35-gkd-npu-mindspeed`。额外安装 Megatron-LM v0.15.3 + MindSpeed core_r0.15.3 + mcore-bridge。`MEGATRON_LM_PATH` 会写入 conda `activate.d/`,激活即生效。验证:
```bash
conda activate qwen35-gkd-npu-mindspeed
python -c "import mindspeed.megatron_adaptor; \
           from swift.megatron.init import init_megatron_env; \
           init_megatron_env(); print('OK')"
```

> **版本绑定不能动**:torch 2.9.0 由 vllm-ascend releases/v0.18.0 强制;Megatron-LM v0.15.3 必须配 MindSpeed core_r0.15.3,错版号会在导入时报兼容性错误。

---

## 3. 权重 & 数据准备

### 3.1 教师权重 `Qwen/Qwen3.5-35B-A3B`
让 `vllm serve` 自己从 ModelScope / HF 拉取即可,占约 65 GB,第一次启动会慢。
预下载可以用:
```bash
modelscope download --model Qwen/Qwen3.5-35B-A3B --local_dir ~/models/Qwen3.5-35B-A3B
# 然后用本地路径替换脚本里的 TEACHER_MODEL
```

### 3.2 学生权重
**标准 Qwen3.5-4B**(用于 `gkd_qwen35_vllm_npu.sh`):自动下载,无需处理。

**Qwen3.5-LatentMoE 40 层**(用于 `gkd_latentmoe_vllm_npu.sh`):
1. 放置 safetensors 到 `ckpt_latentmoe_40l/`;如果原始权重是**按专家分离**的,先跑:
   ```bash
   cd npu_latent_moe
   python repack_ckpt.py   # 打包成 3D experts.gate_up_proj / experts.down_proj
   ```
2. 从参考 repo `sandeep_latentmoe_40layers_original_ckpt/` 拷三个文件到 `ckpt_latentmoe_40l/`:
   - `tokenizer.json`
   - `tokenizer_config.json`
   - `chat_template.jinja`

`ckpt_latentmoe_40l/config.json` 已经写好(`use_grouped_expert_matmul=true`, `use_npu_rmsnorm=true`,架构指向 `Qwen3_5LatentMoeForCausalLM`),不要再手动编辑。

### 3.3 数据
- **标准 Qwen3.5 路径**:nemotron-CC 任意一份未分片的目录,`DATASET_DIR=/data/nemotron-cc`(脚本占位,改成你的真实路径)。
- **LatentMoE 路径**:需要**预采样**好的教师生成 JSONL 分片(每行含 `response` 字段)。在脚本里填 `PRESAMPLE_DATA_ARGS=( "$SCRIPT_DIR/data/shard0.jsonl" ... )` 或通过环境变量传 `PRESAMPLE_DATA='path1 path2' bash ...`。

---

## 4. 文件清单

| 文件 | 作用 | 何时用 |
|---|---|---|
| `ckpt_latentmoe_40l/config.json` | LatentMoE 学生模型配置(40L,256 experts,`use_latent_moe`) | LatentMoE 脚本 |
| `ckpt_latentmoe_40l/v5molae_qwen35_latentmoe.py` | LatentMoE 模型代码(auto_map 指向它) | LatentMoE 脚本 |
| `repack_ckpt.py` | 把按-expert 的权重打包成 3D tensor | 只在初次准备 LatentMoE 权重时 |
| `gkd_plugin.py` | 注册 `qwen3_5_latentmoe` 给 ms-swift,设 Zero3 leaf_modules | LatentMoE + DeepSpeed |
| `gkd_plugin_mindspeed.py` | 同上,但去掉 Zero3 钩子,附 mcore-bridge TODO 脚手架 | LatentMoE + MindSpeed(未跑通) |
| `gkd_qwen35_vllm_npu.sh` | 标准 Qwen3.5 学生 + DeepSpeed + nemotron-CC | **日常最常用** |
| `gkd_latentmoe_vllm_npu.sh` | LatentMoE 学生 + DeepSpeed + 预采样 JSONL | LatentMoE 训练 |
| `gkd_latentmoe_vllm_npu_mindspeed.sh` | LatentMoE 学生 + Megatron,目前会在 bridge 缺失处死 | 不要直接跑,先补 bridge |

---

## 5. 跑脚本

所有训练脚本都**自带教师生命周期管理**:脚本内部启动 vLLM server → 等待 600s 直到 `/health` 通 → 开始训练 → 训练结束/报错时 kill 教师 + watchdog。不需要手动开两个 shell。

### 5.1 场景 A:蒸馏到标准 Qwen3.5-4B(最快上手)

```bash
conda activate qwen35-gkd-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /Users/adrianhwang/Code/ms-swift/npu_latent_moe

# 先改 DATASET_DIR 为你的 nemotron-CC 真实路径
# 或者直接 export:
export DATASET_DIR=/path/to/nemotron-cc
bash gkd_qwen35_vllm_npu.sh |& tee gkd_qwen35_vllm_npu.log
```

默认 NPU 布局:教师用 NPUs 4-7(TP=4),学生用 NPUs 0-3(Zero3)。如需改动:
```bash
TEACHER_NPUS="0,1,2,3" STUDENT_NPUS="4,5,6,7" bash gkd_qwen35_vllm_npu.sh
```

### 5.2 场景 B:蒸馏到 Qwen3.5-LatentMoE(DeepSpeed 路径)

```bash
conda activate qwen35-gkd-npu
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /Users/adrianhwang/Code/ms-swift/npu_latent_moe

# 确保 ckpt_latentmoe_40l/ 下有完整权重 + tokenizer 文件
# 填好预采样数据:
export PRESAMPLE_DATA="$PWD/data/shard0.jsonl $PWD/data/shard1.jsonl"
bash gkd_latentmoe_vllm_npu.sh |& tee gkd_latentmoe_vllm_npu.log
```

### 5.3 场景 C:LatentMoE + MindSpeed(**目前会失败**)

```bash
conda activate qwen35-gkd-npu-mindspeed
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /Users/adrianhwang/Code/ms-swift/npu_latent_moe

bash gkd_latentmoe_vllm_npu_mindspeed.sh
# 预期失败:mcore-bridge 报 model_type 'qwen3_5_latentmoe' not registered
```

跑通的前提是先把 `gkd_plugin_mindspeed.py` 里 `_register_mindspeed_bridge()` 下的 TODO 全部实现(最大头是**把 linear_attention 层移植到 Megatron 并行原语**,Megatron-LM v0.15.3 没有对应层)。细节见该文件头部 docstring。

### 5.4 中途杀脚本

所有脚本都装了 `trap _global_cleanup EXIT INT TERM`,`Ctrl-C` 会按顺序杀 watchdog → 学生进程组 → 教师。如果 `Ctrl-C` 按两次可能残留 NPU 进程,检查:
```bash
npu-smi info            # 看哪张卡还在占
pkill -f "vllm serve"   # 清教师
pkill -f "swift rlhf"   # 清学生(DeepSpeed 路径)
pkill -f "megatron"     # 清学生(MindSpeed 路径)
```

---

## 6. 参数速查

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `TEACHER_MODEL` | `Qwen/Qwen3.5-35B-A3B` | 教师模型名或本地路径 |
| `TEACHER_NPUS` | `4,5,6,7` | 教师用的 NPU(逗号分隔) |
| `TEACHER_TP` | `4` | vLLM 张量并行大小 |
| `TEACHER_MAX_MODEL_LEN` | `4610`(qwen35) / `5000`(latentmoe) | vLLM `--max-model-len`,必须 ≥ `max_length + max_completion_length` |
| `TEACHER_MAX_LOGPROBS` | `64` | vLLM `--max-logprobs`,对应 `--gkd_logits_topk` |
| `TEACHER_NPU_MEM_UTIL` | `0.85` | 教师 HBM 利用率上限 |
| `STUDENT_NPUS` | `0,1,2,3` | 学生训练用的 NPU |
| `STUDENT_NPROC` | `4` | 学生 DDP world size(= 学生 NPU 数) |
| `DATASET_DIR` | `/data/nemotron-cc`(占位) | 标准 Qwen3.5 脚本的数据目录 |
| `PRESAMPLE_DATA` | 空 | LatentMoE 脚本的预采样 JSONL,空格分隔 |

**不要手动减小 `TEACHER_MAX_MODEL_LEN`**:vLLM 强制 `input_tokens + max_tokens ≤ max_model_len`,而学生的 `--max_length 4607 + --max_completion_length 1 = 4608` 已经占掉了大部分;减太小会让 mid-stream prompt(含 BOS + assistant token 的)直接把教师 server 顶崩。

---

## 7. 排错清单

| 症状 | 原因 / 处理 |
|---|---|
| `npu-smi: command not found` | CANN 没 source。先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。 |
| 教师 600s 内没 ready | 35B-A3B 冷启动就是慢,watchdog 已放宽到 600s;还是没起来的话看 vLLM log 里是否 `max_model_len` 超出 HBM。 |
| 学生报 `model_type not registered in mcore_bridge` | 你在跑 MindSpeed 版,且没实现 bridge。回退到 DeepSpeed 脚本或实现 converter。 |
| 学生报 `Qwen3_5LatentMoeSparseMoeBlock is not a leaf module` | DeepSpeed Zero3 没看到 leaf 标记。检查是否用的是 `gkd_plugin.py` 而不是 `gkd_plugin_mindspeed.py`。 |
| `torch_npu.npu_rms_norm` 报 ImportError | torch-npu 版本错了。脚本要求 torch-npu==2.9.0,别让别的安装覆盖。 |
| OOM(教师端) | 降 `TEACHER_NPU_MEM_UTIL` 到 0.80;或者 `TEACHER_MAX_MODEL_LEN` 压到刚好 `4608`(无 headroom,风险自负)。 |
| OOM(学生端,DeepSpeed) | 降 `--per_device_train_batch_size`,相应调高 `--gradient_accumulation_steps` 保持总 batch 不变。 |
| `vllm serve` 启动时报 `flash_attn` ImportError | 不该发生。确认没有用 `--attn_impl flash_attn` 传给教师——教师是推理端,自己用 `--enforce-eager`。 |

---

## 8. 后端选择速查

| 情形 | 推荐脚本 |
|---|---|
| 想今天就跑起来 | `gkd_qwen35_vllm_npu.sh`(DeepSpeed) |
| 学生是 LatentMoE,≤8 NPU | `gkd_latentmoe_vllm_npu.sh`(DeepSpeed) |
| 学生 ≥ 10B 且 ≥ 32 NPU | 先把 `gkd_plugin_mindspeed.py` 的 TODO 实现,再用 `_mindspeed` 脚本 |
| 纯标准 Qwen3.5 大规模(32+ 卡) | 新写一个 `gkd_qwen35_vllm_npu_mindspeed.sh`(mcore-bridge 已内置 qwen3_moe,不需要写 bridge) |

---

## 9. 相关文档

- vllm-ascend 仓内层: `/Users/adrianhwang/Code/vllm-ascend/CLAUDE.md`, `AGENTS.md`
- ms-swift NPU 官方指南: `ms-swift/docs/source_en/BestPractices/NPU-support.md`
- ms-swift Megatron GKD 参考脚本: `ms-swift/examples/megatron/rlhf/gkd/teacher_server.sh`
- vllm-ascend 的 Qwen3.5 patch: `vllm_ascend/patch/worker/patch_qwen3_5.py` + `tests/e2e/multicard/4-cards/test_qwen3_5.py`
