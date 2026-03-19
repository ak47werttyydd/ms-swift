# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SWIFT** (Scalable lightWeight Infrastructure for Fine-Tuning) is a full-pipeline LLM framework supporting 600+ LLMs and 400+ multimodal models. It covers pre-training, SFT, RLHF, inference, evaluation, quantization, and deployment.

Current version: `swift/version.py`

## Common Commands

### Build
```bash
make whl                        # Build wheel: python setup.py sdist bdist_wheel
make clean                      # Remove build artifacts
pip install -e .                # Install in editable mode
```

### Lint
```bash
make linter                     # Runs .dev_scripts/linter.sh
pre-commit run --all-files      # Run all pre-commit hooks (flake8, isort, yapf)
```

Code style: line length 120, isort with `known_first_party=swift`, yapf PEP8.

### Tests
```bash
make test                       # Runs .dev_scripts/citest.sh
python -m unittest tests.llm.test_dataset.TestDataset.test_load   # Single test
python -m unittest tests.tuners.test_peft                          # All tests in module
```

Tests use Python `unittest`. No pytest config — use `python -m unittest` for discovery.

### CLI Usage
```bash
swift sft --model Qwen/Qwen2.5-7B --dataset alpaca-en
swift pt --model Qwen/Qwen2.5-7B --dataset swift/chinese-c4
swift infer --model Qwen/Qwen2.5-7B --ckpt_dir ./output
swift rlhf --rlhf_type dpo --model Qwen/Qwen2.5-7B
swift export --model Qwen/Qwen2.5-7B --quant_method awq
swift eval --model Qwen/Qwen2.5-7B
swift deploy --model Qwen/Qwen2.5-7B

# Multi-GPU via env vars (triggers torchrun automatically):
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft ...

# Pass a YAML config:
swift sft --config my_config.yaml
```

## Architecture

### Request Flow
`swift <cmd>` → `swift/cli/main.py:cli_main()` routes via `ROUTE_MAPPING` → invokes the corresponding `swift/cli/<cmd>.py` via subprocess (with `torchrun` if `NPROC_PER_NODE`/`NNODES` env vars are set) → calls pipeline function in `swift/pipelines/`.

### Key Modules

| Module | Role |
|--------|------|
| `swift/cli/` | CLI entry points; `main.py` contains `ROUTE_MAPPING` |
| `swift/pipelines/` | High-level workflows: `train/` (sft, pt, rlhf), `infer/`, `export/`, `eval/`, `app/` |
| `swift/trainers/` | HuggingFace Trainer subclasses: `Trainer`, `Seq2SeqTrainer`, `EmbeddingTrainer`, `RerankerTrainer` |
| `swift/rlhf_trainers/` | RLHF algorithms: DPO, KTO, GRPO, etc. |
| `swift/model/` | Model registration (`register.py`), patching (`patcher.py`), architecture definitions (`model_arch.py`), NPU patches |
| `swift/dataset/` | Dataset loading (`loader.py`), registration (`register.py`), preprocessing (`preprocessor/`) |
| `swift/template/` | Prompt templates per model family |
| `swift/arguments/` | Dataclass-based argument definitions for all CLI commands |
| `swift/tuners/` | PEFT integrations: LoRA, QLoRA, DoRA, LongLoRA, ReFT |
| `swift/infer_engine/` | Inference backend adapters: Transformers, vLLM, SGLang, LMDeploy |
| `swift/megatron/` | Megatron-LM parallel training integration (separate `megatron` CLI entry point) |
| `swift/utils/` | Shared utilities; uses `_LazyModule` pattern for efficient imports |
| `swift/ui/` | Gradio-based web UI |
| `swift/ray/` | Ray distributed training integration |

### Model & Dataset Registry Pattern
Both models and datasets use a registration pattern:
- **Register**: decorate with `@register_model` / `@register_dataset` in `swift/model/models/` and `swift/dataset/dataset/`
- **Lookup**: resolved at runtime via `swift/model/register.py` and `swift/dataset/register.py`
- New models/datasets only need to add a registered class — no central list to update

### Lazy Imports
`swift/__init__.py` uses `_LazyModule` (from `swift/utils/import_utils.py`) to defer all submodule imports until first access. When adding new public exports, update the `_import_structure` dict there.

### Distributed Training
- Multi-GPU: set `NPROC_PER_NODE` env var; the CLI auto-wraps with `torchrun`
- Multi-node: also set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`
- DeepSpeed: pass `--deepspeed zero2/zero3`
- Megatron: use `megatron` CLI entry point (`swift/cli/_megatron/`)

### Configuration
- All training arguments are dataclasses in `swift/arguments/`
- YAML configs supported via `--config file.yaml` (converted to CLI args at runtime by `prepare_config_args`)
- Model caching: ModelScope uses `MODELSCOPE_CACHE` (default `~/.cache/modelscope/hub`); HuggingFace uses `HF_HOME`
