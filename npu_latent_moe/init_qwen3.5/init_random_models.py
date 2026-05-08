"""
Create randomly initialized Qwen3.5-35B-A3B (MoE) and Qwen3.5-4B (dense)
models in HuggingFace format, using configs pulled directly from HuggingFace Hub.

Output directories (relative to this script):
    qwen35_35B_A3B_init_ckpt/   ← Qwen3.5-35B-A3B
    qwen35_4B_init_ckpt/        ← Qwen3.5-4B

Usage:
    python init_random_models.py [--model {moe,dense,both}]
                                  [--dtype {bfloat16,float16,float32}]
                                  [--device {npu,cpu}] [--npu-id INT]
                                  [--save-tokenizer]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = {
    "moe": {
        "label": "Qwen3.5-35B-A3B (MoE)",
        "hf_id": "Qwen/Qwen3.5-35B-A3B",
        "out_dir": SCRIPT_DIR / "qwen35_35B_A3B_init_ckpt",
    },
    "dense": {
        "label": "Qwen3.5-4B (dense)",
        "hf_id": "Qwen/Qwen3.5-4B",
        "out_dir": SCRIPT_DIR / "qwen35_4B_init_ckpt",
    },
}


def _init_npu(npu_id: int) -> torch.device:
    try:
        import torch_npu  # noqa: F401
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise RuntimeError("torch_npu is installed but no NPU device found.")
        torch.npu.set_device(npu_id)
        return torch.device(f"npu:{npu_id}")
    except ImportError:
        raise RuntimeError("torch_npu not installed. Use --device cpu or install torch_npu.")


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def param_count(model) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"{total/1e9:.2f}B" if total >= 1e9 else f"{total/1e6:.2f}M"


def create_random_model(hf_id: str, dtype: torch.dtype, device: torch.device):
    import json, shutil, tempfile
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModelForCausalLM

    print(f"  Downloading config.json from HuggingFace: {hf_id}", flush=True)
    config_file = hf_hub_download(repo_id=hf_id, filename="config.json")
    with open(config_file) as f:
        config_dict = json.load(f)

    # Write to a temp dir so AutoConfig resolves the model type via the local
    # file — avoids trust_remote_code config classes that may not expose
    # vocab_size as a standard PretrainedConfig attribute.
    tmp = Path(tempfile.mkdtemp())
    try:
        (tmp / "config.json").write_text(json.dumps(config_dict))
        cfg = AutoConfig.from_pretrained(tmp)
        # Patch vocab_size in case the config class didn't forward it to super()
        if not hasattr(cfg, "vocab_size"):
            cfg.vocab_size = config_dict["vocab_size"]
        print("  Randomly initializing weights on CPU ...", flush=True)
        with torch.device("cpu"):
            model = AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if device.type != "cpu":
        print(f"  Moving model to {device} ...", flush=True)
        model = model.to(device)

    return model


def save_model(model, out_dir: Path, hf_id: str, save_tokenizer: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Saving checkpoint to {out_dir} ...", flush=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    if save_tokenizer:
        try:
            from transformers import AutoTokenizer
            print(f"  Downloading tokenizer from {hf_id} ...", flush=True)
            tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            tok.save_pretrained(out_dir)
        except Exception as e:
            print(f"  WARN: tokenizer download failed ({e}); skipping.", flush=True)

    print(f"  Saved → {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["moe", "dense", "both"], default="both",
                   help="Which model(s) to create (default: both)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", choices=["npu", "cpu"], default="cpu",
                   help="Device for weight init (cpu is fine; ckpt is device-agnostic)")
    p.add_argument("--npu-id", type=int, default=0)
    p.add_argument("--no-tokenizer", action="store_true",
                   help="Skip downloading and saving the tokenizer")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = _init_npu(args.npu_id) if args.device == "npu" else torch.device("cpu")
    print(f"Device: {device}  dtype: {args.dtype}", flush=True)

    dtype = _dtype(args.dtype)
    keys = ["moe", "dense"] if args.model == "both" else [args.model]

    for key in keys:
        info = MODELS[key]
        print(f"\n{'='*60}", flush=True)
        print(f"Creating {info['label']}", flush=True)

        model = create_random_model(info["hf_id"], dtype, device)
        print(f"  Parameters: {param_count(model)}", flush=True)

        save_model(model, info["out_dir"], info["hf_id"], not args.no_tokenizer)

    print(f"\nDone. Checkpoints saved under {SCRIPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
