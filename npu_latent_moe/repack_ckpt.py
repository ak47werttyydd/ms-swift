"""
Repack a LatentMoE checkpoint saved in "individual-expert" format into the
packed-3D-tensor format expected by Qwen3_5LatentExperts.

Two transformations applied:
  1. Strip "model.language_model." prefix  →  "model."
  2. Pack per-expert weights into 3D tensors:
       experts.{e}.gate_proj.weight  (I, latent)  \
       experts.{e}.up_proj.weight    (I, latent)   } → experts.gate_up_proj (E, 2I, latent)
       experts.{e}.down_proj.weight  (latent, I)   → experts.down_proj      (E, latent, I)

Usage:
    python repack_ckpt.py <src_dir> <dst_dir>

Example:
    python repack_ckpt.py \
        /home/a00652497/2026/bytedance_project/ckpt/qwen35_latentmoe_2x \
        /home/a00652497/2026/bytedance_project/ckpt/qwen35_latentmoe_2x_repacked
"""

import json
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


# ── helpers ──────────────────────────────────────────────────────────────────

EXPERT_RE = re.compile(
    r"^model\.language_model\.(layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)
LANG_MODEL_RE = re.compile(r"^model\.language_model\.")


def strip_prefix(key: str) -> str:
    return LANG_MODEL_RE.sub("model.", key)


# ── main ─────────────────────────────────────────────────────────────────────

def repack(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    # ── load all shards ──
    shard_files = sorted(src_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {src_dir}")

    print(f"Loading {len(shard_files)} shard(s) ...")
    state_dict: dict[str, torch.Tensor] = {}
    for sf in shard_files:
        state_dict.update(load_file(sf, device="cpu"))
    print(f"  total keys: {len(state_dict)}")

    # ── collect individual expert weights ──
    # Structure: layer_mlp_prefix → {expert_id → {proj_name → tensor}}
    expert_store: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    expert_keys: set[str] = set()

    for k, v in state_dict.items():
        m = EXPERT_RE.match(k)
        if m:
            mlp_prefix, e_str, proj = m.group(1), m.group(2), m.group(3)
            e = int(e_str)
            expert_store.setdefault(mlp_prefix, {}).setdefault(e, {})[proj] = v
            expert_keys.add(k)

    # ── build new state dict ──
    new_sd: dict[str, torch.Tensor] = {}

    # non-expert keys: just strip prefix
    for k, v in state_dict.items():
        if k not in expert_keys:
            new_sd[strip_prefix(k)] = v

    # pack experts
    for mlp_prefix, experts in sorted(expert_store.items()):
        num_experts = max(experts.keys()) + 1
        sample = next(iter(experts[0].values()))
        dtype = sample.dtype
        device = sample.device

        # gate_proj and up_proj each have shape (I, latent_dim)
        # down_proj has shape (latent_dim, I)
        gate_list = [experts[e]["gate_proj"] for e in range(num_experts)]
        up_list   = [experts[e]["up_proj"]   for e in range(num_experts)]
        down_list = [experts[e]["down_proj"] for e in range(num_experts)]

        # gate_up_proj: (E, 2*I, latent_dim)
        gate_up = torch.stack(
            [torch.cat([g, u], dim=0) for g, u in zip(gate_list, up_list)],
            dim=0,
        ).to(dtype=dtype, device=device)

        # down_proj: (E, latent_dim, I)
        down = torch.stack(down_list, dim=0).to(dtype=dtype, device=device)

        new_prefix = "model." + mlp_prefix  # e.g. model.layers.0.mlp
        new_sd[f"{new_prefix}.experts.gate_up_proj"] = gate_up
        new_sd[f"{new_prefix}.experts.down_proj"] = down

        print(
            f"  packed {new_prefix}.experts: "
            f"gate_up_proj {tuple(gate_up.shape)}  down_proj {tuple(down.shape)}"
        )

    # ── save ──
    out_path = dst_dir / "model.safetensors"
    print(f"\nSaving {len(new_sd)} keys → {out_path} ...")
    save_file(new_sd, out_path)
    print(f"  done  ({out_path.stat().st_size / 1e9:.2f} GB)")

    # ── copy and patch config ──
    cfg_src = src_dir / "config.json"
    if cfg_src.exists():
        with open(cfg_src) as f:
            cfg = json.load(f)
        # fix architectures to match the class name used in auto_map
        cfg["architectures"] = ["Qwen3_5LatentMoeForCausalLM"]
        with open(dst_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print("  config.json written")

    # copy tokenizer files
    for fname in src_dir.iterdir():
        if fname.suffix in (".json", ".model", ".tiktoken") and fname.name != "config.json":
            shutil.copy2(fname, dst_dir / fname.name)

    # copy modeling file for trust_remote_code
    modeling_src = src_dir / "molae_qwen35_latentmoe.py"
    if modeling_src.exists():
        shutil.copy2(modeling_src, dst_dir / modeling_src.name)

    print("\nDone.")
    print(f"  src  keys : {len(state_dict)}")
    print(f"  dst  keys : {len(new_sd)}")
    print(f"  expert keys removed : {len(expert_keys)}")
    print(f"  packed layers : {len(expert_store)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python repack_ckpt.py <src_dir> <dst_dir>")
        sys.exit(1)
    repack(Path(sys.argv[1]), Path(sys.argv[2]))
