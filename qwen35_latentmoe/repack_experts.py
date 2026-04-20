#!/usr/bin/env python
"""Check and convert legacy per-expert unpacked weights into packed tensors.

Usage:
    python repack_experts.py <path/to/model.safetensors>
    python repack_experts.py <path/to/checkpoint_dir>

When given a directory, looks for model.safetensors (single file) or
model-*.safetensors (sharded) inside it.

Modes:
    --check     Report format without modifying anything (default).
    --convert   Convert legacy → packed and overwrite in place.
                Original file backed up as model.safetensors.legacy.
    --dry-run   Show what --convert would do without writing.

Examples:
    python repack_experts.py --check  qwen35_latentmoe/sandeep_latentmoe_40layers_original_ckpt
    python repack_experts.py --convert qwen35_latentmoe/rezaul_latentmoe_24layers_original_ckpt
"""
import argparse
import os
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def find_safetensors(path: str) -> list[str]:
    """Return list of safetensors files from a file path or directory."""
    p = Path(path)
    if p.is_file() and p.suffix == '.safetensors':
        return [str(p)]
    if p.is_dir():
        single = p / 'model.safetensors'
        if single.exists():
            return [str(single)]
        shards = sorted(p.glob('model-*.safetensors'))
        if shards:
            return [str(s) for s in shards]
    return []


def load_state_dict(files: list[str]) -> dict[str, torch.Tensor]:
    """Load all tensors from one or more safetensors files."""
    state_dict = {}
    for f in files:
        with safe_open(f, framework='pt') as sf:
            for k in sf.keys():
                state_dict[k] = sf.get_tensor(k)
    return state_dict


def analyze(state_dict: dict[str, torch.Tensor]) -> dict:
    """Analyze expert format in the state_dict."""
    layer_indices = sorted(set(
        int(m.group(1))
        for k in state_dict
        for m in [re.search(r'\.layers\.(\d+)\.', k)]
        if m
    ))

    packed_layers = []
    legacy_layers = []
    mixed_layers = []

    for idx in layer_indices:
        # Try multiple prefix patterns
        for prefix_tmpl in [
            'model.language_model.layers.{}.mlp.experts.',
            'model.layers.{}.mlp.experts.',
        ]:
            prefix = prefix_tmpl.format(idx)
            has_packed = (f'{prefix}gate_up_proj' in state_dict
                         or f'{prefix}down_proj' in state_dict)
            has_legacy = f'{prefix}0.gate_proj.weight' in state_dict

            if has_packed and has_legacy:
                mixed_layers.append(idx)
                break
            elif has_packed:
                packed_layers.append(idx)
                break
            elif has_legacy:
                legacy_layers.append(idx)
                break

    # Count experts in first legacy layer
    num_experts = 0
    if legacy_layers:
        idx = legacy_layers[0]
        for prefix_tmpl in [
            'model.language_model.layers.{}.mlp.experts.',
            'model.layers.{}.mlp.experts.',
        ]:
            prefix = prefix_tmpl.format(idx)
            if f'{prefix}0.gate_proj.weight' in state_dict:
                while f'{prefix}{num_experts}.gate_proj.weight' in state_dict:
                    num_experts += 1
                break

    return {
        'total_keys': len(state_dict),
        'layer_indices': layer_indices,
        'packed_layers': packed_layers,
        'legacy_layers': legacy_layers,
        'mixed_layers': mixed_layers,
        'num_experts': num_experts,
    }


def repack(state_dict: dict[str, torch.Tensor], num_experts: int, legacy_layers: list[int]) -> int:
    """Convert legacy per-expert keys → packed tensors in-place. Returns count of repacked layers."""
    repacked = 0
    for idx in legacy_layers:
        for prefix_tmpl in [
            'model.language_model.layers.{}.mlp.experts.',
            'model.layers.{}.mlp.experts.',
        ]:
            prefix = prefix_tmpl.format(idx)
            if f'{prefix}0.gate_proj.weight' not in state_dict:
                continue

            gate_rows, up_rows, down_rows = [], [], []
            legacy_keys = []
            complete = True
            for e in range(num_experts):
                gk = f'{prefix}{e}.gate_proj.weight'
                uk = f'{prefix}{e}.up_proj.weight'
                dk = f'{prefix}{e}.down_proj.weight'
                if gk not in state_dict or uk not in state_dict or dk not in state_dict:
                    print(f'  WARNING: layer {idx} expert {e} incomplete, skipping layer')
                    complete = False
                    break
                gate_rows.append(state_dict[gk])
                up_rows.append(state_dict[uk])
                down_rows.append(state_dict[dk])
                legacy_keys.extend([gk, uk, dk])

            if not complete:
                continue

            packed_gate_up = torch.cat([torch.stack(gate_rows), torch.stack(up_rows)], dim=1)
            packed_down = torch.stack(down_rows)

            state_dict[f'{prefix}gate_up_proj'] = packed_gate_up
            state_dict[f'{prefix}down_proj'] = packed_down
            for k in legacy_keys:
                del state_dict[k]

            repacked += 1
            break

    return repacked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='Path to model.safetensors file or checkpoint directory')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true', default=True, help='Report format only (default)')
    group.add_argument('--convert', action='store_true', help='Convert legacy → packed in place')
    group.add_argument('--dry-run', action='store_true', help='Show what --convert would do')
    args = parser.parse_args()

    files = find_safetensors(args.path)
    if not files:
        print(f'ERROR: No safetensors files found at {args.path}', file=sys.stderr)
        sys.exit(1)

    print(f'Loading {len(files)} safetensors file(s)...')
    state_dict = load_state_dict(files)
    info = analyze(state_dict)

    print(f'Total keys:     {info["total_keys"]}')
    print(f'Total layers:   {len(info["layer_indices"])}')
    print(f'Packed layers:  {len(info["packed_layers"])}')
    print(f'Legacy layers:  {len(info["legacy_layers"])}')
    print(f'Mixed layers:   {len(info["mixed_layers"])}')
    if info['num_experts']:
        print(f'Experts/layer:  {info["num_experts"]}')

    if info['legacy_layers']:
        sample_prefix = None
        idx = info['legacy_layers'][0]
        for tmpl in ['model.language_model.layers.{}.mlp.experts.', 'model.layers.{}.mlp.experts.']:
            if f'{tmpl.format(idx)}0.gate_proj.weight' in state_dict:
                sample_prefix = tmpl.format(idx)
                break
        if sample_prefix:
            g = state_dict[f'{sample_prefix}0.gate_proj.weight']
            print(f'Legacy expert 0 gate_proj: shape={tuple(g.shape)} dtype={g.dtype}')

    if info['packed_layers']:
        idx = info['packed_layers'][0]
        for tmpl in ['model.language_model.layers.{}.mlp.experts.', 'model.layers.{}.mlp.experts.']:
            k = f'{tmpl.format(idx)}gate_up_proj'
            if k in state_dict:
                t = state_dict[k]
                zeros_pct = (t == 0).sum().item() / t.numel() * 100
                print(f'Packed gate_up_proj: shape={tuple(t.shape)} dtype={t.dtype} zeros={zeros_pct:.1f}%')
                break

    # Verdict
    print()
    if not info['legacy_layers'] and info['packed_layers']:
        print('STATUS: Already packed. No conversion needed.')
        return
    if not info['legacy_layers'] and not info['packed_layers']:
        print('STATUS: No expert keys found.')
        return
    if info['legacy_layers']:
        print(f'STATUS: {len(info["legacy_layers"])} layers need conversion '
              f'({len(info["legacy_layers"]) * info["num_experts"] * 3} per-expert keys '
              f'→ {len(info["legacy_layers"]) * 2} packed keys)')

    if not args.convert and not args.dry_run:
        print('\nRun with --convert to convert in place, or --dry-run to preview.')
        return

    # Repack
    print('\nRepacking...')
    count = repack(state_dict, info['num_experts'], info['legacy_layers'])
    print(f'Repacked {count} layers. Final keys: {len(state_dict)}')

    if args.dry_run:
        print('\n(dry-run: no files written)')
        return

    # Save
    if len(files) == 1:
        dst = files[0]
        legacy = dst + '.legacy'
        if not os.path.exists(legacy):
            os.rename(dst, legacy)
            print(f'Backed up original to {legacy}')
        else:
            print(f'Legacy backup already exists at {legacy}, overwriting output only')
        print(f'Saving packed checkpoint to {dst}...')
        save_file(state_dict, dst)
        print(f'Done. Size: {os.path.getsize(dst) / 1e9:.1f} GB')
    else:
        print('ERROR: Sharded safetensors conversion not supported. '
              'Merge shards first, then convert.', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
