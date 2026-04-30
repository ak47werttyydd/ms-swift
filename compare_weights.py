#!/usr/bin/env python3
"""Compare weights between two safetensors checkpoints.

Usage:
    python compare_weights.py <ckpt_a> <ckpt_b> [--filter PATTERN]

Examples:
    python compare_weights.py path/to/ckpt-900 path/to/ckpt-100
    python compare_weights.py path/to/ckpt-900 path/to/ckpt-100 --filter self_attn
    python compare_weights.py path/to/ckpt-900 path/to/ckpt-100 --filter linear_attn
"""

import argparse
import json
from safetensors import safe_open


def load_index(ckpt):
    with open(f'{ckpt}/model.safetensors.index.json') as f:
        return json.load(f)['weight_map']


def load_tensor(ckpt, key, index):
    fname = index[key]
    with safe_open(f'{ckpt}/{fname}', framework='pt') as f:
        return f.get_tensor(key)


def compare_group(label, keys, ckpt_a, ckpt_b, idx_a, idx_b, threshold=1e-6):
    print(f'=== {label} ({len(keys)} keys) ===')
    if not keys:
        print('  (no keys matched)')
        return

    diffs = []
    for k in keys:
        t_a = load_tensor(ckpt_a, k, idx_a).float()
        t_b = load_tensor(ckpt_b, k, idx_b).float()
        diff = (t_a - t_b).abs().max().item()
        diffs.append((k, diff))

    max_diff = max(d for _, d in diffs)
    changed = [(k, d) for k, d in diffs if d > threshold]
    print(f'  Max diff: {max_diff:.6e}')
    print(f'  Changed (> {threshold:.0e}): {len(changed)}/{len(keys)}')
    if changed:
        for k, d in changed:
            print(f'    {k}: {d:.6e}')
    else:
        print('  All weights identical.')
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare safetensors checkpoint weights.')
    parser.add_argument('ckpt_a', help='Path to first checkpoint directory')
    parser.add_argument('ckpt_b', help='Path to second checkpoint directory')
    parser.add_argument('--filter', default=None,
                        help='Only compare keys containing this substring (e.g. self_attn, linear_attn)')
    parser.add_argument('--threshold', type=float, default=1e-6,
                        help='Diff threshold to consider a weight changed (default: 1e-6)')
    args = parser.parse_args()

    idx_a = load_index(args.ckpt_a)
    idx_b = load_index(args.ckpt_b)

    all_keys = sorted(idx_a.keys())

    if args.filter:
        groups = {args.filter: [k for k in all_keys if args.filter in k]}
    else:
        groups = {
            'self_attn': [k for k in all_keys if 'self_attn' in k],
            'linear_attn (GDN)': [k for k in all_keys if 'linear_attn' in k],
        }

    print(f'Comparing:\n  A: {args.ckpt_a}\n  B: {args.ckpt_b}\n')
    for label, keys in groups.items():
        compare_group(label, keys, args.ckpt_a, args.ckpt_b, idx_a, idx_b, args.threshold)


if __name__ == '__main__':
    main()
