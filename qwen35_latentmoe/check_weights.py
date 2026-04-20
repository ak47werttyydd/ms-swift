#!/usr/bin/env python
"""Check checkpoint weights for anomalies (all-zero tensors, NaN, Inf).

Usage:
    python check_weights.py <path/to/checkpoint_dir_or_safetensors>

Checks every tensor and reports per-category:
  - Packed experts (gate_up_proj, down_proj)
  - latent_down_proj / latent_up_proj
  - Attention (linear_attn, self_attn/full)
  - Shared expert / shared_expert_gate
  - Router (mlp.gate)
  - A_log, dt_bias, linear_attn.norm
  - Layer norms, Embeddings, lm_head
  - Vision encoder
"""
import argparse
import os
import sys
from pathlib import Path

from safetensors import safe_open


def find_safetensors(path: str) -> list[str]:
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


def categorize(key: str) -> str:
    if '.experts.gate_up_proj' in key:
        return 'Packed experts (gate_up_proj)'
    if '.experts.down_proj' in key:
        return 'Packed experts (down_proj)'
    if 'latent_down_proj' in key:
        return 'latent_down_proj'
    if 'latent_up_proj' in key:
        return 'latent_up_proj'
    if 'linear_attn.A_log' in key:
        return 'A_log'
    if 'linear_attn.dt_bias' in key:
        return 'dt_bias'
    if 'linear_attn.norm' in key:
        return 'linear_attn.norm'
    if 'linear_attn' in key:
        return 'Attention (linear_attn)'
    if 'self_attn' in key:
        return 'Attention (self_attn/full)'
    if 'shared_expert_gate' in key:
        return 'Shared expert gate'
    if 'shared_expert' in key:
        return 'Shared expert'
    if 'mlp.gate' in key:
        return 'Router (mlp.gate)'
    if 'input_layernorm' in key or 'post_attention_layernorm' in key or key.endswith('.norm.weight'):
        return 'Layer norms'
    if 'embed_tokens' in key:
        return 'Embeddings'
    if 'lm_head' in key:
        return 'lm_head'
    if '.visual.' in key:
        return 'Vision'
    return 'Other'


# Display order
CATEGORY_ORDER = [
    'Packed experts (gate_up_proj)',
    'Packed experts (down_proj)',
    'latent_down_proj',
    'latent_up_proj',
    'Attention (linear_attn)',
    'Attention (self_attn/full)',
    'Shared expert',
    'Shared expert gate',
    'Router (mlp.gate)',
    'A_log',
    'dt_bias',
    'linear_attn.norm',
    'Layer norms',
    'Embeddings',
    'lm_head',
    'Vision',
    'Other',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='Path to safetensors file or checkpoint directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='List every problematic tensor')
    args = parser.parse_args()

    files = find_safetensors(args.path)
    if not files:
        print(f'ERROR: No safetensors files found at {args.path}', file=sys.stderr)
        sys.exit(1)

    print(f'Loading {len(files)} safetensors file(s) from {args.path}')

    # Collect per-tensor stats
    stats = {}  # key -> {category, shape, zeros_pct, has_nan, has_inf, mean_abs}
    for fpath in files:
        with safe_open(fpath, framework='pt') as f:
            for key in f.keys():
                t = f.get_tensor(key)
                numel = t.numel()
                t_float = t.float()
                zeros_pct = (t == 0).sum().item() / numel * 100 if numel > 0 else 0
                has_nan = t_float.isnan().any().item()
                has_inf = t_float.isinf().any().item()
                mean_abs = t_float.abs().mean().item()
                stats[key] = {
                    'category': categorize(key),
                    'shape': tuple(t.shape),
                    'zeros_pct': zeros_pct,
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                    'mean_abs': mean_abs,
                }

    total_keys = len(stats)
    print(f'Total tensors: {total_keys}')
    print()

    # Aggregate per category
    categories = {}
    for key, s in stats.items():
        cat = s['category']
        if cat not in categories:
            categories[cat] = {
                'keys': [],
                'all_zero': [],
                'has_nan': [],
                'has_inf': [],
                'worst_zeros_pct': 0,
                'worst_zeros_key': None,
                'sample_shape': None,
                'sample_mean': 0,
            }
        c = categories[cat]
        c['keys'].append(key)
        if s['zeros_pct'] > 99.9:
            c['all_zero'].append(key)
        if s['has_nan']:
            c['has_nan'].append(key)
        if s['has_inf']:
            c['has_inf'].append(key)
        if s['zeros_pct'] > c['worst_zeros_pct']:
            c['worst_zeros_pct'] = s['zeros_pct']
            c['worst_zeros_key'] = key
        if c['sample_shape'] is None:
            c['sample_shape'] = s['shape']
            c['sample_mean'] = s['mean_abs']

    # Report
    problems = []
    for cat in CATEGORY_ORDER:
        if cat not in categories:
            continue
        c = categories[cat]
        n = len(c['keys'])
        n_zero = len(c['all_zero'])
        n_nan = len(c['has_nan'])
        n_inf = len(c['has_inf'])

        status_parts = []
        if n_zero > 0 and cat != 'Vision':
            status_parts.append(f'\033[91m{n_zero}/{n} ALL-ZERO\033[0m')
            problems.append((cat, 'all-zero', c['all_zero']))
        if n_nan > 0:
            status_parts.append(f'\033[91m{n_nan}/{n} NaN\033[0m')
            problems.append((cat, 'NaN', c['has_nan']))
        if n_inf > 0:
            status_parts.append(f'\033[91m{n_inf}/{n} Inf\033[0m')
            problems.append((cat, 'Inf', c['has_inf']))
        if not status_parts:
            if n_zero > 0 and cat == 'Vision':
                status_parts.append(f'{n_zero}/{n} zero (expected for unused vision bias)')
            status_parts.append('\033[92mOK\033[0m')

        status = '  '.join(status_parts)
        print(f'  {cat:<35} keys={n:>4}  shape={str(c["sample_shape"]):<25} mean_abs={c["sample_mean"]:.6f}  {status}')

    print()
    if not problems:
        print('\033[92m=== ALL CLEAR ===\033[0m')
        print('No all-zero (non-vision), NaN, or Inf weights detected.')
    else:
        print(f'\033[91m=== {len(problems)} PROBLEM(S) FOUND ===\033[0m')
        for cat, issue, keys in problems:
            print(f'\n  {cat} — {issue}:')
            show = keys if args.verbose else keys[:5]
            for k in show:
                s = stats[k]
                print(f'    {k}')
                print(f'      shape={s["shape"]}  zeros={s["zeros_pct"]:.1f}%  mean_abs={s["mean_abs"]:.8f}  nan={s["has_nan"]}  inf={s["has_inf"]}')
            if len(keys) > len(show):
                print(f'    ... and {len(keys) - len(show)} more (use --verbose to show all)')
        sys.exit(1)


if __name__ == '__main__':
    main()
