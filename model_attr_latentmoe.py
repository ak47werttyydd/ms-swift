"""Diagnostic script that inspects /home/s00919127/latent-moe/qwen35_latentmoe
and surfaces the EXACT missing/unexpected key warnings that
`AutoModelForCausalLM.from_pretrained(...)` would emit.

Unlike `model_attr.py`, this script does NOT load weights into CPU memory
(that would need ~37 GB RAM). It instantiates the model on `meta` device so
it only consumes the structural params/buffers, then compares state_dict
keys against the safetensors file directly — producing the same
missing_keys / unexpected_keys lists that from_pretrained computes
internally before emitting its warnings.
"""
import sys
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_DIR = '/home/s00919127/latent-moe/qwen35_latentmoe'
# The checkpoint dir has its own molae_qwen35_latentmoe.py next to config.json,
# loaded via auto_map + trust_remote_code=True. We don't need to import it ourselves.
sys.path.insert(0, MODEL_DIR)

# ─── Step 1: load config + instantiate on meta (no weight RAM) ──────────────
cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
print('>>> Config class :', type(cfg).__name__)
print('    model_type   :', cfg.model_type)
print('    architectures:', cfg.architectures)
print('    auto_map     :', getattr(cfg, 'auto_map', None))
print()

with torch.device('meta'):
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

print('>>> Model class name:', m.__class__.__name__)
print()

# ─── Step 2: inspect structure (like model_attr.py) ─────────────────────────
layer0 = m.model.layers[0]
print('Layer0 type:', type(layer0).__name__)
print('Layer0.mlp type:', type(layer0.mlp).__name__)
if hasattr(layer0.mlp, 'experts'):
    experts = layer0.mlp.experts
    print('Layer0.mlp.experts type:', type(experts).__name__)
    for cand in ('gate', 'gate_proj', 'up_proj', 'down_proj',
                 'gate_up_proj', 'latent_down_proj', 'latent_up_proj'):
        if hasattr(experts, cand):
            obj = getattr(experts, cand)
            if isinstance(obj, torch.Tensor):
                print(f'    .{cand}: Tensor shape={tuple(obj.shape)}')
            else:
                print(f'    .{cand}: {type(obj).__name__}')
print()

# ─── Step 3: compute missing/unexpected like from_pretrained does ───────────
# This is the same set math transformers.modeling_utils.PreTrainedModel
# ._load_pretrained_model performs to build its warning text.
model_keys = set(m.state_dict().keys())
with safe_open(f'{MODEL_DIR}/model.safetensors', framework='pt') as f:
    ckpt_keys = set(f.keys())

matched = model_keys & ckpt_keys
missing = model_keys - ckpt_keys       # random-initialized parameters
unexpected = ckpt_keys - model_keys    # discarded weights from the file

print(f'===== Key comparison =====')
print(f'  Model  state_dict keys: {len(model_keys):>6}')
print(f'  Ckpt   tensor    keys: {len(ckpt_keys):>6}')
print(f'  Matched (loaded)       : {len(matched):>6}')
print(f'  Missing (random-init)  : {len(missing):>6}')
print(f'  Unexpected (discarded) : {len(unexpected):>6}')
print()

# ─── Step 4: print the exact warnings from_pretrained WOULD emit ────────────
HORIZONTAL_RULE = '─' * 78
print(HORIZONTAL_RULE)
print('WARNINGS from_pretrained WOULD emit (suppressed/missed in normal runs):')
print(HORIZONTAL_RULE)

if unexpected:
    print()
    print(f'Some weights of the model checkpoint at {MODEL_DIR}')
    print(f'were not used when initializing {m.__class__.__name__}:')
    for k in sorted(unexpected)[:12]:
        print(f'  - {k}')
    if len(unexpected) > 12:
        print(f'  ... and {len(unexpected) - 12} more')

if missing:
    print()
    print(f'Some weights of {m.__class__.__name__} were not initialized from the model checkpoint file')
    print(f'and are newly initialized:')
    for k in sorted(missing)[:12]:
        print(f'  - {k}')
    if len(missing) > 12:
        print(f'  ... and {len(missing) - 12} more')
    print('You should probably TRAIN this model on a down-stream task to use it for predictions.')

print()
print(HORIZONTAL_RULE)

# ─── Step 5: categorize the damage ──────────────────────────────────────────
import re

def classify(keys, pattern_map):
    result = {label: [] for label in pattern_map}
    result['(other)'] = []
    for k in keys:
        for label, pat in pattern_map.items():
            if re.search(pat, k):
                result[label].append(k)
                break
        else:
            result['(other)'].append(k)
    return result

missing_cats = classify(missing, {
    'MoE experts (packed tensors)': r'\.mlp\.experts\.(gate_up_proj|down_proj)$',
    'latent projections':           r'\.mlp\.latent_(down|up)_proj\.weight$',
    'shared expert / gate':         r'\.mlp\.(shared_expert|shared_expert_gate|gate)\b',
    'attention (linear/full)':      r'\.(self_attn|linear_attn)\.',
    'layer norms':                  r'layernorm|\.norm\.',
    'embeddings / lm_head':         r'embed_tokens|lm_head',
})
unexpected_cats = classify(unexpected, {
    'MoE experts (unpacked per-expert)': r'\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$',
    'attention w/ language_model prefix': r'model\.language_model\..*\.(self_attn|linear_attn)',
    'embed/norm w/ language_model prefix': r'model\.language_model\.(embed_tokens|norm)',
})

print()
print('===== Missing keys by category =====')
for label, keys in missing_cats.items():
    if keys:
        print(f'  {label:<40} {len(keys):>6}')

print()
print('===== Unexpected keys by category =====')
for label, keys in unexpected_cats.items():
    if keys:
        print(f'  {label:<40} {len(keys):>6}')

print()
print(HORIZONTAL_RULE)
print(f'TOTAL trainable params in model:     {sum(p.numel() for p in m.parameters()):,}')
print('(Note: on meta device these are structural counts, identical to what')
print(' a real from_pretrained would instantiate.)')
