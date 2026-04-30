"""Layer-2 smoke test: load the LatentMoE checkpoint, run one forward, check aux_loss."""
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

CKPT = "output/gkd_rezaul_latentmoe_24l/v1-20260418-125431/checkpoint-900"

print("Loading model (trust_remote_code=True) ...")
model = AutoModelForCausalLM.from_pretrained(
    CKPT, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0"
)
model.eval()
tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)

print(f"Model class: {type(model).__name__}")
print(f"forward defined on: {type(model).forward.__qualname__}")

inputs = tok("Hello world, this is a test of the auxiliary loss path.", return_tensors="pt").to("cuda:0")
labels = inputs.input_ids.clone()

with torch.no_grad():
    out = model(**inputs, labels=labels)

print("\n" + "=" * 60)
print("Forward output diagnostics")
print("=" * 60)
print(f"loss            = {out.loss}")
print(f"aux_loss        = {getattr(out, 'aux_loss', 'NO_ATTR')}")
rl = getattr(out, "router_logits", None)
print(f"router_logits is None: {rl is None}")
if rl is not None:
    print(f"  num router_logits tensors = {len(rl)}  (expect 24)")
    print(f"  first tensor shape        = {rl[0].shape}  (expect (T, 256))")
    print(f"  first tensor dtype        = {rl[0].dtype}")

# Compare against a forward with output_router_logits=False to see loss difference
with torch.no_grad():
    out_off = model(**inputs, labels=labels, output_router_logits=False)
print(f"\nloss WITH    aux_loss = {out.loss.item():.6f}")
print(f"loss WITHOUT aux_loss = {out_off.loss.item():.6f}")
print(f"delta                  = {(out.loss - out_off.loss).item():.6f}")
coef = getattr(getattr(model.config, "text_config", model.config), "router_aux_loss_coef", float("nan"))
print(f"expected               = router_aux_loss_coef * aux_loss "
      f"= {coef} * {out.aux_loss.item() if out.aux_loss is not None else float('nan'):.4f} "
      f"= {coef * (out.aux_loss.item() if out.aux_loss is not None else 0):.6f}")
