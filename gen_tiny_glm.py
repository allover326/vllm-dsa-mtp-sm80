"""Generate a tiny random-init glm_moe_dsa checkpoint (incl. MTP layer) for plumbing tests."""
import json, os, torch

OUT = "/models/tiny-glm-dsa"
os.makedirs(OUT, exist_ok=True)
cfg_dict = json.load(open("/work/tiny_glm_config.json"))
json.dump(cfg_dict, open(f"{OUT}/config.json", "w"), indent=2)

from huggingface_hub import snapshot_download
snapshot_download("zai-org/GLM-5.2", allow_patterns=["tokenizer*", "special_tokens_map.json", "generation_config.json"],
                  local_dir=OUT)

from transformers import AutoConfig, AutoModelForCausalLM
cfg = AutoConfig.from_pretrained(OUT)
torch.manual_seed(0)
model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
sd = {k: v.contiguous() for k, v in model.state_dict().items()}
print("raw tensors:", len(sd), flush=True)

H = cfg_dict["hidden_size"]
MI = cfg_dict["moe_intermediate_size"]
NE = cfg_dict["n_routed_experts"]
NL = cfg_dict["num_hidden_layers"]
g = torch.Generator().manual_seed(1)
def rnd(*shape, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=g) * 0.02).to(dtype)

# conform: fused HF expert tensors -> per-expert checkpoint-layout tensors
for L in range(NL):
    pre = f"model.layers.{L}.mlp.experts"
    fused = [k for k in sd if k.startswith(pre) and sd[k].dim() == 3]
    if not fused:
        continue
    for k in fused:
        print(f"replacing fused {k} shape={tuple(sd[k].shape)}", flush=True)
        del sd[k]
    for e in range(NE):
        sd[f"{pre}.{e}.gate_proj.weight"] = rnd(MI, H)
        sd[f"{pre}.{e}.up_proj.weight"] = rnd(MI, H)
        sd[f"{pre}.{e}.down_proj.weight"] = rnd(H, MI)
    bias_key = f"model.layers.{L}.mlp.gate.e_score_correction_bias"
    if bias_key not in sd:
        sd[bias_key] = torch.zeros(NE, dtype=torch.float32)
        print(f"added {bias_key}", flush=True)

# MTP layer NL: clone layer 2 (full indexer + sparse mlp) + MTP extras
mtp = {}
for k, v in list(sd.items()):
    if k.startswith("model.layers.2."):
        mtp[k.replace("model.layers.2.", f"model.layers.{NL}.")] = v.clone()
mtp[f"model.layers.{NL}.eh_proj.weight"] = rnd(H, 2 * H)
mtp[f"model.layers.{NL}.enorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
mtp[f"model.layers.{NL}.hnorm.weight"] = torch.ones(H, dtype=torch.bfloat16)
mtp[f"model.layers.{NL}.shared_head.norm.weight"] = torch.ones(H, dtype=torch.bfloat16)
sd.update(mtp)

# verify against real-checkpoint patterns
import re
manifest = json.load(open("/work/glm_key_manifest.json"))
def expand(patterns, L):
    out = set()
    for p in patterns:
        p = p.replace("{L}", str(L))
        if "{e}" in p:
            out |= {p.replace("{e}", str(e)) for e in range(NE)}
        else:
            out.add(p)
    return out
exp_mtp = expand(manifest["mtp"], NL)
got_mtp = {k for k in sd if k.startswith(f"model.layers.{NL}.")}
print("MTP MISSING:", sorted(exp_mtp - got_mtp), flush=True)
print("MTP EXTRA:", sorted(got_mtp - exp_mtp), flush=True)
# layer 2 must match the full-indexer sparse pattern = mtp pattern minus MTP extras
exp_l2 = {k.replace(f"layers.{NL}.", "layers.2.") for k in exp_mtp
          if not any(x in k for x in ("eh_proj", "enorm", "hnorm", "shared_head"))}
got_l2 = {k for k in sd if k.startswith("model.layers.2.")}
print("L2 MISSING:", sorted(exp_l2 - got_l2), flush=True)
print("L2 EXTRA:", sorted(got_l2 - exp_l2), flush=True)

from safetensors.torch import save_file
save_file(sd, f"{OUT}/model.safetensors", metadata={"format": "pt"})
total = sum(v.numel() * v.element_size() for v in sd.values())
print(f"SAVED ({total/1e9:.2f} GB, {len(sd)} tensors)", flush=True)
