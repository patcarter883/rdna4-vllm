#!/usr/bin/env python3
"""DIAGNOSTIC: is the TiDAR ZAYA forward causal w.r.t. APPENDED future tokens?

The β=1 gate diverged in a way only possible if appending tokens changes earlier
positions' logits. Recompute-the-full-sequence verification assumes that appending
tokens at positions >= L does NOT change the logits at position L-1. Test it directly,
and dump the arch knobs (MoD / MoE / sliding window / attn impl) that could break it.
"""
import os
import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya

CKPT = os.environ.get(
    "TIDAR_CKPT",
    "/root/.cache/huggingface/hub/models--pat883--zaya1-8b-tidar-experts/"
    "snapshots/e6f2ba2d904688059a9e4bd50531504554b02f6d",
)
DEVICE = "cpu"
DT = torch.bfloat16


def causal_logits(model, ids):
    L = len(ids)
    idx = torch.arange(L)
    allow = idx[:, None] >= idx[None, :]
    b = torch.zeros(L, L, dtype=DT)
    b.masked_fill_(~allow, torch.finfo(DT).min)
    zmp.set_bias(b.view(1, 1, L, L))
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), attention_mask=None,
                    position_ids=torch.arange(L).view(1, -1), use_cache=False)
    return out.logits[0]


def main():
    model, tok, mask_id, B, _ = load_tidar_zaya(CKPT, device=DEVICE, dtype=DT, from_config=True)
    zmp.install()

    cfg = model.config
    print("=== arch knobs ===")
    for k in ["model_type", "num_hidden_layers", "num_experts", "num_experts_per_tok",
              "n_routed_experts", "mod", "use_mod", "mod_capacity", "capacity_factor",
              "sliding_window", "swa_layers", "_attn_implementation", "router_aux_loss_coef"]:
        if hasattr(cfg, k):
            print(f"  {k} = {getattr(cfg, k)}")
    print("  attn_impl(runtime):", getattr(model.config, "_attn_implementation", "?"))

    prompt = tok("In the beginning was the", return_tensors="pt",
                 add_special_tokens=True)["input_ids"][0].tolist()
    L = len(prompt)
    base = causal_logits(model, prompt)
    print(f"\n=== causality test (L={L}) ===")
    # Append k arbitrary tokens; position L-1 must be unchanged if the model is causal.
    for app in [[5], [5, 6], [5, 6, 7, 8], [mask_id] * 4]:
        ext = causal_logits(model, prompt + app)
        d_last = (ext[L - 1].float() - base[L - 1].float()).abs().max().item()
        flip = int(ext[L - 1].float().argmax()) != int(base[L - 1].float().argmax())
        # also check an interior position (L//2)
        m = L // 2
        d_mid = (ext[m].float() - base[m].float()).abs().max().item()
        print(f"  append {app}: max|Δ| pos[L-1]={d_last:.4g} argmax-flip={flip} | pos[{m}]={d_mid:.4g}")

    # Determinism check: same input twice
    a = causal_logits(model, prompt)[L - 1].float()
    b = causal_logits(model, prompt)[L - 1].float()
    print(f"\n  determinism (same input twice) max|Δ|={ (a-b).abs().max().item():.4g}")


if __name__ == "__main__":
    main()
