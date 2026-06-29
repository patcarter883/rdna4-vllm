#!/usr/bin/env python3
"""Bisect the §7.5 fused-forward contamination to its exact layer/op.

GATE A (coherence_gate.py) found that the FUSED single forward over
[committed | S(drafts) | R_0..R_{B-1}(mask*B each)] shifts the S verify rows vs a plain causal
forward over [committed | drafts] (max|Δlogit|≈1.8, flips argmax), even though S is masked from R in
attention. §7.5 attributed it to a sequence-GLOBAL op (MoE load-balance / global norm). But a code
audit says ZAYA MoE routing + ResidualScaling are strictly PER-TOKEN — which would mean S CANNOT be
contaminated by R's presence. This script resolves the contradiction by capturing the S-row hidden
state at EVERY layer (and at the attention-vs-MoE sub-step of the first diverging layer) in the fused
vs causal forward, and printing where S first diverges + by how much. That localizes the true cause:
  - diverges at the EMBED/conv/attention input  -> a mask/conv/position CONSTRUCTION bug (FIXABLE -> fused forward unlocks)
  - diverges only at MoE                          -> a real per-token-but-something MoE effect (audit missed it)
  - small, growing, no single jump               -> bf16 sequence-length numerical drift (not structural; fusion ~ok)

Run on CPU (no lease) via the fork venv — same container cmd as coherence_gate.py.
"""
import argparse
import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya
from tidar_mask import MaskDescriptor, square_additive_bias

DT = torch.bfloat16   # overridden by --dtype
DEV = "cpu"


def causal_bias_4d(L):
    idx = torch.arange(L)
    b = torch.zeros(L, L, dtype=DT)
    b.masked_fill_(~(idx[:, None] >= idx[None, :]), torch.finfo(DT).min)
    return b.view(1, 1, L, L)


def run_capture(model, ids, pos, bias_4d, s_slice):
    """Forward with the given bias; return {layer_idx: S-row hidden [B, H]} captured at each
    decoder layer output, plus the final logits S-rows."""
    caps = {}
    handles = []
    layers = model.model.layers
    for i, layer in enumerate(layers):
        def mk(i):
            def hook(mod, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                caps[i] = hs[0, s_slice].detach().float().clone()
            return hook
        handles.append(layer.register_forward_hook(mk(i)))
    zmp.set_bias(bias_4d)
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids]), attention_mask=None,
                  position_ids=torch.tensor([pos]), use_cache=False)
    for h in handles:
        h.remove()
    return caps, o.logits[0, s_slice].float()


def main():
    global DT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    DT = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    zmp.install()
    print(f"[load] {args.ckpt} on CPU ({args.dtype}, eager) …", flush=True)
    model, tok, mask_id, B, (miss, unexp) = load_tidar_zaya(args.ckpt, device="cpu", dtype=DT)
    model.eval()
    print(f"[load] mask_id={mask_id} B={B} missing={len(miss)} unexpected={len(unexp)}", flush=True)

    # A short committed prefix + B drafts (deterministic, like GATE A's first prompt tail).
    pid = tok("The capital of France is Paris and the capital of", return_tensors=None)["input_ids"]
    committed = pid
    L = len(committed)
    # Drafts: use the model's own AR picks so they're realistic (causal forward).
    zmp.set_bias(causal_bias_4d(L))
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([committed]), attention_mask=None,
                   position_ids=torch.tensor([list(range(L))]), use_cache=False).logits[0]
    drafts = [int(lg[-1].argmax())]
    for _ in range(B - 1):
        seq = committed + drafts
        zmp.set_bias(causal_bias_4d(len(seq)))
        with torch.no_grad():
            lg = model(input_ids=torch.tensor([seq]), attention_mask=None,
                       position_ids=torch.tensor([list(range(len(seq)))]), use_cache=False).logits[0]
        drafts.append(int(lg[-1].argmax()))
    print(f"[setup] L={L} B={B} drafts={drafts}", flush=True)

    s_slice = slice(L, L + B)  # the S (draft) verify rows, same indices in both forwards

    # CAUSAL reference: [committed | drafts]
    cseq = committed + drafts
    cpos = list(range(len(cseq)))
    caus_caps, caus_logits = run_capture(model, cseq, cpos, causal_bias_4d(len(cseq)), s_slice)

    # FUSED: [committed | drafts | R_0..R_{B-1}] with the square TiDAR mask + replica positions
    d = MaskDescriptor(prefix_len=L, block_len=B)
    fseq = list(committed) + list(drafts) + [mask_id] * (B * B)
    fpos = list(range(L + B))
    for _ in range(B):
        fpos += list(range(L + B, L + 2 * B))
    assert len(fseq) == d.kv_len == len(fpos)
    fbias = square_additive_bias(d, dtype=DT, device=DEV).view(1, 1, d.kv_len, d.kv_len)
    fused_caps, fused_logits = run_capture(model, fseq, fpos, fbias, s_slice)

    print("\n=== S-row hidden-state divergence per decoder layer (fused vs causal) ===", flush=True)
    print(f"{'layer':>5} {'max|Δ|':>12} {'mean|Δ|':>12} {'|hidden|~':>10}")
    first = None
    for i in sorted(caus_caps):
        d_ = (fused_caps[i] - caus_caps[i]).abs()
        mx, mn = d_.max().item(), d_.mean().item()
        scale = caus_caps[i].abs().mean().item()
        if first is None and mx > 1e-2:
            first = i
        flag = "  <-- FIRST DIVERGENCE" if i == first else ""
        if i < 4 or i == first or mx > 0.05 or i >= len(caus_caps) - 2:
            print(f"{i:>5} {mx:>12.4e} {mn:>12.4e} {scale:>10.3f}{flag}", flush=True)
    dlog = (fused_logits - caus_logits).abs()
    print(f"\nFINAL logits S-rows: max|Δ|={dlog.max().item():.4f}  "
          f"argmax_match={(fused_logits.argmax(-1)==caus_logits.argmax(-1)).all().item()}", flush=True)
    print(f"first-diverging layer = {first} (of {len(caus_caps)})", flush=True)
    print("\nInterpretation: layer 0 ⇒ embed/conv/attention construction (FIXABLE); a clean jump at one "
          "layer ⇒ that layer's op; slow growth from ~0 ⇒ bf16 seq-length drift (not structural).",
          flush=True)


if __name__ == "__main__":
    main()
