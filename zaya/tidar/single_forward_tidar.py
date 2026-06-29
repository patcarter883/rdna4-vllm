#!/usr/bin/env python3
"""Capstone: the FUSED single-forward TiDAR loop — confirm it is lossless == AR-greedy and ~2.4×.

`bisect_fusion.py` proved the fused forward over [committed | S | R_0..R_{B-1}] gives fp32 bit-exact S
verify rows (§7.5 "contamination" = bf16 noise, not a global op) AND validated the replica positions
(§7.6). So production can do ONE forward per step that BOTH verifies the previous block's drafts (S
rows) AND pre-drafts the next block conditioned on every accept-length (replicas R_r). This script
runs that loop and checks:
  (1) LOSSLESS: the committed stream == plain AR-greedy (run in fp32 to dodge bf16 borderline flips).
  (2) DRAFT CORRECTNESS: the selected replica R_k == the two-forward `block_predict([committed+k], mask*B)`.
  (3) FORWARD COUNT: single-forward TiDAR uses 1 forward/step → (k+1) committed tokens ⇒ speedup =
      avg_accept+1 (vs the two-forward 0.896 fwd/tok ⇒ (avg+1)/2). Reports tok/s vs the ~27 baseline.

Run on CPU via the fork venv (same container cmd as coherence_gate.py / throughput_tidar.py).
"""
import argparse
import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya
from tidar_loop import beta_verify
from tidar_mask import MaskDescriptor, square_additive_bias, select_next_drafts_row_range

DT = torch.float32   # default fp32 for the strict losslessness check; --dtype bf16 for the realistic run
DEV = "cpu"
_NFWD = 0


def causal_bias_4d(L):
    idx = torch.arange(L)
    b = torch.zeros(L, L, dtype=DT)
    b.masked_fill_(~(idx[:, None] >= idx[None, :]), torch.finfo(DT).min)
    return b.view(1, 1, L, L)


def block_bias_4d(P, B):
    L = P + B
    idx = torch.arange(L)
    keep = (idx[:, None] >= idx[None, :]) | ((idx >= P)[:, None] & (idx >= P)[None, :])
    b = torch.where(keep, torch.zeros((), dtype=DT), torch.full((), torch.finfo(DT).min, dtype=DT))
    return b.view(1, 1, L, L)


def _fwd(model, ids, pos, bias):
    global _NFWD
    zmp.set_bias(bias)
    with torch.no_grad():
        o = model(input_ids=torch.tensor([ids]), attention_mask=None,
                  position_ids=torch.tensor([pos]), use_cache=False)
    _NFWD += 1
    return o.logits[0]


def causal_logits(model, ids):
    return _fwd(model, ids, list(range(len(ids))), causal_bias_4d(len(ids)))


def block_predict(model, committed, B, mask_id):
    L = len(committed)
    lg = _fwd(model, list(committed) + [mask_id] * B, list(range(L + B)), block_bias_4d(L, B))
    return lg[L:L + B].float().argmax(-1).tolist()


def ar_greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    for _ in range(n_new):
        ids.append(int(causal_logits(model, ids)[-1].float().argmax()))
    return ids[len(prompt_ids):]


def fused_positions(L, B):
    # Replica R_r predicts the block at positions [L+r, L+r+B-1] (matching block_predict after r
    # accepts). The naive all-at-[L+B,L+2B-1] scheme was only correct for r=B; for r<B it put the
    # replica's RoPE positions B-r too far → off-distribution drafts → low acceptance (replica_diag.py).
    pos = list(range(L + B))
    for r in range(B):
        pos += list(range(L + r, L + r + B))
    return pos


def fused_step(model, committed, drafts, B, mask_id):
    """ONE forward over [committed | drafts(S) | R_0..R_{B-1}]. Returns (p_ar [B+1,V], R_rows[B,B,V])."""
    L = len(committed)
    d = MaskDescriptor(prefix_len=L, block_len=B)
    seq = list(committed) + list(drafts) + [mask_id] * (B * B)
    bias = square_additive_bias(d, dtype=DT, device=DEV).view(1, 1, d.kv_len, d.kv_len)
    lg = _fwd(model, seq, fused_positions(L, B), bias)
    p_ar = lg[L - 1:L + B].float()                              # rows predict positions L..L+B
    R = lg[L + B:].float().reshape(B, B, -1)                    # [replica r, token m, V]
    return p_ar, R


def tidar_single_forward(model, prompt_ids, n_new, B, mask_id, check_drafts=False):
    """ONE fused forward per step: verify prev drafts (S) + read next drafts (R_k). Lossless β=1."""
    committed = list(prompt_ids)
    L0, target = len(committed), len(prompt_ids) + n_new
    drafts = block_predict(model, committed, B, mask_id)        # bootstrap the first block (1 fwd)
    accepts, draft_mismatch = [], 0
    while len(committed) < target:
        L = len(committed)
        p_ar, R = fused_step(model, committed, drafts, B, mask_id)
        k, bonus = beta_verify(drafts, p_ar, beta=1.0)
        committed = committed + drafts[:k] + [bonus]
        accepts.append(k)
        # next drafts = replica R_k (conditioned on k accepted), per select_next_drafts_row_range
        d = MaskDescriptor(prefix_len=L, block_len=B)
        r0, r1 = select_next_drafts_row_range(d, k)
        next_drafts = R[(r0 - B) // B].argmax(-1).tolist()      # R is [B,B,V]; replica index = (r0-B)/B
        if check_drafts:
            ref = block_predict(model, committed, B, mask_id)   # two-forward draft for the SAME context
            draft_mismatch += int(next_drafts != ref)
        drafts = next_drafts
    return committed[L0:target], accepts, draft_mismatch


def main():
    global DT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--check-drafts", action="store_true",
                    help="DIAGNOSTIC: add a block_predict forward/step to compare replica drafts "
                         "(inflates the forward count — OFF for a clean speedup number).")
    args = ap.parse_args()
    DT = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    zmp.install()
    print(f"[load] {args.ckpt} ({args.dtype}) …", flush=True)
    model, tok, mask_id, B, (miss, unexp) = load_tidar_zaya(args.ckpt, device="cpu", dtype=DT)
    model.eval()
    print(f"[load] mask_id={mask_id} B={B} missing={len(miss)} unexpected={len(unexp)}", flush=True)

    prompts = ["The capital of France is", "In the beginning God created the",
               "Q: What is 2+2? A:", "The mitochondria is the powerhouse of the"]
    global _NFWD
    tot_fwd = tot_tok = 0
    all_acc = []
    print("\n=== single-forward TiDAR β=1 vs AR-greedy ===", flush=True)
    for p in prompts:
        pid = tok(p, return_tensors=None)["input_ids"]
        ar = ar_greedy(model, pid, args.n_new)
        _NFWD = 0
        td, acc, dmm = tidar_single_forward(model, pid, args.n_new, B, mask_id,
                                            check_drafts=args.check_drafts)
        f = _NFWD
        ok = (ar == td)
        avg = sum(acc) / len(acc)
        all_acc += acc
        tot_fwd += f; tot_tok += args.n_new
        print(f"  [{'LOSSLESS' if ok else 'DIVERGED'}] {p[:32]!r:34s} fwd={f} "
              f"avg_accept={avg:.2f}/{B} draft_mismatch={dmm}/{len(acc)} fwd/tok={f/args.n_new:.3f}",
              flush=True)
    avg_accept = sum(all_acc) / len(all_acc)
    fpt = tot_fwd / tot_tok
    print(f"\n=== RESULT ({args.dtype}) ===")
    print(f"  avg accepted = {avg_accept:.2f}/{B}")
    print(f"  single-forward TiDAR fwd/token = {fpt:.3f}  (AR=1.000; two-forward was 0.896)")
    print(f"  speedup vs AR = {1/fpt:.2f}x   (≈ {27/fpt:.1f} tok/s vs ~27 baseline)")
    print(f"  predicted (avg_accept+1) = {avg_accept+1:.2f}x")


if __name__ == "__main__":
    main()
