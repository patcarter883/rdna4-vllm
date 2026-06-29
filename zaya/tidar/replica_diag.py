#!/usr/bin/env python3
"""Pin down the single-forward replica-draft gap: construction bug vs fundamental bonus-token gap.

single_forward_tidar.py found single-forward TiDAR is lossless but acceptance-limited: the replica
R_k drafts trail a fresh block_predict. Two hypotheses:
  (H1, FIXABLE) R_k's construction (positions/conditioning/selection) is wrong, so it doesn't even
       match block_predict for ITS OWN context (committed + k accepted drafts, no bonus).
  (H2, FUNDAMENTAL) R_k correctly drafts for the k-token context, but the committed stream advances by
       k+1 (the post-forward BONUS token), which R_k cannot see — so it trails the k+1-token draft.

Per step we compare argmax-equal:
  R_k                       (selected replica from the fused forward)
  bp_k  = block_predict(committed + drafts[:k])            # k-token context  -> R_k SHOULD match this
  bp_k1 = block_predict(committed + drafts[:k] + [bonus])  # k+1-token context -> what is actually needed
Verdict: R_k==bp_k (mostly) ⇒ H2 (bonus is the only gap; fix needs the bonus folded in / a mini-pass).
         R_k≠bp_k          ⇒ H1 (replica construction bug — fixable in tidar_mask/select/positions).

fp32, CPU, fork venv. Short (a few steps × 2 prompts).
"""
import argparse
import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya
from tidar_loop import beta_verify
from tidar_mask import MaskDescriptor, square_additive_bias, select_next_drafts_row_range

DT = torch.float32
DEV = "cpu"


def causal_bias_4d(L):
    idx = torch.arange(L)
    b = torch.zeros(L, L, dtype=DT)
    b.masked_fill_(~(idx[:, None] >= idx[None, :]), torch.finfo(DT).min)
    return b.view(1, 1, L, L)


def block_bias_4d(P, B):
    L = P + B
    idx = torch.arange(L)
    keep = (idx[:, None] >= idx[None, :]) | ((idx >= P)[:, None] & (idx >= P)[None, :])
    return torch.where(keep, torch.zeros((), dtype=DT),
                       torch.full((), torch.finfo(DT).min, dtype=DT)).view(1, 1, L, L)


def fwd(model, ids, pos, bias):
    zmp.set_bias(bias)
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids]), attention_mask=None,
                     position_ids=torch.tensor([pos]), use_cache=False).logits[0]


def block_predict(model, committed, B, mask_id):
    L = len(committed)
    lg = fwd(model, list(committed) + [mask_id] * B, list(range(L + B)), block_bias_4d(L, B))
    return lg[L:L + B].float().argmax(-1).tolist()


def fused_positions(L, B):
    # FIX: replica R_r predicts the block at positions [L+r, L+r+B-1] (matching block_predict after r
    # accepts), NOT all replicas at [L+B, L+2B-1]. The old fixed-position scheme was correct only for
    # r=B (all accepted); for r<B the RoPE positions were off by B-r → wrong-distribution drafts. (The
    # bisection validated [L+B..] only for the VERIFY S-rows, which don't depend on replica positions.)
    pos = list(range(L + B))
    for r in range(B):
        pos += list(range(L + r, L + r + B))
    return pos


def fused_step(model, committed, drafts, B, mask_id):
    L = len(committed)
    d = MaskDescriptor(prefix_len=L, block_len=B)
    seq = list(committed) + list(drafts) + [mask_id] * (B * B)
    lg = fwd(model, seq, fused_positions(L, B),
             square_additive_bias(d, dtype=DT, device=DEV).view(1, 1, d.kv_len, d.kv_len))
    p_ar = lg[L - 1:L + B].float()
    R = lg[L + B:].float().reshape(B, B, -1)
    return p_ar, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    zmp.install()
    print(f"[load] {args.ckpt} (fp32) …", flush=True)
    model, tok, mask_id, B, (miss, unexp) = load_tidar_zaya(args.ckpt, device="cpu", dtype=DT)
    model.eval()
    print(f"[load] B={B} miss={len(miss)} unexp={len(unexp)}", flush=True)

    n_h1 = n_h2 = n_tot = 0
    for prompt in ["The capital of France is", "In the beginning God created the"]:
        committed = tok(prompt, return_tensors=None)["input_ids"]
        drafts = block_predict(model, committed, B, mask_id)
        print(f"\n### {prompt!r}", flush=True)
        for s in range(args.steps):
            L = len(committed)
            p_ar, R = fused_step(model, committed, drafts, B, mask_id)
            k, bonus = beta_verify(drafts, p_ar, beta=1.0)
            d = MaskDescriptor(prefix_len=L, block_len=B)
            r0, _ = select_next_drafts_row_range(d, k)
            Rk = R[(r0 - B) // B].argmax(-1).tolist()
            bp_k = block_predict(model, list(committed) + drafts[:k], B, mask_id)
            bp_k1 = block_predict(model, list(committed) + drafts[:k] + [bonus], B, mask_id)
            m_k = (Rk == bp_k)
            m_k1 = (Rk == bp_k1)
            n_tot += 1
            n_h1 += int(m_k)        # R_k matches its own k-token context  -> construction OK (H2)
            n_h2 += int(m_k1)       # R_k matches the needed k+1-token draft -> no gap
            # token-by-token: if only the first total_padding(=2) tokens differ, the causal CONV
            # (sequence-order left-neighbors, ignores the attention mask) is the culprit (§1.1).
            tokmatch = [int(a == b) for a, b in zip(Rk, bp_k)]
            print(f"  step{s} k={k} bonus={bonus}: R_k==bp_k={m_k} R_k==bp_k1={m_k1} "
                  f"per-token(R_k vs bp_k)={tokmatch}  R_k={Rk} bp_k={bp_k}", flush=True)
            committed = list(committed) + drafts[:k] + [bonus]
            drafts = Rk
            if len(committed) > L + 60:
                break

    print(f"\n=== VERDICT ===", flush=True)
    print(f"  R_k == bp_k  (own k-token context) : {n_h1}/{n_tot}", flush=True)
    print(f"  R_k == bp_k1 (needed k+1 context)  : {n_h2}/{n_tot}", flush=True)
    if n_h1 >= 0.8 * n_tot and n_h2 < 0.5 * n_tot:
        print("  => H2 FUNDAMENTAL: replica construction is CORRECT for its k-token context; the gap is the\n"
              "     post-forward BONUS token (needs folding in / a mini-pass), not a fixable mask/pos bug.")
    elif n_h1 < 0.8 * n_tot:
        print("  => H1 FIXABLE: R_k does NOT even match its own k-token block_predict — a replica\n"
              "     construction bug (positions / conditioning / selection). Inspect tidar_mask + select.")
    else:
        print("  => MIXED: R_k mostly matches both — replicas may already be near-optimal; acceptance is\n"
              "     checkpoint-limited, not a construction gap.")


if __name__ == "__main__":
    main()
