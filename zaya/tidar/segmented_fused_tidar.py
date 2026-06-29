#!/usr/bin/env python3
"""Realize the SEGMENTED-conv fused forward (no model patch) and validate it recovers ~2.4×.

replica_diag.py root-caused the single-forward draft loss to the §1.1 causal conv: in the flat fused
sequence each replica R_r's leading total_padding(=2) tokens read the S DRAFT tokens through conv_qk
(which ignores the attention mask). The serving fix is a SEGMENTED conv (cca.py:_decode_verify_spec).
Here we realize the same effect on the HF dense forward WITHOUT patching the model, by a construction
trick: insert R_r's correct 2-token conv context `ctx_r` (= last 2 tokens of committed+first-r-drafts)
immediately before R_r in the sequence, MASKED OUT of attention so it only feeds the causal conv. Since
conv_qk runs on the per-token q/k projections (modeling_zaya.py:222-245), ctx_r supplies R_r's conv the
right neighbors — exactly what bp_r=block_predict(committed+drafts[:r]) sees.

Layout (B=block_len, L=len(committed)):
  [ committed(L) | S(B) | (ctx_0[2] R_0[B]) (ctx_1[2] R_1[B]) ... (ctx_{B-1}[2] R_{B-1}[B]) ]
Mask: committed causal; S[i]->committed+S[<=i]; ctx_r->self only (unused); R_r[m]->committed + S[<r] +
own R_r block (bidir). Positions: committed/S contiguous; ctx_r at [L+r-2,L+r-1]; R_r at [L+r,L+r+B-1].

Checks: (1) R_k == bp_k token-exact now; (2) lossless == AR-greedy (fp32); (3) acceptance recovers ->
speedup ~ (avg_accept+1) at 1 forward/step. fp32 CPU, fork venv. Requires L>=2.
"""
import argparse
import torch

import zaya_mask_patch as zmp
from serve_loader import load_tidar_zaya
from tidar_loop import beta_verify

DT = torch.float32
DEV = "cpu"
TP = 2          # conv left-context tokens per replica (total_padding)
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
    return torch.where(keep, torch.zeros((), dtype=DT),
                       torch.full((), torch.finfo(DT).min, dtype=DT)).view(1, 1, L, L)


def fwd(model, ids, pos, bias):
    global _NFWD
    zmp.set_bias(bias)
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids]), attention_mask=None,
                   position_ids=torch.tensor([pos]), use_cache=False).logits[0]
    _NFWD += 1
    return lg


def causal_logits(model, ids):
    return fwd(model, ids, list(range(len(ids))), causal_bias_4d(len(ids)))


def block_predict(model, committed, B, mask_id):
    L = len(committed)
    lg = fwd(model, list(committed) + [mask_id] * B, list(range(L + B)), block_bias_4d(L, B))
    return lg[L:L + B].float().argmax(-1).tolist()


def ar_greedy(model, prompt_ids, n_new):
    ids = list(prompt_ids)
    for _ in range(n_new):
        ids.append(int(causal_logits(model, ids)[-1].float().argmax()))
    return ids[len(prompt_ids):]


def build_segmented(committed, drafts, B, mask_id):
    """Return (ids, positions, allow[bool seq,seq], R_row_index[r]->list of B seq positions)."""
    L = len(committed)
    cd = list(committed) + list(drafts)            # length L+B
    ids = list(cd)                                  # committed | S
    pos = list(range(L + B))
    seg_S = list(range(L, L + B))                   # S seq indices
    R_rows = {}
    ctx_idx = {}
    for r in range(B):
        # ctx_r = last TP tokens of committed+first-r-drafts (indices L+r-TP .. L+r-1 of cd)
        base = len(ids)
        for t in range(TP):
            src = L + r - TP + t                     # index into cd
            ids.append(cd[src]); pos.append(src)
        ctx_idx[r] = list(range(base, base + TP))
        rbase = len(ids)
        for m in range(B):
            ids.append(mask_id); pos.append(L + r + m)
        R_rows[r] = list(range(rbase, rbase + B))
    n = len(ids)
    allow = torch.zeros(n, n, dtype=torch.bool)
    # committed causal
    for i in range(L):
        allow[i, :i + 1] = True
    # S[i] -> committed(all) + S[<=i]
    for i in range(B):
        qi = L + i
        allow[qi, :L] = True
        for j in range(i + 1):
            allow[qi, L + j] = True
    # ctx_r -> self only (output unused; avoid all-masked NaN)
    for r in range(B):
        for c in ctx_idx[r]:
            allow[c, c] = True
    # R_r[m] -> committed(all) + first r drafts of S + own R_r block (bidir)
    for r in range(B):
        for m, qi in enumerate(R_rows[r]):
            allow[qi, :L] = True                     # all committed
            for j in range(r):                       # first r drafts
                allow[qi, L + j] = True
            for c in R_rows[r]:                       # own block bidirectional
                allow[qi, c] = True
    return ids, pos, allow, R_rows


def fused_step(model, committed, drafts, B, mask_id):
    L = len(committed)
    ids, pos, allow, R_rows = build_segmented(committed, drafts, B, mask_id)
    bias = torch.where(allow, torch.zeros((), dtype=DT),
                       torch.full((), torch.finfo(DT).min, dtype=DT)).view(1, 1, len(ids), len(ids))
    lg = fwd(model, ids, pos, bias)
    p_ar = lg[L - 1:L + B].float()                   # verify rows predict positions L..L+B
    R = {r: lg[R_rows[r]].float().argmax(-1).tolist() for r in range(B)}
    return p_ar, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/ckpt")
    ap.add_argument("--n-new", type=int, default=24)
    ap.add_argument("--verify-drafts", action="store_true")
    args = ap.parse_args()
    zmp.install()
    print(f"[load] {args.ckpt} (fp32) …", flush=True)
    model, tok, mask_id, B, (miss, unexp) = load_tidar_zaya(args.ckpt, device="cpu", dtype=DT)
    model.eval()
    print(f"[load] B={B} miss={len(miss)} unexp={len(unexp)}", flush=True)

    prompts = ["The capital of France is", "In the beginning God created the",
               "Q: What is 2+2? A:", "The mitochondria is the powerhouse of the"]
    global _NFWD
    tot_fwd = tot_tok = 0
    all_acc = []
    rk_match = rk_tot = 0
    print("\n=== SEGMENTED-conv single-forward TiDAR ===", flush=True)
    for p in prompts:
        pid = tok(p, return_tensors=None)["input_ids"]
        ar = ar_greedy(model, pid, args.n_new)
        _NFWD = 0
        committed = list(pid)
        drafts = block_predict(model, committed, B, mask_id)
        acc = []
        while len(committed) < len(pid) + args.n_new:
            L = len(committed)
            p_ar, R = fused_step(model, committed, drafts, B, mask_id)
            k, bonus = beta_verify(drafts, p_ar, beta=1.0)
            Rk = R[k if k < B else B - 1]
            if args.verify_drafts:
                bp_k = block_predict(model, list(committed) + drafts[:k], B, mask_id)
                rk_match += int(Rk == bp_k); rk_tot += 1
                if rk_tot <= 8:
                    tm = [int(a == b) for a, b in zip(Rk, bp_k)]
                    print(f"      k={k} per-token(R_k vs bp_k)={tm}  R_k={Rk} bp_k={bp_k}", flush=True)
            committed = list(committed) + drafts[:k] + [bonus]
            acc.append(k); drafts = Rk
        f = _NFWD
        td = committed[len(pid):len(pid) + args.n_new]
        ok = (ar == td)
        avg = sum(acc) / len(acc)
        all_acc += acc
        tot_fwd += f; tot_tok += args.n_new
        print(f"  [{'LOSSLESS' if ok else 'DIVERGED'}] {p[:30]!r:32s} fwd={f} avg_accept={avg:.2f}/{B} "
              f"fwd/tok={f/args.n_new:.3f}", flush=True)
    avg_accept = sum(all_acc) / len(all_acc)
    fpt = tot_fwd / tot_tok
    print(f"\n=== RESULT (segmented conv, fp32) ===")
    print(f"  avg accepted = {avg_accept:.2f}/{B}  (flat-conv single-forward was ~0.50-0.75)")
    if rk_tot:
        print(f"  R_k == bp_k (drafts now correct?) : {rk_match}/{rk_tot}")
    print(f"  fwd/token = {fpt:.3f}  -> speedup {1/fpt:.2f}x  (≈ {27/fpt:.1f} tok/s; two-forward=1.12x, "
          f"flat single-forward=1.52x)")
    print(f"  predicted (avg_accept+1) = {avg_accept+1:.2f}x")


if __name__ == "__main__":
    main()
