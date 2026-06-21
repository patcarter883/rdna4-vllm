# SPDX-License-Identifier: Apache-2.0
#
# DECISIVE pre-PPL measurement for ParoQuant stage (c) — does activation
# conditioning (wider span and/or a LEARNED rotation) actually reduce the REAL
# per-token int8 ACTIVATION-quant error on a real model? Run BEFORE spending a
# quant+serve+PPL cycle, on a leased card:
#
#   scripts/gpu-lease.sh -n 1 -- docker run --rm ... analyze_act_conditioning.py
#
# It loads the base fp16 model, collects SIGNED input activations of every Linear
# (K % 256 == 0 so spans {32,64,128,256} all divide), and for each span reports
# the aggregate per-TOKEN int8 activation-quant MSE (full-row absmax scale — the
# exact runtime quant in rxf_kernels._rxf_rotate_quant_int8_kernel) under:
#   - no rotation,
#   - the fixed Hadamard-S,
#   - a LEARNED Givens-S fit to the activation objective (fit_givens_rotation).
# vs the shipped Hadamard-32 baseline. Tells us, on REAL data, whether the win is
# SPAN width (a fixed Hadamard) or LEARNING — before any PPL run.

import argparse
import ast
import json
import math
import os
import textwrap

import torch

_HERE = os.path.dirname(os.path.realpath(__file__))
_WANT = ["_fwht_rows", "_apply_rotation_rows", "_givens_quant_mse",
         "_rotated_importance", "_act_int8_quant_mse", "fit_givens_rotation"]


def _load_offline():
    src = open(os.path.join(_HERE, "quantize_rxf.py")).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "math": math, "GROUP": 32}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _WANT:
            exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
    return ns


def _hadamard(S):
    H = torch.tensor([[1.0]])
    while H.shape[0] < S:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(S)


def _rot_blockdiag(X, R):
    M, K = X.shape
    S = R.shape[0]
    return (X.reshape(M, K // S, S).float() @ R.float().t()).reshape(M, K)


def _row_int8_mse(X, R):
    """REAL per-token int8 quant MSE: rotate block-diagonally, per-row absmax/127,
    round to int8, dequant. X:[M,K] -> scalar (sum of sq err, count) for weighting."""
    Xr = _rot_blockdiag(X, R) if R is not None else X.float()
    absmax = Xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = absmax / 127.0
    deq = torch.round(Xr / scale).clamp_(-127, 127) * scale
    se = ((Xr - deq) ** 2).sum().item()
    return se, Xr.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True, help="calibration .jsonl")
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rows-per-mod", type=int, default=128,
                    help="signed activation rows kept per module")
    ap.add_argument("--spans", default="32,64,128,256")
    args = ap.parse_args()
    off = _load_offline()
    fit = off["fit_givens_rotation"]
    NL = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10,
                       1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.float32)
    spans = [int(s) for s in args.spans.split(",")]
    smax = max(spans)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True).eval()

    texts = []
    with open(args.calib) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= args.n_seq:
                break

    rows = {}            # name -> list of [t, K] cpu fp32 (signed)
    counts = {}

    def hook(name):
        def fn(m, inp, out):
            x = inp[0].detach()
            if x.shape[-1] % smax != 0:
                return
            if counts.get(name, 0) >= args.rows_per_mod:
                return
            r = x.reshape(-1, x.shape[-1]).float().cpu()
            need = args.rows_per_mod - counts.get(name, 0)
            if r.shape[0] > need:
                sel = torch.linspace(0, r.shape[0] - 1, need).long()
                r = r.index_select(0, sel)
            rows.setdefault(name, []).append(r)
            counts[name] = counts.get(name, 0) + r.shape[0]
        return fn

    hs = [m.register_forward_hook(hook(n))
          for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    print(f"hooked {len(hs)} Linear modules; running {len(texts)} seqs ...",
          flush=True)
    with torch.no_grad():
        for i, t in enumerate(texts):
            ids = tok.encode(t, add_special_tokens=False,
                             max_length=args.seq_len, truncation=True)
            if len(ids) < 16:
                continue
            model(torch.tensor([ids], device="cuda:0"))
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{len(texts)}", flush=True)
    for h in hs:
        h.remove()

    mods = {n: torch.cat(v, 0) for n, v in rows.items() if counts.get(n, 0)}
    mods = {n: r for n, r in mods.items() if r.shape[-1] % smax == 0}
    print(f"\ncollected activations for {len(mods)} modules "
          f"(K % {smax} == 0), {sum(r.shape[0] for r in mods.values())} rows total\n")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def agg_mse(R):
        se = nt = 0.0
        for r in mods.values():
            s, n = _row_int8_mse(r, R)
            se += s; nt += n
        return se / nt

    base = agg_mse(_hadamard(32))
    print(f"{'config':24s} {'per-tok int8 MSE':>18s} {'vs Had-32':>12s} "
          f"{'vs Had-S':>10s}")
    print(f"{'no-rotate':24s} {agg_mse(None):18.4e} "
          f"{base/agg_mse(None):11.3f}x {'-':>10s}")
    for S in spans:
        H = _hadamard(S)
        mse_h = agg_mse(H)
        # fit ONE shared R on pooled S-blocks (stratified across modules), CPU.
        pool = []
        per = max(1, 16384 // len(mods))
        for r in mods.values():
            b = r.reshape(-1, S)
            if b.shape[0] > per:
                sel = torch.linspace(0, b.shape[0] - 1, per).long()
                b = b.index_select(0, sel)
            pool.append(b)
        blocks = torch.cat(pool, 0).float()
        R = fit(blocks, NL, span=S, score="activation", seed=0)
        mse_f = agg_mse(R)
        print(f"{'Hadamard-'+str(S):24s} {mse_h:18.4e} "
              f"{base/mse_h:11.3f}x {'1.000':>10s}")
        print(f"{'fitted-'+str(S):24s} {mse_f:18.4e} "
              f"{base/mse_f:11.3f}x {mse_h/mse_f:9.3f}x")


if __name__ == "__main__":
    main()
