"""CAM step 3c — LOCAL N-scaling sweep of the canonical-Z PKM store (bf16 value bank).

Pushes the store slot count N=n_sub^2 as high as fits on ONE 16 GB gfx1201 card in bf16, running the
already-validated 3b training+eval loop (train_mem_canonical) at each N for the DENSE arm, recording
held-out recall (memory / no_memory / ceiling, ΔNLL) and peak VRAM. The largest N that fits + trains
stable is the LOCAL CEILING; the recall-vs-N curve answers the real question — does the canonical-Z
store stay addressable at knowledge-store scale, or degrade?

This is a thin orchestration over train_mem_canonical's pieces (no new memory mechanics): per N it
builds a fresh CanonicalMemoryFrontEnd (bf16 bank), trains under a per-N time budget, evals held-out,
prints a row. OOM at an N => that N is the CEILING; we DO NOT retry (operating rule), we record and
stop the sweep. s24 spot-check at the largest fitting N confirms 2:4 holds at scale.

Run (1 leased card):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-3c -- \
    titans/warmstart/run_m2.sh titans-3c --entry warmstart/sweep_n_3c.py -- \
      --n-subs 32,64,128,180,224,256,320 --per-n-budget 420 --batch 8 --eval-batch 4 --M 8
"""
import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "deep_mem"))

from m2_adapter import MODEL, DEV, load_frozen_base                       # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
import train_mem_canonical as T                                          # noqa: E402


def run_one(base, base_embed, builder, args, n_sub, arm, Z, d_base, d_hub, L, budget):
    """Build a fresh front-end at N=n_sub^2 (bf16 bank), smoke 1 step, train under `budget` seconds,
    eval held-out. Returns a result dict, or {'oom': True} if an OOM is caught (=> the ceiling)."""
    bank_dtype = torch.bfloat16 if args.bank_dtype == "bf16" else torch.float32
    N = n_sub * n_sub
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    try:
        fe = T.CanonicalMemoryFrontEnd(
            d_base, d_hub, Z, arm=arm, n_sub=n_sub, topk=args.topk, sub_topk=args.sub_topk,
            n_heads=args.read_heads, tap_heads=args.tap_heads, k_inject=args.k_inject,
            bank_dtype=bank_dtype).to(DEV)
        n_params = sum(p.numel() for p in fe.parameters())

        # smoke ONE step at this N before committing the budget (operating rule)
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        handle, _ = T.attach_tap(base, fe, L)
        logits, head_norms, addr0 = T.run_step(base, base_embed, fe, builder, ids, apos,
                                                addr_loss=(args.addr_weight > 0))
        lm = F.cross_entropy(logits, ans)
        loss0 = lm.item()
        (lm + (args.addr_weight * addr0 if addr0 is not None else 0.0)).backward()
        handle.remove()
        fe.zero_grad(set_to_none=True)
        smoke_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[3c][{arm}][N={N}] SMOKE step0 loss {loss0:.3f} "
              f"addr {addr0.item():.3f} trainable {n_params/1e6:.1f}M "
              f"bank/episode(train) {args.batch*N*d_hub*(2 if args.bank_dtype=='bf16' else 4)/1e9:.2f}GB "
              f"smoke_peak {smoke_peak:.2f}GB", flush=True)

        # train a FIXED step count (so recall-vs-N is comparable across N — same task difficulty,
        # same training budget; the only variable is N) with a SAFETY time cap so a slow large-N
        # point can't run away. Reuse train_arm.
        targs = argparse.Namespace(**vars(args))
        targs.steps = args.steps
        targs.time_budget = budget          # safety cap, not the primary stop
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        n_steps, sec_step, conv_step, win_acc = T.train_arm(
            base, base_embed, fe, builder, rng, targs, arm, L, time_budget=budget)

        gen = T.evaluate(base, base_embed, fe, builder, rng, targs, n=args.eval_n, L=L)
        m_acc, nm_acc, ceil = gen["memory"][1], gen["no_memory"][1], gen["ceiling"][1]
        m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
        passed = (m_acc > nm_acc + 0.15 and m_acc > 0.5)
        peak = torch.cuda.max_memory_allocated() / 1e9
        res = dict(n_sub=n_sub, N=N, arm=arm, m_acc=m_acc, nm_acc=nm_acc, ceil=ceil,
                   dnll=nm_nll - m_nll, steps=n_steps, sec_step=sec_step, conv=conv_step,
                   win_acc=win_acc, peak=peak, passed=passed, oom=False)
        print(f"[3c][{arm}][N={N}] memory {m_acc:.3f} / no_memory {nm_acc:.3f} / ceiling {ceil:.3f} | "
              f"ΔNLL {nm_nll - m_nll:+.2f} | {n_steps} steps @ {sec_step:.3f}s | conv@{conv_step} | "
              f"win_acc {win_acc:.3f} | peak {peak:.2f}GB | {'PASS' if passed else 'FAIL'}", flush=True)
        del fe, gen
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        return res
    except torch.cuda.OutOfMemoryError as e:
        print(f"[3c][{arm}][N={N}] *** OOM (CEILING) *** {str(e)[:120]}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return dict(n_sub=n_sub, N=N, arm=arm, oom=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="warmstart/ckpt/atlas/canonical_z_v1_local6.pt")
    ap.add_argument("--n-subs", default="32,64,128,224,320", dest="n_subs")
    ap.add_argument("--steps", type=int, default=2500, help="fixed steps/N (comparable recall-vs-N)")
    ap.add_argument("--per-n-budget", type=float, default=900.0, dest="per_n_budget",
                    help="SAFETY time cap per N (s); fixed --steps is the primary stop")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=4, dest="eval_batch")
    ap.add_argument("--bank-dtype", default="bf16", choices=["fp32", "bf16"], dest="bank_dtype")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--sub-topk", type=int, default=4, dest="sub_topk")
    ap.add_argument("--read-heads", type=int, default=3, dest="read_heads")
    ap.add_argument("--tap-heads", type=int, default=8, dest="tap_heads")
    ap.add_argument("--k-inject", type=int, default=8, dest="k_inject")
    ap.add_argument("--tap-layer", type=int, default=24, dest="tap_layer")
    ap.add_argument("--addr-weight", type=float, default=1.0, dest="addr_weight")
    ap.add_argument("--conv-acc", type=float, default=0.90, dest="conv_acc")
    ap.add_argument("--eval-n", type=int, default=512, dest="eval_n")
    ap.add_argument("--log-every", type=int, default=200, dest="log_every")
    ap.add_argument("--s24-at-ceiling", action="store_true", dest="s24_ceiling",
                    help="after the dense sweep, spot-check the s24 arm at the largest fitting N")
    args = ap.parse_args()

    atlas_path = args.atlas if os.path.isabs(args.atlas) else os.path.join(
        os.path.dirname(_HERE), args.atlas)
    atlas = torch.load(atlas_path, map_location="cpu", weights_only=False)
    Z, d_hub = atlas["Z"], atlas["d_hub"]
    print(f"[3c] atlas Z {tuple(Z.shape)} d_hub={d_hub} sha {atlas['anchor_sha'][:12]} | "
          f"bank-dtype={args.bank_dtype} | batch {args.batch} eval-batch {args.eval_batch} | "
          f"M={args.M} budget {args.per_n_budget}s/N | n_subs {args.n_subs}", flush=True)

    torch.manual_seed(args.seed)
    base, tok = load_frozen_base()
    d_base = base.config.get_text_config().hidden_size
    n_layers = base.config.get_text_config().num_hidden_layers
    base_embed = base.get_input_embeddings()
    L = min(args.tap_layer, n_layers - 1)
    base_only = torch.cuda.max_memory_allocated() / 1e9
    print(f"[3c] {MODEL} d_base={d_base} n_layers={n_layers} tap L={L} | base-only VRAM {base_only:.2f}GB",
          flush=True)

    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")

    n_subs = [int(x) for x in args.n_subs.split(",") if x]
    rows, ceiling_n = [], None
    for n_sub in n_subs:
        r = run_one(base, base_embed, builder, args, n_sub, "dense", Z, d_base, d_hub, L,
                    args.per_n_budget)
        rows.append(r)
        if r.get("oom"):
            print(f"[3c] N={n_sub*n_sub} OOMed -> CEILING reached; stopping sweep (no retry).", flush=True)
            break
        ceiling_n = n_sub  # last fitting N

    # s24 spot-check at the largest fitting N (2:4 holds at scale?)
    s24_row = None
    if args.s24_ceiling and ceiling_n is not None:
        print(f"[3c] s24 spot-check at ceiling N={ceiling_n*ceiling_n}", flush=True)
        s24_row = run_one(base, base_embed, builder, args, ceiling_n, "s24", Z, d_base, d_hub, L,
                          args.per_n_budget)

    print("\n[3c] === N-SCALING SWEEP (canonical-Z PKM, bf16 value bank, DENSE) ===", flush=True)
    print(f"{'n_sub':>6} {'N':>8} {'memory':>7} {'no_mem':>7} {'ceiling':>8} {'ΔNLL':>8} "
          f"{'steps':>6} {'s/step':>7} {'conv@':>6} {'peakGB':>7} {'verdict':>7}", flush=True)
    for r in rows:
        if r.get("oom"):
            print(f"{r['n_sub']:>6} {r['N']:>8} {'OOM — CEILING (no retry)':>40}", flush=True)
            continue
        print(f"{r['n_sub']:>6} {r['N']:>8} {r['m_acc']:7.3f} {r['nm_acc']:7.3f} {r['ceil']:8.3f} "
              f"{r['dnll']:8.2f} {r['steps']:6d} {r['sec_step']:7.3f} {str(r['conv']):>6} "
              f"{r['peak']:7.2f} {'PASS' if r['passed'] else 'FAIL':>7}", flush=True)
    if s24_row and not s24_row.get("oom"):
        r = s24_row
        print(f"{'s24':>6} {r['N']:>8} {r['m_acc']:7.3f} {r['nm_acc']:7.3f} {r['ceil']:8.3f} "
              f"{r['dnll']:8.2f} {r['steps']:6d} {r['sec_step']:7.3f} {str(r['conv']):>6} "
              f"{r['peak']:7.2f} {'PASS' if r['passed'] else 'FAIL':>7}", flush=True)
    fit = [r for r in rows if not r.get("oom")]
    if fit:
        top = fit[-1]
        print(f"[3c] LOCAL CEILING (dense): N={top['N']} (n_sub={top['n_sub']}) "
              f"memory {top['m_acc']:.3f} peak {top['peak']:.2f}GB | "
              f"recall-vs-N: {' '.join(f'{r['N']}:{r['m_acc']:.2f}' for r in fit)}", flush=True)
    print("[3c] PASS rule: memory > no_memory+0.15 and > 0.5. bf16 bank, fp32-compute reads.", flush=True)


if __name__ == "__main__":
    main()
