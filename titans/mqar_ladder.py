"""MQAR expressiveness ladder — 5 arms, one variable each (the memory cell only).

  1 ablated     : no memory (segment-local attention floor)
  2 gated-delta : DecoupledGateCell(channelwise=False) -- scalar gates (KDA/GatedDeltaNet limit)
  3 gdn2        : DecoupledGateCell(channelwise=True)  -- decoupled channel-wise erase/write
  4 linear      : NeuralMemory + MemoryMLP(depth=1)    -- linear Titans memory (surprise inner-loop)
  5 deep        : NeuralMemory + MemoryMLP(depth=2)    -- deep Titans memory (the arm that passed)

Isolates: (a) does a better LINEAR cell help (gdn2 vs gated-delta), (b) does DEEP beat the best
linear cell (deep vs max(gated-delta, gdn2, linear)). Same backbone / task / ablation throughout.

Training uses variable load N~U[nmin,nmax]; eval sweeps fixed N (the multi-key / MK-NIAH
interference slice) and buckets by definition->query distance (cross-segment >32 = memory-only).
Reports per-arm param count AND recurrent-state size so a win isn't just "more memory".

Guardrails honored: one variable per arm; inner-loop loss stays L2 (Titans default; no Miras
knobs); no CCA; no deep-mem chunked kernel (DecoupledGateCell is a correctness-first sequential
scan). Kernel cost is reported, not built.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch

from titans_common import materialize_overlapping_params, dedup_params
from mem_cells import DecoupledGateCell, swap_memory_cells
from mqar_probe import gen_batch, loss_and_pred, SPECIALS

ARMS = ["ablated", "gated-delta", "gdn2", "linear", "deep"]


def build(arm, num_tokens, args, device):
    from titans_pytorch import MemoryAsContextTransformer, MemoryMLP
    mem_layers = [] if arm == "ablated" else list(args.mem_layers)
    has_nm = arm in ("linear", "deep")          # arms that keep titans NeuralMemory
    uses_cell = arm in ("gated-delta", "gdn2")  # arms that swap in DecoupledGateCell
    depth1 = arm == "linear"
    # placeholder MemoryMLP for cell arms (will be swapped out); real depth for linear/deep
    mm_depth = 1 if depth1 else 2
    nm_model = MemoryMLP(dim=args.mem_dim_head, depth=mm_depth) if mem_layers else None

    model = MemoryAsContextTransformer(
        num_tokens=num_tokens, dim=args.dim, depth=args.depth,
        segment_len=args.segment_len, num_persist_mem_tokens=4, num_longterm_mem_tokens=4,
        neural_memory_layers=tuple(mem_layers) if mem_layers else (args.depth + 10,),
        neural_memory_segment_len=args.nm_seg, neural_memory_batch_size=args.nm_batch,
        neural_mem_weight_residual=False,            # off for all -> uniform, and cells need no .updates
        neural_memory_qkv_receives_diff_views=False, # 3 identical views -> cells project their own qkv
        use_flex_attn=False, sliding_window_attn=False,
        neural_memory_model=nm_model,
        neural_memory_kwargs=dict(
            dim_head=args.mem_dim_head, heads=args.mem_heads, attn_pool_chunks=True, qk_rmsnorm=True,
            momentum=True, momentum_order=1, default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False, per_parameter_lr_modulation=True,
            per_head_learned_parameters=True,
        ) if mem_layers else dict(),
    )

    if uses_cell:
        ch = arm == "gdn2"
        n = swap_memory_cells(model, lambda: DecoupledGateCell(
            dim=args.dim, heads=args.mem_heads, head_dim=args.mem_dim_head, channelwise=ch))
        assert n == len(mem_layers), f"swapped {n} expected {len(mem_layers)}"

    n_nm = sum(1 for m in model.modules() if type(m).__name__ == "NeuralMemory")
    n_cell = sum(1 for m in model.modules() if type(m).__name__ == "DecoupledGateCell")
    assert (n_nm == len(mem_layers)) if has_nm else (n_nm == 0)
    assert (n_cell == len(mem_layers)) if uses_cell else (n_cell == 0)

    if has_nm:
        materialize_overlapping_params(model)
    model = model.to(device)

    # recurrent state size per arm (floats carried across the sequence), for capacity context
    dh, hh = args.mem_dim_head, args.mem_heads
    if arm == "ablated":
        state = 0
    elif uses_cell:
        state = hh * dh * dh                          # S in R^{d_k x d_v} per head
    elif depth1:
        state = hh * dh * dh                          # linear memory weight per head
    else:
        exp = int(dh * 2.0)
        state = hh * (dh * exp + exp * dh)            # 2-layer MemoryMLP per head
    return model, state


def evaluate(model, rng, args, device, n_per=None):
    n_per = n_per or args.eval_samples
    model.eval()
    buckets = [(1, 16), (17, 32), (33, 64), (65, 96), (97, 10 ** 9)]
    out = {}
    with torch.no_grad():
        for N in args.eval_pairs:
            hit = {b: [0, 0] for b in buckets}; oh = ot = 0
            done = 0
            while done < n_per:
                B = min(args.batch, n_per - done)
                seq, ans_pos, dist = gen_batch(rng, B, N, args.vocab, device)
                _, correct = loss_and_pred(model, seq, ans_pos)
                c = correct.cpu().numpy(); d = dist.cpu().numpy()
                oh += int(c.sum()); ot += c.size
                for lo, hi in buckets:
                    m = (d >= lo) & (d <= hi)
                    hit[(lo, hi)][0] += int(c[m].sum()); hit[(lo, hi)][1] += int(m.sum())
                done += B
            by = {f"{lo}-{hi if hi < 10**9 else 'inf'}": (h / t if t else None) for (lo, hi), (h, t) in hit.items()}
            out[N] = dict(overall=oh / ot, by_dist=by)
    return out


def train_arm(arm, args, device):
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 777)
    num_tokens = SPECIALS + args.vocab
    model, state = build(arm, num_tokens, args, device)
    params = dedup_params(model)
    nparam = sum(p.numel() for p in params)
    print(f"\n=== arm [{arm}] params={nparam/1e6:.2f}M recurrent_state={state} floats/seq ===", flush=True)
    try:
        from adam_atan2_pytorch import AdoptAtan2
        optim = AdoptAtan2(params, lr=args.lr)
    except Exception:
        optim = torch.optim.Adam(params, lr=args.lr)

    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        N = int(rng.integers(args.nmin, args.nmax + 1))      # variable load
        seq, ans_pos, _ = gen_batch(rng, args.batch, N, args.vocab, device)
        loss, correct = loss_and_pred(model, seq, ans_pos)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        if step % args.log_every == 0:
            print(f"[{arm}] step {step:5d} loss {loss.item():.3f} acc {correct.float().mean():.3f} "
                  f"N {N} t {(time.time()-t0)/60:.1f}m", flush=True)
        if args.eval_every and step > 0 and step % args.eval_every == 0:
            qev = evaluate(model, np.random.default_rng(args.seed + 777), args, device, n_per=256)
            bigN = max(args.eval_pairs)
            by = qev[bigN]["by_dist"]
            xs = [by[k] for k in ("33-64", "65-96", "97-inf") if by.get(k) is not None]
            xseg = float(np.mean(xs)) if xs else float("nan")
            print(f"[{arm}] ~eval step {step}: N{bigN} overall={qev[bigN]['overall']:.3f} "
                  f"cross-seg={xseg:.3f}", flush=True)
    ev = evaluate(model, eval_rng, args, device)
    print(f"[{arm}] FINAL " + " ".join(f"N{N}={ev[N]['overall']:.3f}" for N in args.eval_pairs), flush=True)
    return dict(arm=arm, params=nparam, state=state, eval=ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--nmin", type=int, default=8)
    p.add_argument("--nmax", type=int, default=28)
    p.add_argument("--eval-pairs", type=int, nargs="+", default=[8, 16, 24, 32], dest="eval_pairs")
    p.add_argument("--vocab", type=int, default=64)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--segment-len", type=int, default=32, dest="segment_len")
    p.add_argument("--mem-layers", type=int, nargs="+", default=[1, 3], dest="mem_layers")
    p.add_argument("--mem-dim-head", type=int, default=64, dest="mem_dim_head")
    p.add_argument("--mem-heads", type=int, default=4, dest="mem_heads")
    p.add_argument("--nm-seg", type=int, default=4, dest="nm_seg")
    p.add_argument("--nm-batch", type=int, default=16, dest="nm_batch")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=500, dest="log_every")
    p.add_argument("--eval-samples", type=int, default=1536, dest="eval_samples")
    p.add_argument("--eval-every", type=int, default=2000, dest="eval_every")
    p.add_argument("--seed", type=int, default=20260623)
    args = p.parse_args()

    device = torch.device("cuda")
    chance = 1.0 / args.vocab
    print(f"[ladder] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[ladder] arms={args.arms} trainN~U[{args.nmin},{args.nmax}] evalN={args.eval_pairs} "
          f"vocab={args.vocab} chance={chance:.4f} steps={args.steps} seed={args.seed}\n", flush=True)

    results = {}
    for arm in args.arms:
        results[arm] = train_arm(arm, args, device)

    # ---- ladder table: overall recall by eval-N ----
    print("\n" + "=" * 78, flush=True)
    print("LADDER — overall MQAR recall by load N (MK-NIAH interference slice)", flush=True)
    hdr = f"{'arm':>12} {'params':>8} {'state':>7} " + " ".join(f"{'N='+str(N):>8}" for N in args.eval_pairs)
    print(hdr, flush=True); print("-" * len(hdr), flush=True)
    for arm in args.arms:
        r = results[arm]
        row = f"{arm:>12} {r['params']/1e6:>7.2f}M {r['state']:>7} " + \
              " ".join(f"{r['eval'][N]['overall']:>8.3f}" for N in args.eval_pairs)
        print(row, flush=True)
    print(f"{'(chance)':>12} {'':>8} {'':>7} " + " ".join(f"{chance:>8.3f}" for _ in args.eval_pairs), flush=True)

    # ---- cross-segment (>32) recall: the memory-only regime ----
    def cross(arm, N):
        by = results[arm]["eval"][N]["by_dist"]
        vals = [by[k] for k in ("33-64", "65-96", "97-inf") if by.get(k) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    bigN = max(args.eval_pairs)
    print("\n" + "=" * 78, flush=True)
    print(f"CROSS-SEGMENT (>32 tok, memory-only) recall @ N={bigN}:", flush=True)
    for arm in args.arms:
        print(f"  {arm:>12}: {cross(arm, bigN):.3f}", flush=True)

    # ---- decision rule: payoff gate (deep vs best linear, on the hardest multi-key slice) ----
    linear_arms = [a for a in ("gated-delta", "gdn2", "linear") if a in results]
    if "deep" in results and linear_arms:
        best_lin = max(linear_arms, key=lambda a: cross(a, bigN))
        margin = cross("deep", bigN) - cross(best_lin, bigN)
        gdn2_lift = (cross("gdn2", bigN) - cross("gated-delta", bigN)) if {"gdn2", "gated-delta"} <= set(results) else float("nan")
        print("\n" + "=" * 78, flush=True)
        print(f"[ladder verdict] (single-seed, directional — read margins, not thresholds)", flush=True)
        print(f"  better-linear (gdn2 - gated-delta) cross-seg lift @N{bigN}: {gdn2_lift:+.3f}", flush=True)
        print(f"  deep - best_linear({best_lin}) cross-seg margin @N{bigN}: {margin:+.3f}", flush=True)
        print(f"  PAYOFF-GATE (need >~0.05 AND seed-robust to fund deep-mem kernel): "
              f"{'PASS-directional' if margin > 0.05 else 'not met'}", flush=True)
        print(f"  -> if PASS: run compression-survival probe (project mem dim to CCA ratio) before kernel.\n"
              f"     if deep~=best_linear: ship the winning linear cell, build no kernel.", flush=True)
    print(json.dumps({a: {"params": results[a]["params"], "state": results[a]["state"],
                          "eval": {str(N): results[a]["eval"][N] for N in args.eval_pairs}}
                      for a in args.arms}), flush=True)


if __name__ == "__main__":
    main()
