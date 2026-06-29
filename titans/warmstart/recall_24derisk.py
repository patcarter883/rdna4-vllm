"""CAM build step 0 — the 2:4-by-design vs dense fidelity DE-RISK on the v0 harness.

The only residual risk of the locked 2:4 serve path is FIDELITY: does training the tap's serve-weight
projections 2:4-sparse FROM INIT (SR-STE) hold the v0 recall PASS, vs a dense reference? This is ~atlas
-independent, so we answer it on the existing v0 harness before any atlas spend (CANONICAL_BUILD_PLAN §3
step 0).

Method (CHEAP, reuses the frozen v0 memory — does NOT re-bind):
  - Load the frozen base (Qwen3.5-4B) + reload the FROZEN v0 memory front-end (BoltAdapter) from
    ckpt/cam_v0_L24.pt. The memory is fixed; only the TAP is (re)trained.
  - Train a FRESH GatedMemoryTap at L=24 in TWO arms from the SAME seed/recipe:
        dense  — the v0 tap (nn.Linear projections), the fidelity control.
        s24    — identical, but to_q/to_k/to_v/to_o are Mask24Linear (2:4-by-design SR-STE from init).
    Only the tap trains (LM-loss through the frozen base), gamma-alone zero-init, NaN/grad guards,
    fp32-compute/cast-back dtype pattern — all inherited from gated_tap.py.
  - Eval both arms with the v0 eval (memory / no_memory / ceiling) and report the fidelity DELTA.

Smoke first: --steps 40 to prove the 2:4 mask plumbs + trains (gate opens, sparsity ~0.5) before any
longer run. A ~1-hr smoke (--time-budget) on the s24 arm pins step-time / steps-to-converge.

Run (1 leased card; absolute arbiter path):
  /home/pat/code/vllm-gfx1201/scripts/gpu-lease.sh -n 1 --name titans-d24 -- \
    titans/warmstart/run_m2.sh titans-d24 --entry warmstart/recall_24derisk.py -- \
      --load-ckpt warmstart/ckpt/cam_v0_L24.pt --steps 3000 --arms dense,s24
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "deep_mem"))
from m2_adapter import MODEL, DEV, load_frozen_base                       # noqa: E402
from recall_deepmem import NAME_CANDIDATES, CARGO_CANDIDATES, single_token_ids, DocBuilder  # noqa: E402
from recall_boltA import BoltAdapter                                      # noqa: E402
from gated_tap import GatedMemoryTap, MAGInjector, decoder_layers         # noqa: E402
from recall_mag import memory_bank, _leakfree_ctx, eval_generative_mag, load_ckpt  # noqa: E402
from sparse24 import Mask24Linear                                         # noqa: E402

LN2 = math.log(2.0)


class Sparse24Tap(GatedMemoryTap):
    """GatedMemoryTap whose four projection weights are 2:4-by-design (SR-STE). Forward math is the
    parent's verbatim (mask lives inside Mask24Linear); gamma-alone zero-init keeps the no-op at init."""

    def __init__(self, base_hidden, mem_dim, n_heads=8, srste_lambda=2e-4):
        super().__init__(base_hidden, mem_dim, n_heads)
        self.to_q = Mask24Linear(base_hidden, base_hidden, srste_lambda)
        self.to_k = Mask24Linear(mem_dim, base_hidden, srste_lambda)
        self.to_v = Mask24Linear(mem_dim, base_hidden, srste_lambda)
        self.to_o = Mask24Linear(base_hidden, base_hidden, srste_lambda)
        # gamma already zeros from the parent; tap compute dtype is fp32 by default (params are fp32).

    def sparsity(self):
        return {n: getattr(self, n).sparsity_report() for n in ("to_q", "to_k", "to_v", "to_o")}


class DeriskInjector(MAGInjector):
    """MAGInjector that builds dense OR 2:4 taps. arm in {'dense','s24'}."""

    def __init__(self, base, tap_layers, mem_dim, n_heads=8, arm="dense", srste_lambda=2e-4):
        H = base.config.get_text_config().hidden_size
        self.layers = decoder_layers(base)
        self.tap_layers = list(tap_layers)
        if arm == "dense":
            taps = {str(L): GatedMemoryTap(H, mem_dim, n_heads) for L in self.tap_layers}
        elif arm == "s24":
            taps = {str(L): Sparse24Tap(H, mem_dim, n_heads, srste_lambda) for L in self.tap_layers}
        else:
            raise ValueError(arm)
        self.taps = nn.ModuleDict(taps)
        self._handles = []
        self.arm = arm

    def sparsity_stats(self):
        out = {}
        for L in self.tap_layers:
            t = self.taps[str(L)]
            if isinstance(t, Sparse24Tap):
                out[L] = t.sparsity()
        return out


def train_taps(base, adapter, injector, builder, rng, args, tag, time_budget=0.0):
    """LM-loss-through-frozen-base tap training with NaN/grad-skip guard. Returns (n_steps, sec/step,
    step-at-which acc first crossed conv_acc, final-window acc)."""
    injector.attach().train()
    opt = torch.optim.AdamW(injector.parameters(), lr=args.lr)
    t0 = time.time()
    conv_step, acc_hist, n_done = None, [], 0
    for step in range(args.steps):
        if time_budget and (time.time() - t0) > time_budget:
            print(f"[d24][{tag}] time budget {time_budget:.0f}s hit at step {step}", flush=True)
            break
        opt.zero_grad()
        ids, ans, apos = builder.build(rng, args.batch, local=False)
        ids, ans = ids.to(DEV), ans.to(DEV)
        with torch.no_grad():
            bank = memory_bank(adapter, ids, args.seg_len, builder.qa_start, apos, carry=True)
        injector.set_bank(bank)
        ctx_emb = _leakfree_ctx(base, builder, ids, apos)
        logits = base(inputs_embeds=ctx_emb).logits[:, -1].float()
        loss = F.cross_entropy(logits, ans)
        if not torch.isfinite(loss):
            print(f"[d24][{tag}] step {step}: non-finite loss -> skip", flush=True)
            continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(list(injector.parameters()), 1.0)
        if not torch.isfinite(gn):
            print(f"[d24][{tag}] step {step}: non-finite grad -> skip", flush=True)
            opt.zero_grad()
            continue
        opt.step()
        n_done = step + 1
        acc = (logits.argmax(-1) == ans).float().mean().item()
        acc_hist.append(acc)
        if conv_step is None and np.mean(acc_hist[-10:]) >= args.conv_acc and len(acc_hist) >= 10:
            conv_step = step
        if step % 50 == 0 or step == args.steps - 1:
            extra = ""
            if isinstance(injector, DeriskInjector) and injector.arm == "s24":
                sp = injector.sparsity_stats()
                extra = f" sparsity={ {L: round(np.mean(list(d.values())), 3) for L, d in sp.items()} }"
            print(f"[d24][{tag}] step {step:4d} loss {loss.item():.3f} acc {acc:.3f} "
                  f"gate {injector.gate_stats()}{extra}", flush=True)
    injector.set_bank(None)
    sec_step = (time.time() - t0) / max(n_done, 1)
    win_acc = float(np.mean(acc_hist[-50:])) if acc_hist else 0.0
    return n_done, sec_step, conv_step, win_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load-ckpt", type=str, required=True, dest="load_ckpt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seg-len", type=int, default=32, dest="seg_len")
    ap.add_argument("--qa-seg", type=int, default=2, dest="qa_seg")
    ap.add_argument("--M", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260629)
    ap.add_argument("--arms", type=str, default="dense,s24")
    ap.add_argument("--srste-lambda", type=float, default=2e-4, dest="srste_lambda")
    ap.add_argument("--conv-acc", type=float, default=0.90, dest="conv_acc")
    ap.add_argument("--eval-n", type=int, default=512, dest="eval_n")
    ap.add_argument("--time-budget", type=float, default=0.0, dest="time_budget",
                    help="seconds; if >0, cap each arm's training wall-time (1-hr smoke)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    base, tok = load_frozen_base()
    H = base.config.get_text_config().hidden_size
    embed_weight = base.get_input_embeddings().weight.detach().float().clone()

    names = single_token_ids(tok, NAME_CANDIDATES)
    cargo = single_token_ids(tok, CARGO_CANDIDATES, prefix="")
    builder = DocBuilder(tok, names, cargo, args.M, args.seg_len, args.qa_seg, phrasing="dict")

    # reload the FROZEN v0 memory front-end (BoltAdapter) — do NOT re-bind. We discard the ckpt's trained
    # dense tap and train fresh taps so dense and s24 are apples-to-apples from the same init recipe.
    adapter, _ck_injector, L, ck = load_ckpt(args.load_ckpt, embed_weight, base, DEV)
    _ck_injector.detach()
    _ck_injector.layers = None
    # free VRAM before training: the standalone embed clone, the ckpt injector's stale tap, and the
    # adapter's tied unembed buffer (~1.5GB fp32, used ONLY by the direct bind loss — never at eval).
    del embed_weight, _ck_injector
    if hasattr(adapter, "unembed"):
        adapter.unembed = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mem_dim, tap_heads, d_carry = ck["mem_dim"], ck["tap_heads"], ck.get("d_carry", float("nan"))
    print(f"[d24] {MODEL} | H={H} | reload-memory tap L={L} carry {d_carry:.3f} | "
          f"mem_dim={mem_dim} tap_heads={tap_heads} | chance {1/args.M:.3f} | arms={args.arms} "
          f"steps={args.steps} budget={args.time_budget}s", flush=True)

    rows = []
    for arm in [a for a in args.arms.split(",") if a]:
        # reseed per arm so dense & s24 share the SAME data stream + tap-q/k/v init recipe
        torch.manual_seed(args.seed)
        rng_arm = np.random.default_rng(args.seed)
        injector = DeriskInjector(base, [L], mem_dim, n_heads=tap_heads, arm=arm,
                                  srste_lambda=args.srste_lambda).to(DEV)
        n_steps, sec_step, conv_step, win_acc = train_taps(
            base, adapter, injector, builder, rng_arm, args, arm, time_budget=args.time_budget)
        gen = eval_generative_mag(base, adapter, injector, builder, rng_arm, args, n=args.eval_n)
        m_acc, nm_acc = gen["memory"][1], gen["no_memory"][1]
        m_nll, nm_nll = gen["memory"][0], gen["no_memory"][0]
        ceil = gen["local_control"][1]
        passed = (m_acc > nm_acc + 0.15 and m_acc > 0.5)
        print(f"\n[d24][{arm}] === generative through frozen base ===", flush=True)
        for c in ("local_control", "memory", "no_memory"):
            print(f"  {c:>14} NLL {gen[c][0]:8.3f}  acc {gen[c][1]:.3f}", flush=True)
        print(f"[d24][{arm}] memory {m_acc:.3f} / no_memory {nm_acc:.3f} / ceiling {ceil:.3f}; "
              f"ΔNLL {nm_nll - m_nll:+.3f} bits; train {n_steps} steps @ {sec_step:.3f}s/step; "
              f"converged@{conv_step}; {'PASS' if passed else 'FAIL'}", flush=True)
        if arm == "s24":
            print(f"[d24][s24] mask sparsity {injector.sparsity_stats()}", flush=True)
        injector.detach()
        rows.append((arm, m_acc, nm_acc, ceil, nm_nll - m_nll, n_steps, sec_step, conv_step, passed))
        # free this arm's tap + autograd graph before building the next (16GB card is tight: the GDN
        # base forward peaks ~13GB, so a stale tap/optimizer from the prior arm OOMs the next one).
        injector.layers = None
        del injector, gen
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    print("\n[d24] === STEP-0 SUMMARY (2:4-by-design vs dense fidelity) ===", flush=True)
    print(f"{'arm':>6} {'memory':>7} {'no_mem':>7} {'ceiling':>8} {'ΔNLL':>9} {'steps':>6} "
          f"{'s/step':>7} {'conv@':>6} {'verdict':>7}", flush=True)
    for arm, m, nm, c, dn, ns, ss, cs, p in rows:
        print(f"{arm:>6} {m:7.3f} {nm:7.3f} {c:8.3f} {dn:9.3f} {ns:6d} {ss:7.3f} "
              f"{str(cs):>6} {'PASS' if p else 'FAIL':>7}", flush=True)
    d = {a: m for a, m, *_ in rows}
    if "dense" in d and "s24" in d:
        print(f"[d24] FIDELITY DELTA (s24 − dense) memory-acc = {d['s24'] - d['dense']:+.3f}", flush=True)
    print("[d24] PASS rule: memory > no_memory+0.15 and > 0.5 (the v0 bar). 2:4 HOLDS if s24 PASSES "
          "and its delta vs dense is small.", flush=True)


if __name__ == "__main__":
    main()
