"""Recall probe — does the trained tiny Titans MAC actually RETAIN and RETRIEVE
information through its neural-memory state across segment boundaries?

This is the cheap (~1 GPU-hour) decision gate before funding any training run or
serve integration: if the memory mechanism cannot carry a planted "needle" across
a segment boundary at this (toy) scale, the whole "ingest a codebase into the
memory state and reuse it" premise has no legs at any scale.

## Why this isolates the memory cleanly
The MAC model has segment-local attention: q/k/v are reshaped per segment of
total_segment_len = segment_len(32) + num_longterm_mem_tokens(4) = 36 and attended
block-diagonally (sliding_window_attn=False). So tokens in *different* 32-char
input segments CANNOT attend to each other directly. The ONLY pathway for
information to cross a segment boundary is:
  - the neural memory (layers 2,4,6) recurrent state, and
  - the longterm-mem tokens, which are themselves *read from* that memory.
The 4 persistent-mem tokens are static (input-independent) and cannot carry a
per-sequence needle. Therefore any cross-segment recall == the neural memory.

## The task (associative recall by NLL)
Per trial, on a length-512 sequence of real enwik8 filler:
  - EARLY: overwrite a slice with a random needle string S (L lowercase chars).
  - PROBE (in a later segment, `dist` segments away): write S again as
    [cue (C chars)] + [rest (L-C chars)] and measure mean per-char NLL on `rest`.
S is random, so `rest` is unpredictable from local context (chance = log2(26) =
4.70 bits/char). The model can only do better than chance on `rest` if it
retrieved S from memory (set up by the EARLY occurrence) and matched on the cue.

Conditions per distance:
  MATCH    : EARLY = S            -> recall possible
  MISMATCH : EARLY = S' (other)   -> needle never seen; same filler/probe -> floor
  recall_gap = NLL_mismatch - NLL_match  (bits/char; >0 == memory recalled)

POSITIVE CONTROL (within-segment): two copies of S inside segment 0 — attention
*can* see the first copy, so a working probe MUST show a large gap here. This
proves the methodology detects recall when info is locally available; if even
this is ~0 the probe (not the memory) is broken.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from titans_common import build_mac_model, materialize_overlapping_params

LN2 = math.log(2.0)
SEG = 32  # input segment_len (attention window in original-token space)
A_LO, A_HI = 97, 123  # 'a'..'z'


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_mac_model(cfg)
    materialize_overlapping_params(model)
    model = model.to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing
                    if "memory_model.model" not in k and "memory_model.norm" not in k]
    print(f"[probe] ckpt step={ckpt.get('step')} params={cfg.get('param_count','?')} "
          f"load: missing={len(missing)}(real={len(real_missing)}) unexpected={len(unexpected)}",
          flush=True)
    model.eval()
    return model, cfg


def make_needle(rng, pool, L, kind):
    if kind == "random":
        return rng.integers(A_LO, A_HI, size=L, dtype=np.uint8)
    # in-distribution: a real L-char snippet sampled from the data pool
    j = rng.integers(0, pool.shape[0] - L - 1)
    return pool[j:j + L].copy()


def build_trial(rng, filler_pool, seq_len, L, C, dist, condition, kind):
    """Return (tokens[seq_len] uint8, predicted_positions list[int]).

    dist==0 -> within-segment positive control: both copies inside segment 0
              (attention CAN see EARLY), so a working probe MUST show a big gap.
    dist>=1 -> cross-segment: EARLY in segment 0, PROBE in segment `dist`; the only
              pathway between them is the neural memory.
    condition in {"match","mismatch"}: EARLY = S (match) or an independent S' (mismatch);
    PROBE is always S. recall_gap = NLL_mismatch - NLL_match.
    """
    i = rng.integers(0, filler_pool.shape[0] - seq_len - 1)
    toks = filler_pool[i:i + seq_len].copy()

    S = make_needle(rng, filler_pool, L, kind)
    S_other = make_needle(rng, filler_pool, L, kind)
    early_S = S if condition == "match" else S_other

    if dist == 0:                          # within segment 0: EARLY [0:L], PROBE [L:2L]
        early_off, probe_off = 0, L
    else:
        early_off = 4                      # segment 0
        probe_off = dist * SEG + 4         # segment `dist`
    toks[early_off:early_off + L] = early_S
    toks[probe_off:probe_off + L] = S      # PROBE always S

    # predicted positions = the "rest" of the PROBE occurrence (after the cue).
    # Char at position t is predicted from logits[t-1]; we score the rest chars.
    rest_positions = list(range(probe_off + C, probe_off + L))
    return toks, rest_positions


@torch.no_grad()
def score_batch(model, toks_batch, positions_batch, device):
    """Mean per-char NLL (bits) over the predicted positions, per trial in the batch."""
    x = torch.from_numpy(np.stack(toks_batch)).long().to(device)   # [B, seq_len]
    logits = model(x, return_loss=False)                           # [B, seq_len, V]
    logp = F.log_softmax(logits.float(), dim=-1)
    out = []
    for b, positions in enumerate(positions_batch):
        pos = torch.tensor(positions, device=device)
        tgt = x[b, pos]                       # true chars at predicted positions
        lp = logp[b, pos - 1].gather(-1, tgt[:, None]).squeeze(-1)  # log p(char|prefix)
        out.append((-lp.mean().item()) / LN2)  # nats -> bits
    return out


def run_condition(model, rng, filler, args, dist, condition, device):
    bits = []
    n = args.n_trials
    bs = args.batch
    done = 0
    while done < n:
        cur = min(bs, n - done)
        tb, pb = [], []
        for _ in range(cur):
            toks, pos = build_trial(rng, filler, args.seq_len, args.L, args.C,
                                    dist, condition, args.needle)
            tb.append(toks); pb.append(pos)
        bits.extend(score_batch(model, tb, pb, device))
        done += cur
    arr = np.array(bits)
    return arr.mean(), arr.std(ddof=1) / math.sqrt(len(arr))  # mean, sem


def baseline_bpc(model, rng, filler, args, device):
    """Sanity: score plain (un-modified) filler text. Must reproduce the known ~1.83 BPC,
    which validates the position indexing / logit alignment before trusting any gap."""
    bits = []
    done = 0
    # score a contiguous run of chars in the middle of each sequence
    lo, hi = 64, args.seq_len - 64
    while done < args.n_trials:
        cur = min(args.batch, args.n_trials - done)
        tb, pb = [], []
        for _ in range(cur):
            i = rng.integers(0, filler.shape[0] - args.seq_len - 1)
            tb.append(filler[i:i + args.seq_len].copy())
            pb.append(list(range(lo, hi)))
        bits.extend(score_batch(model, tb, pb, device))
        done += cur
    return float(np.mean(bits))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="/ckpt/model_best.pt")
    p.add_argument("--data", default="/work/ref-titans-pytorch/data/enwik8.gz")
    p.add_argument("--seq-len", type=int, default=512, dest="seq_len")
    p.add_argument("--L", type=int, default=16, help="needle length (chars)")
    p.add_argument("--C", type=int, default=6, help="cue length (chars) at the probe")
    p.add_argument("--n-trials", type=int, default=256, dest="n_trials")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--needle", choices=["english", "random"], default="english",
                   help="needle drawn in-distribution (english) or OOD random a-z")
    p.add_argument("--dists", type=int, nargs="+", default=[0, 1, 2, 4, 8, 12],
                   help="probe distance in segments; 0 = within-segment positive control")
    p.add_argument("--seed", type=int, default=20260623)
    args = p.parse_args()

    assert 2 * args.L <= SEG, "within-segment control needs 2L <= 32"
    device = torch.device("cuda")
    rng = np.random.default_rng(args.seed)

    print(f"[probe] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
          f"needle={args.needle} L={args.L} C={args.C} predict={args.L-args.C} chars/trial "
          f"n={args.n_trials}", flush=True)

    model, _ = load_model(args.ckpt, device)

    with gzip.open(args.data) as fh:
        raw = np.frombuffer(fh.read(int(95e6)), dtype=np.uint8).copy()
    _, va = np.split(raw, [int(90e6)])
    filler = va  # held-out split

    # SANITY: plain-text BPC must reproduce the known ~1.83 to trust logit alignment
    base = baseline_bpc(model, rng, filler, args, device)
    print(f"[sanity] plain-text baseline = {base:.3f} bits/char (expect ~1.83; "
          f"validates indexing)\n", flush=True)

    print(f"{'dist(seg)':>9} {'chars apart':>11} {'NLL_match':>12} {'NLL_mismatch':>14} "
          f"{'recall_gap':>13}", flush=True)
    print("-" * 64, flush=True)
    results = []
    for d in args.dists:
        mm, ms = run_condition(model, rng, filler, args, d, "match", device)
        xm, xs = run_condition(model, rng, filler, args, d, "mismatch", device)
        gap = xm - mm
        gap_sem = math.sqrt(ms ** 2 + xs ** 2)
        tag = "  (within/control)" if d == 0 else ""
        results.append(dict(dist=d, chars=d * SEG, nll_match=mm, nll_mismatch=xm,
                            gap=gap, gap_sem=gap_sem))
        print(f"{d:>9} {d*SEG:>11} {mm:>8.3f}±{ms:.2f} {xm:>10.3f}±{xs:.2f} "
              f"{gap:>8.3f}±{gap_sem:.2f}{tag}", flush=True)

    # verdict
    print("\n" + "=" * 64, flush=True)
    indexing_ok = abs(base - 1.83) < 0.6
    within = next(r for r in results if r["dist"] == 0)
    within_works = within["gap"] > 3 * within["gap_sem"] and within["gap"] > 0.5
    cross = [r for r in results if r["dist"] >= 1]
    best = max(cross, key=lambda r: r["gap"])
    cross_signal = any(r["gap"] > 3 * r["gap_sem"] and r["gap"] > 0.25 for r in cross)
    print(f"[verdict] indexing sane (baseline≈1.83 BPC): {indexing_ok} (got {base:.2f})", flush=True)
    print(f"[verdict] method sanity (within-segment recall): {within_works} "
          f"(within gap {within['gap']:.3f}±{within['gap_sem']:.2f} bits)", flush=True)
    print(f"[verdict] cross-segment memory recall detected: {cross_signal} "
          f"(best gap {best['gap']:.3f}±{best['gap_sem']:.2f} bits at dist={best['dist']} seg / "
          f"{best['chars']} chars)", flush=True)
    if not indexing_ok or not within_works:
        print("[verdict] => INCONCLUSIVE — fix the probe (indexing/positive-control) "
              "before drawing memory conclusions.", flush=True)
    elif cross_signal:
        print("[verdict] => the neural memory RETAINS+RETRIEVES across segments at this scale. "
              "Ingestion premise has legs; proceed to choose training path.", flush=True)
    else:
        print("[verdict] => method works but memory shows NO cross-segment recall at this scale. "
              "Ingestion premise unsupported here; do not fund a training run on this basis alone.",
              flush=True)
    print(json.dumps(dict(baseline_bpc=base, results=results)), flush=True)


if __name__ == "__main__":
    main()
