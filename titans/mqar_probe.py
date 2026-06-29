"""MQAR — synthetic multi-query associative recall: the field-standard test (Zoology/Based,
and what the Mamba/GDN/DeltaNet/Titans papers use) for whether a recurrent memory learns to
STORE and RETRIEVE key->value associations across distance.

Why this is the right instrument (the enwik8 char-LM was not): English next-char prediction
never *requires* arbitrary key->value recall, so it can't force the neural memory to work or
expose it failing. MQAR makes recall the *only* way to score, and accuracy (not NLL deltas) is
a high-signal metric.

The discriminator: attention here is segment-local (segment_len=32, block-diagonal), so a key
defined >32 tokens before its query is UNREACHABLE by attention — only the neural memory's
recurrent state can carry it. We train two otherwise-identical models from scratch on the SAME
data:
  - WITH neural memory   (mem layers active)
  - ABLATED              (no mem layers -> attention-only)
and compare recall accuracy bucketed by definition->query distance. If the memory works,
the WITH model keeps high accuracy past 32 tokens while the ABLATED model collapses to chance.

Sequence layout (one trial), symbols drawn from a vocab of S:
  [BOS] k0 v0 k1 v1 ... k(N-1) v(N-1)   kq0 a0 kq1 a1 ... kq(N-1) a(N-1)
  - keys distinct within a trial; values uniform over S (chance accuracy = 1/S).
  - queries are the same keys in shuffled order; target at each query's value slot is the
    value bound in the context. Loss/accuracy scored only at those value slots.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from titans_common import materialize_overlapping_params, dedup_params

PAD, BOS = 0, 1
SPECIALS = 2  # number of special tokens before symbols


def build(num_tokens, args, mem_layers):
    from titans_pytorch import MemoryAsContextTransformer, MemoryMLP
    has_mem = len(mem_layers) > 0
    nm_model = MemoryMLP(dim=args.mem_dim_head, depth=args.mem_depth) if has_mem else None
    model = MemoryAsContextTransformer(
        num_tokens=num_tokens,
        dim=args.dim,
        depth=args.depth,
        segment_len=args.segment_len,
        num_persist_mem_tokens=4,
        num_longterm_mem_tokens=4,
        # out-of-range sentinel guarantees zero memory layers regardless of default() semantics
        neural_memory_layers=tuple(mem_layers) if has_mem else (args.depth + 10,),
        neural_memory_segment_len=args.nm_seg,
        neural_memory_batch_size=args.nm_batch,
        neural_mem_weight_residual=has_mem,
        neural_memory_qkv_receives_diff_views=has_mem,
        use_flex_attn=False,            # RDNA4
        sliding_window_attn=False,
        neural_memory_model=nm_model,
        neural_memory_kwargs=dict(
            dim_head=args.mem_dim_head, heads=4, attn_pool_chunks=True, qk_rmsnorm=True,
            momentum=True, momentum_order=1, default_step_transform_max_lr=1e-1,
            use_accelerated_scan=False,  # RDNA4
            per_parameter_lr_modulation=True, per_head_learned_parameters=True,
        ) if has_mem else dict(),
    )
    n_mem = sum(1 for m in model.modules() if type(m).__name__ == "NeuralMemory")
    assert n_mem == len(mem_layers), f"expected {len(mem_layers)} NeuralMemory, got {n_mem}"
    return model


def gen_batch(rng, B, N, S, device):
    """Returns seq[B,L] long, ans_pos[N] (fixed answer-slot indices), dist[B,N] (def->query gap)."""
    L = 1 + 2 * N + 2 * N
    seq = np.empty((B, L), dtype=np.int64)
    dist = np.empty((B, N), dtype=np.int64)
    sym = lambda x: x + SPECIALS
    ctx_len = 1 + 2 * N
    ans_pos = [ctx_len + 2 * r + 1 for r in range(N)]  # value slot for query rank r
    for b in range(B):
        keys = rng.choice(S, size=N, replace=False)        # distinct keys
        vals = rng.integers(0, S, size=N)                  # values (uniform -> chance 1/S)
        seq[b, 0] = BOS
        for i in range(N):
            seq[b, 1 + 2 * i] = sym(keys[i])
            seq[b, 1 + 2 * i + 1] = sym(vals[i])
        order = rng.permutation(N)
        for r, i in enumerate(order):
            qk_pos = ctx_len + 2 * r
            seq[b, qk_pos] = sym(keys[i])
            seq[b, qk_pos + 1] = sym(vals[i])              # teacher-forced; scored here
            dist[b, r] = qk_pos - (1 + 2 * i)              # query-key pos minus def pos
    return (torch.from_numpy(seq).to(device),
            torch.tensor(ans_pos, device=device),
            torch.from_numpy(dist).to(device))


def loss_and_pred(model, seq, ans_pos):
    logits = model(seq, return_loss=False)                 # [B,L,V]; logits[:,t] predicts t+1
    pred_logits = logits[:, ans_pos - 1]                   # [B,N,V] predict the value slots
    tgt = seq[:, ans_pos]                                  # [B,N]
    loss = F.cross_entropy(pred_logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    pred = pred_logits.argmax(-1)                          # [B,N]
    return loss, (pred == tgt)                             # correct mask [B,N]


@torch.no_grad()
def evaluate(model, rng, args, device, n_eval=2048):
    model.eval()
    buckets = [(1, 16), (17, 32), (33, 64), (65, 96), (97, 10 ** 9)]
    hit = {b: 0 for b in buckets}; tot = {b: 0 for b in buckets}
    overall_hit = overall_tot = 0
    done = 0
    while done < n_eval:
        B = min(args.batch, n_eval - done)
        seq, ans_pos, dist = gen_batch(rng, B, args.pairs, args.vocab, device)
        _, correct = loss_and_pred(model, seq, ans_pos)
        c = correct.cpu().numpy(); d = dist.cpu().numpy()
        overall_hit += int(c.sum()); overall_tot += c.size
        for lo, hi in buckets:
            m = (d >= lo) & (d <= hi)
            hit[(lo, hi)] += int(c[m].sum()); tot[(lo, hi)] += int(m.sum())
        done += B
    acc = {f"{lo}-{hi if hi < 10**9 else 'inf'}": (hit[(lo, hi)] / tot[(lo, hi)] if tot[(lo, hi)] else None)
           for lo, hi in buckets}
    return overall_hit / overall_tot, acc


def train_one(tag, mem_layers, args, device):
    rng = np.random.default_rng(args.seed)
    eval_rng = np.random.default_rng(args.seed + 777)
    num_tokens = SPECIALS + args.vocab
    model = build(num_tokens, args, mem_layers).to(device)
    if mem_layers:
        materialize_overlapping_params(model)
    params = dedup_params(model)
    n = sum(p.numel() for p in params)
    print(f"\n=== train [{tag}] mem_layers={list(mem_layers)} params={n/1e6:.2f}M ===", flush=True)
    try:
        from adam_atan2_pytorch import AdoptAtan2
        optim = AdoptAtan2(params, lr=args.lr)
    except Exception:
        optim = torch.optim.Adam(params, lr=args.lr)

    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        seq, ans_pos, _ = gen_batch(rng, args.batch, args.pairs, args.vocab, device)
        loss, correct = loss_and_pred(model, seq, ans_pos)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        if step % args.log_every == 0:
            acc = correct.float().mean().item()
            print(f"[{tag}] step {step:5d} loss {loss.item():.3f} train_acc {acc:.3f} "
                  f"t {(time.time()-t0)/60:.1f}m", flush=True)
    overall, by_dist = evaluate(model, eval_rng, args, device)
    print(f"[{tag}] FINAL overall_acc={overall:.3f}  by_distance={by_dist}", flush=True)
    return dict(tag=tag, mem_layers=list(mem_layers), params=n, overall=overall, by_dist=by_dist)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=24, help="N key-value pairs (and N queries)")
    p.add_argument("--vocab", type=int, default=64, help="S symbol vocab (chance acc = 1/S)")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--segment-len", type=int, default=32, dest="segment_len")
    p.add_argument("--mem-layers", type=int, nargs="+", default=[1, 3], dest="mem_layers")
    p.add_argument("--mem-dim-head", type=int, default=64, dest="mem_dim_head")
    p.add_argument("--mem-depth", type=int, default=2, dest="mem_depth")
    p.add_argument("--nm-seg", type=int, default=4, dest="nm_seg",
                   help="neural_memory chunk size (smaller = more frequent stores)")
    p.add_argument("--nm-batch", type=int, default=16, dest="nm_batch",
                   help="neural_memory_batch_size (store-commit granularity)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--log-every", type=int, default=500, dest="log_every")
    p.add_argument("--seed", type=int, default=20260623)
    args = p.parse_args()

    device = torch.device("cuda")
    chance = 1.0 / args.vocab
    L = 1 + 4 * args.pairs
    print(f"[mqar] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[mqar] N={args.pairs} vocab={args.vocab} seq_len={L} segment_len={args.segment_len} "
          f"chance_acc={chance:.4f} steps={args.steps}", flush=True)
    print(f"[mqar] cross-segment buckets (>{args.segment_len} tokens) are reachable ONLY via "
          f"the neural memory\n", flush=True)

    res_mem = train_one("MEM", tuple(args.mem_layers), args, device)
    res_abl = train_one("ABLATED", (), args, device)

    print("\n" + "=" * 72, flush=True)
    print(f"{'distance bucket':>16} {'MEM acc':>10} {'ABLATED acc':>12} {'memory lift':>12}",
          flush=True)
    print("-" * 72, flush=True)
    for k in res_mem["by_dist"]:
        a = res_mem["by_dist"][k]; b = res_abl["by_dist"][k]
        lift = (a - b) if (a is not None and b is not None) else None
        fa = f"{a:.3f}" if a is not None else "  -  "
        fb = f"{b:.3f}" if b is not None else "  -  "
        fl = f"{lift:+.3f}" if lift is not None else "  -  "
        print(f"{k:>16} {fa:>10} {fb:>12} {fl:>12}", flush=True)
    print("-" * 72, flush=True)
    print(f"{'OVERALL':>16} {res_mem['overall']:>10.3f} {res_abl['overall']:>12.3f} "
          f"{res_mem['overall']-res_abl['overall']:>+12.3f}", flush=True)

    # verdict: cross-segment (>32) accuracy is the discriminator
    cross_keys = ["33-64", "65-96", "97-inf"]
    mem_cross = [res_mem["by_dist"][k] for k in cross_keys if res_mem["by_dist"].get(k) is not None]
    abl_cross = [res_abl["by_dist"][k] for k in cross_keys if res_abl["by_dist"].get(k) is not None]
    mem_x = float(np.mean(mem_cross)) if mem_cross else float("nan")
    abl_x = float(np.mean(abl_cross)) if abl_cross else float("nan")
    print("\n" + "=" * 72, flush=True)
    print(f"[verdict] cross-segment (>{args.segment_len}) acc: MEM={mem_x:.3f} ABLATED={abl_x:.3f} "
          f"chance={chance:.3f}", flush=True)
    if mem_x > 0.5 and mem_x - abl_x > 0.2:
        print("[verdict] => NEURAL MEMORY RETAINS+RETRIEVES across segments. Architecture works "
              "on RDNA4; ingestion premise validated -> proceed to training-path choice.", flush=True)
    elif abl_x > 0.5:
        print("[verdict] => ATTENTION-ONLY also solves far recall — task too easy / not isolating "
              "memory (raise pairs or shrink segment_len) before concluding.", flush=True)
    else:
        print("[verdict] => memory shows NO clear cross-segment recall. Before concluding the "
              "architecture fails on RDNA4, sweep memory hyperparams (--nm-seg/--nm-batch, "
              "mem-layers, steps) — a false-negative from misconfig is the main risk.", flush=True)
    print(json.dumps(dict(chance=chance, mem=res_mem, ablated=res_abl)), flush=True)


if __name__ == "__main__":
    main()
