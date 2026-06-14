"""Offline AOT tuner for the MoE M-adaptive dispatch (gfx1201).

Mirrors the dense path's tune_crossover.py: measures our grouped FP8-WMMA MoE op
vs the stock Triton moe_wna16 (`fused_experts`) across a grid of M for each expert
shape, finds the CONTIGUOUS [lo, hi] window of M where we actually win (ours/stock
< 1 - margin), and writes it to `moe_crossover_cache.json` keyed by the per-GPU
(possibly TP-sharded) expert shape. The runtime hook does an O(1) lookup and
engages our op ONLY inside that window; any shape/M not proven-winning falls back
to stock -> the MoE pathway is always >= stock (no regression).

Run ONCE per (model, TP) in the target container (needs GPU + the built op + vLLM):
    # TP=2 per-GPU shape for Qwen3.6-35B-A3B (inter sharded 896 -> 448):
    python tune_moe_crossover.py --E 128 --hidden 2304 --inter 448 --top_k 8 --group 32
    # TP=1 / full shape:
    python tune_moe_crossover.py --E 128 --hidden 2304 --inter 896 --top_k 8 --group 32
"""
import argparse
import json
import os
import time

import torch
import w4a8_fp8_wmma
from w4a8_fp8_wmma.moe_experts import _wna16_moe_to_op_layout, _run_grouped_moe
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

CACHE = os.environ.get(
    "VLLM_ROCM_W4A8_FP8_WMMA_MOE_CACHE",
    os.path.join(os.path.dirname(__file__), "w4a8_fp8_wmma",
                 "moe_crossover_cache.json"))
M_GRID = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]


def _bench(fn, iters=40, warmup=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def tune_shape(E, hidden, inter, group, top_k, margin, dev="cuda"):
    torch.manual_seed(0)
    N13, K13 = 2 * inter, hidden
    N2, K2 = hidden, inter
    w13 = torch.randint(0, 256, (E, N13, K13 // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device=dev)
    s13 = torch.rand(E, N13, K13 // group, device=dev, dtype=torch.float16) * 0.02 + 0.005
    s2 = torch.rand(E, N2, K2 // group, device=dev, dtype=torch.float16) * 0.02 + 0.005
    z13 = torch.randint(0, 256, (E, N13 // 2, K13 // group), dtype=torch.uint8, device=dev)
    z2 = torch.randint(0, 256, (E, N2 // 2, K2 // group), dtype=torch.uint8, device=dev)
    w13o, s13o, z13o = _wna16_moe_to_op_layout(w13, s13, z13)
    w2o, s2o, z2o = _wna16_moe_to_op_layout(w2, s2, z2)
    qc = int4_w4a16_moe_quant_config(w1_scale=s13, w2_scale=s2, w1_zp=z13,
                                     w2_zp=z2, block_shape=[0, group])
    wins = []
    print(f"  shape E={E} h={hidden} inter={inter} g={group} tk={top_k}: "
          f"{'M':>6} {'ours/stock':>10}")
    for M in M_GRID:
        x = torch.randn(M, hidden, dtype=torch.float16, device=dev) * 0.5
        tids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
        tw = torch.rand(M, top_k, dtype=torch.float32, device=dev)
        ours = lambda: _run_grouped_moe(
            x, w13o, w2o, s13o, s2o, z13o, z2o, tw, tids, MoEActivation.SILU,
            E, None, False, 5, out_dtype=x.dtype)
        stock = lambda: fused_experts(
            x, w13, w2, topk_weights=tw, topk_ids=tids,
            activation=MoEActivation.SILU, apply_router_weight_on_input=False,
            global_num_experts=E, expert_map=None, quant_config=qc)
        r = _bench(ours) / _bench(stock)
        win = r < (1.0 - margin)
        wins.append((M, win))
        print(f"  {'':>21} {M:>6} {r:>9.3f}{'  WIN' if win else ''}")
    # contiguous winning intervals [[lo, hi], ...] (the win region can be
    # non-contiguous -- e.g. a mid-M dip -- so a single [min,max] would wrongly
    # engage the loss zone; store each proven-win run separately).
    intervals, run = [], None
    for m, w in wins:
        if w:
            run = [m, m] if run is None else [run[0], m]
        elif run is not None:
            intervals.append(run); run = None
    if run is not None:
        intervals.append(run)
    return intervals or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--inter", type=int, required=True)
    ap.add_argument("--top_k", type=int, required=True)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.02,
                    help="require ours/stock < 1-margin to count as a win")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    print(f"device: {torch.cuda.get_device_name(0)}  margin={args.margin}")
    window = tune_shape(args.E, args.hidden, args.inter, args.group,
                        args.top_k, args.margin)
    key = f"{args.E},{args.hidden},{args.inter},{args.group},{args.top_k}"
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            cache = json.load(f)
    cache[key] = window  # [lo, hi] or null (never engage)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    print(f"\n{key} -> {window}  (written to {CACHE})")


if __name__ == "__main__":
    main()
