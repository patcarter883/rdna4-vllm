"""Offline tuner (AOT Profile & Cache): measure the ours-vs-Triton crossover for
a set of (N, K, group) GEMM shapes and write w4a8_fp8_wmma/crossover_cache.json.

Run ONCE per GPU (and optionally per model) inside the gfx1201 container; the
runtime adapter then does an O(1) JSON lookup with zero startup benchmarking.

Usage:
    python tune_crossover.py                    # default shape list
    python tune_crossover.py --model <hf_id>    # introspect a model's linear shapes
    python tune_crossover.py --shapes 4096,4096,128 11008,4096,128
"""
import argparse
import json
import os
import time

import torch

import w4a8_fp8_wmma  # noqa: F401  loads the op
from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
    triton_w4a16_gemm,
)

DEFAULT_SHAPES = [  # (N, K, group) — common Llama/Qwen-ish projections
    (2048, 4096, 128), (4096, 4096, 128), (11008, 4096, 128),
    (4096, 11008, 128), (4096, 14336, 128), (14336, 4096, 128),
    (12288, 4096, 128), (4096, 12288, 128), (28672, 8192, 128), (8192, 28672, 128),
]
M_SWEEP = [512, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192]
MARGIN = 0.98  # require ours <= triton_time * MARGIN to engage


def pack_nk8(w_int4, N, K, dev):  # (N,K) -> (N,K//8) our layout
    o = torch.zeros(N, K // 8, dtype=torch.int32, device=dev)
    for j in range(8):
        o |= (w_int4[:, j::8] & 0xF) << (j * 4)
    return o


def pack_kn8(w_int4, N, K, dev):  # (N,K) -> (K,N//8) triton layout
    wt = w_int4.t().contiguous()
    o = torch.zeros(K, N // 8, dtype=torch.int32, device=dev)
    for j in range(8):
        o |= (wt[:, j::8] & 0xF) << (j * 4)
    return o


def timed(f, it=15):
    for _ in range(4):
        f()
    torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / it


def crossover(N, K, group, dev):
    w4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    owp = pack_nk8(w4, N, K, dev)
    tbq = pack_kn8(w4, N, K, dev)
    sc = torch.rand(N, K // group, dtype=torch.float16, device=dev) * 0.02 + 0.001
    tsc = sc.t().contiguous()
    empty = torch.empty(0, dtype=torch.int32, device=dev)
    for M in M_SWEEP:
        x = torch.randn(M, K, dtype=torch.float16, device=dev)
        t_o = timed(lambda: torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, owp, sc, empty, 5))
        t_t = timed(lambda: triton_w4a16_gemm(a=x, b_q=tbq, scales=tsc, qzeros=None,
                                              group_size=group, zp_bias=8))
        if t_o <= t_t * MARGIN:
            return M
    return None


def shapes_from_model(model_id):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    qc = getattr(cfg, "quantization_config", {}) or {}
    g = qc.get("group_size", 128)
    H = cfg.hidden_size
    I = cfg.intermediate_size
    kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    hd = H // cfg.num_attention_heads
    qkv = (cfg.num_attention_heads + 2 * kv) * hd
    shapes = {(qkv, H, g), (H, H, g), (2 * I, H, g), (I, H, g), (H, I, g)}
    return sorted(shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--shapes", nargs="*", help="N,K,group ...")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "w4a8_fp8_wmma", "crossover_cache.json"))
    args = ap.parse_args()

    if args.shapes:
        shapes = [tuple(int(x) for x in s.split(",")) for s in args.shapes]
    elif args.model:
        shapes = shapes_from_model(args.model)
    else:
        shapes = DEFAULT_SHAPES

    dev = "cuda"
    table = {"_comment": "AOT crossover cache; null = always Triton. "
                         "Regenerate per GPU with tune_crossover.py"}
    for (N, K, g) in shapes:
        c = crossover(N, K, g, dev)
        table[f"{N},{K},{g}"] = c
        print(f"  N={N} K={K} g={g} -> crossover M = {c}")
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {args.out} ({len(shapes)} shapes)")


if __name__ == "__main__":
    main()
