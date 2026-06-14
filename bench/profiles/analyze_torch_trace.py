"""Summarize vLLM torch-profiler kineto traces: GPU-kernel time by bucket.

Usage: python analyze_torch_trace.py bench/profiles/tp2-torch/*.json.gz

Reads each per-rank kineto trace, sums device-side event durations (cat=kernel /
gpu_memcpy / gpu_memset), groups kernels into ZAYA-relevant buckets, and prints a
per-trace breakdown. The buckets mirror the single-GPU rocprof baseline
(bench/profiles/README.md) plus the two TP=2-specific ones: all-reduce (RCCL) and
the replicated CCA cost.
"""
import gzip
import json
import sys
from collections import defaultdict

DEVICE_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def bucket(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("nccl", "rccl", "allreduce", "all_reduce", "reducescatter",
                            "allgather", "all_gather")):
        return "all-reduce / collective (TP)"
    if "fused_moe" in n or "grouped" in n and "gemm" in n:
        return "MoE expert GEMM (Triton fp8)"
    if any(k in n for k in ("cijk", "tensile", "hipblas", "rocblas", "gemm", "matmul",
                            "wmma", "_mm_", "mfma")):
        return "dense GEMM (rocBLAS/Tensile)"
    if any(k in n for k in ("conv", "roll", "cca")):
        return "CCA conv/state (replicated)"
    if any(k in n for k in ("paged", "attention", "attn", "flash")):
        return "paged attention decode"
    if any(k in n for k in ("rmsnorm", "norm", "softmax", "topk", "top_k", "argmax",
                            "gather", "scatter")):
        return "router/norm/topk fused"
    if any(k in n for k in ("fp8", "quant", "cvt", "scaled")):
        return "fp8 activation quant"
    if any(k in n for k in ("memcpy", "memset", "copy")):
        return "memcpy/copy"
    if any(k in n for k in ("elementwise", "vectorized", "at::native", "pointwise",
                            "add", "mul", "cat", "reduce", "fill", "index", "where",
                            "unrolled", "binary", "unary", "_efunc", "elemwise")):
        return "elementwise/reduce/cat (pointwise)"
    return "other"


def analyze(path: str):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    by_kernel = defaultdict(lambda: [0.0, 0])  # name -> [dur_us, count]
    for e in events:
        if e.get("cat") in DEVICE_CATS and "dur" in e:
            k = e["name"]
            by_kernel[k][0] += e["dur"]
            by_kernel[k][1] += 1
    total = sum(v[0] for v in by_kernel.values())
    by_bucket = defaultdict(lambda: [0.0, 0])
    for name, (dur, cnt) in by_kernel.items():
        b = by_bucket[bucket(name)]
        b[0] += dur
        b[1] += cnt

    print(f"\n=== {path} ===")
    print(f"total device-kernel time: {total/1e3:.1f} ms across "
          f"{sum(v[1] for v in by_kernel.values()):,} launches, "
          f"{len(by_kernel)} distinct kernels")
    print(f"{'%':>6} {'time(ms)':>10} {'calls':>10}  bucket")
    for b, (dur, cnt) in sorted(by_bucket.items(), key=lambda x: -x[1][0]):
        print(f"{100*dur/total:6.1f} {dur/1e3:10.1f} {cnt:10,}  {b}")
    print("  -- top 12 kernels --")
    for name, (dur, cnt) in sorted(by_kernel.items(), key=lambda x: -x[1][0])[:12]:
        print(f"{100*dur/total:6.1f} {dur/1e3:10.1f} {cnt:10,}  {name[:78]}")
    return total, by_bucket


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
