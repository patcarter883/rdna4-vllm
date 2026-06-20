"""Perf-neutrality gate for the dense TileConfig consolidation (Commit 3).

The gemm_tiled re-expression is already proven BIT-EXACT to v5/v10
(test_dense_tiled_bitexact.py), so this only decides whether the tiled path may
become the served default: it must be perf-NEUTRAL (ratio in ~[0.95, 1.05]).
Times mmq_fp8_gemm(prefill_wmma / prefill_wmma_ashuffle) with VLLM_W4A8_DENSE_TILED
unset (standalone v5/v10) vs =1 (tiled), same .so, prefill-regime shapes.
Since the kernels emit identical instructions at matched __launch_bounds__, the
expectation is ~1.00; a real delta would be a register/occupancy refactor artifact.
"""
import os
import sys
import time

import torch

try:
    import w4a8_fp8_wmma
except ImportError as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)

TILED_ENV = "VLLM_W4A8_DENSE_TILED"


def pack_uint4(w):
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def _bench(fn, it=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it * 1e6  # us


def time_one(kernel, M, N, K, group_size):
    dev = torch.device("cuda")
    G = K // group_size
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    w_packed = pack_uint4(torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev))
    scales = torch.randn(N, G, dtype=torch.float16, device=dev).abs() * 0.01 + 0.001

    def call():
        return w4a8_fp8_wmma.mmq_fp8_gemm(x, w_packed, scales, kernel=kernel)

    os.environ.pop(TILED_ENV, None)
    t_orig = _bench(call)
    os.environ[TILED_ENV] = "1"
    t_tiled = _bench(call)
    os.environ.pop(TILED_ENV, None)

    ratio = t_tiled / t_orig if t_orig > 0 else float("inf")
    ok = 0.95 <= ratio <= 1.05
    print(f"  {kernel:22s} M={M} N={N} K={K} g={group_size}: "
          f"v5/v10={t_orig:8.1f}us tiled={t_tiled:8.1f}us  ratio={ratio:.3f} "
          f"-> {'NEUTRAL' if ok else 'OFF'}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("FAIL: no device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    res = []
    # Prefill regime (large M) on model-ish shapes.
    print("=== prefill_wmma (v5) tiled-vs-standalone ===")
    for (M, N, K, g) in [(256, 4096, 4096, 128), (512, 4096, 4096, 128),
                         (1024, 4096, 4096, 128), (256, 4096, 4096, 32)]:
        res.append(time_one("prefill_wmma", M, N, K, g))
    print("=== prefill_wmma_ashuffle (v10) tiled-vs-standalone ===")
    for (M, N, K, g) in [(256, 4096, 4096, 128), (512, 4096, 4096, 128),
                         (1024, 4096, 4096, 128), (256, 4096, 4096, 32)]:
        res.append(time_one("prefill_wmma_ashuffle", M, N, K, g))
    print("=" * 56)
    n_off = sum(1 for r in res if not r)
    if n_off == 0:
        print(f"PERF-NEUTRAL ({len(res)}) — safe to flip the dense default to tiled")
        sys.exit(0)
    print(f"NOT NEUTRAL: {n_off}/{len(res)} off [0.95,1.05] — keep v5/v10 as default")
    sys.exit(1)


if __name__ == "__main__":
    main()
