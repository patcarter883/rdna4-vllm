"""Bit-exact acceptance gate for the dense TileConfig consolidation (Commit 3).

Same _C.so, same inputs, the dense WMMA op called TWICE — VLLM_W4A8_DENSE_TILED
unset (the standalone mmq_fp8_gemm_kernel_v5 / _v10) vs =1 (the gemm_tiled_kernel /
gemm_tiled_ashuffle_kernel re-expression over the DenseWmmaFp8 MMA policy) — and
asserts torch.equal (max|diff|==0). The env is read per-launch via getenv, so a
single process toggles os.environ between calls. This is the dense analogue of
test_tiled_bitexact.py (the MoE pre-gate) and the §2.4 acceptance criterion: the
tiled kernels become the served default ONLY if this passes AND perf is neutral.
"""
import os
import sys

import torch

try:
    import w4a8_fp8_wmma
except ImportError as e:
    print(f"FAIL: import error: {e}")
    sys.exit(1)

TILED_ENV = "VLLM_W4A8_DENSE_TILED"


def pack_uint4(w):  # (N,K) int8 -> (N,K//8) int32, 8 nibbles/int32 low-first
    N, K = w.shape
    w = w.to(torch.int32)
    packed = torch.zeros((N, K // 8), dtype=torch.int32, device=w.device)
    for i in range(8):
        packed |= (w[:, i::8] & 0xF) << (i * 4)
    return packed


def pack_zeros(z):  # (N,G) int8 -> (N//8,G) int32, N-packed nibble n%8
    N, G = z.shape
    z = z.to(torch.int32)
    packed = torch.zeros((N // 8, G), dtype=torch.int32, device=z.device)
    for i in range(8):
        packed |= (z[i::8, :] & 0xF) << (i * 4)
    return packed


def _run(kernel, x, w_packed, scales, zeros_packed, tiled):
    if tiled:
        os.environ[TILED_ENV] = "1"
    else:
        os.environ.pop(TILED_ENV, None)
    try:
        return w4a8_fp8_wmma.mmq_fp8_gemm(
            x, w_packed, scales, kernel=kernel, w_zeros=zeros_packed)
    finally:
        os.environ.pop(TILED_ENV, None)


def gate(kernel, M, N, K, group_size, asym, db=None):
    dev = torch.device("cuda")
    G = K // group_size
    x = torch.randn(M, K, dtype=torch.float16, device=dev)
    w_int4 = torch.randint(0, 16, (N, K), dtype=torch.int8, device=dev)
    w_packed = pack_uint4(w_int4)
    scales = torch.randn(N, G, dtype=torch.float16, device=dev).abs() * 0.01 + 0.001
    if asym:
        zeros = torch.randint(0, 16, (N, G), dtype=torch.int8, device=dev)
        zeros_packed = pack_zeros(zeros)
    else:
        zeros_packed = None

    if db is not None:
        os.environ["VLLM_W4A8_V10_DB"] = str(db)
    try:
        out_orig = _run(kernel, x, w_packed, scales, zeros_packed, tiled=False)
        out_tiled = _run(kernel, x, w_packed, scales, zeros_packed, tiled=True)
    finally:
        if db is not None:
            os.environ.pop("VLLM_W4A8_V10_DB", None)

    equal = torch.equal(out_orig, out_tiled)
    max_abs = (out_orig.float() - out_tiled.float()).abs().max().item()
    tag = "ASYM" if asym else "SYM "
    dbs = "" if db is None else f" db={db}"
    print(f"  {kernel:22s} {tag} M={M} N={N} K={K} g={group_size}{dbs}: "
          f"max|diff|={max_abs:.1e} -> {'PASS' if equal else 'FAIL (NOT bit-exact)'}")
    return equal


def main():
    if not torch.cuda.is_available():
        print("FAIL: no device"); sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    res = []

    # prefill_wmma (v5): A+B in LDS, any group size. Default tile <256,64>.
    print("=== prefill_wmma (v5) == gemm_tiled_kernel ===")
    for (M, N, K, g) in [(256, 512, 1024, 128), (256, 512, 1024, 32),
                         (128, 256, 512, 32), (64, 128, 256, 32),
                         (1, 512, 1024, 128), (300, 80, 256, 32)]:
        res.append(gate("prefill_wmma", M, N, K, g, asym=False))
        res.append(gate("prefill_wmma", M, N, K, g, asym=True))

    # prefill_wmma_ashuffle (v10): A-shuffle, B-only LDS, gs in {32,128}, DB flag.
    # Cover the served default (256x128, db=1) AND db=0 (the other DB instantiation).
    print("=== prefill_wmma_ashuffle (v10) == gemm_tiled_ashuffle_kernel ===")
    for db in (1, 0):
        for (M, N, K, g) in [(256, 512, 1024, 128), (256, 512, 1024, 32),
                             (512, 256, 2048, 128), (1, 512, 1024, 128),
                             (300, 128, 1024, 32)]:
            res.append(gate("prefill_wmma_ashuffle", M, N, K, g, asym=False, db=db))
            res.append(gate("prefill_wmma_ashuffle", M, N, K, g, asym=True, db=db))

    print("=" * 56)
    if all(res):
        print(f"ALL BIT-EXACT ({len(res)}) — gemm_tiled == v5/v10")
        sys.exit(0)
    print(f"FAIL: {sum(1 for r in res if not r)}/{len(res)} NOT bit-exact")
    sys.exit(1)


if __name__ == "__main__":
    main()
