#!/usr/bin/env python3
"""
Layout-share probe — Phase 1 (pure measurement, no new kernel, no image rebuild).

Question under test (proposal "(c)"): keep ONE int4 weight layout in VRAM (the
Triton (K, N//8) packing), serve small-M decode with native Triton int4->fp16 for
free, and run large-M prefill on the HIP fp8-WMMA kernel reading that same layout
via an on-the-fly translation. Does that pencil out?

This script does NOT build the fused-swizzle kernel (that's the expensive thing we're
gating). It measures the components we already have, eager AND under HIP-graph capture:

  triton      : stock Triton int4->fp16, reads (K, N//8)              [small-M reference winner]
  hip_v5      : HIP fp8-WMMA, native (N, K//8), any-M fallback
  hip_v10     : HIP fp8-WMMA, native (N, K//8), large-M prefill kernel [large-M reference winner]
  repack_tax  : GPU cost to translate (K,N//8) -> (N,K//8)            [UPPER BOUND on the share tax]
  hip_from_triton : repack_tax + hip_v10 in one shot                  [naive per-call layout share]

Decision framing (per advisor):
  * HEADLINE: does hip_v10 still beat triton at large M *under graph capture*? If not,
    (c) is moot — run Triton everywhere on one layout, never write the fused kernel.
  * repack_tax is a CONFIRM-GO-ONLY proxy: it moves strictly MORE HBM traffic than a
    fused in-register swizzle would. A SMALL tax green-lights the fused kernel; a LARGE
    tax is INCONCLUSIVE (only proves the naive separate-pass is bad), never fatal.
  * hip_from_triton must be ~bit-identical to hip_v10 on native weights. If it isn't,
    that's a packing/scale-order bug — itself a useful signal.

Run (combined image, GPU window required):
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add CAP_SYS_PTRACE \
    --shm-size 16g \
    -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v /home/pat/code/vllm-gfx1201/profiling/layout-share-probe:/probe \
    -v /home/pat/code/vllm-gfx1201/w4a8_fp8_wmma:/w4a8 \
    --entrypoint bash vllm22-w4a8:combined \
    -lc 'source /app/.venv/bin/activate && cd /w4a8 && exec python3 /probe/bench_layout_share.py'
"""
import os, sys, json, time
import torch

# Import the INSTALLED package first (it has the compiled _C.so). Do NOT put the source
# tree (/w4a8) ahead of it on sys.path or it shadows the build with a _C-less source pkg.
import w4a8_fp8_wmma  # noqa: F401  (registers torch.ops.w4a8_fp8_wmma.*)
# triton_w4a16_ref is a loose module living only in the w4a8 source dir — add it at the END.
sys.path.append("/w4a8")
sys.path.append(os.getcwd())
from triton_w4a16_ref import triton_w4a16_gemm

OP = torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm
DEV = "cuda"
EMPTY_ZP = torch.empty(0, dtype=torch.int32, device=DEV)  # symmetric -> implicit zp=8

# (N, K, group_size, label). Dense shapes have proven v10 crossovers in crossover_cache.json;
# the moe-* shapes are the 35B's actually-quantized expert GEMMs (gs=32).
SHAPES = [
    (4096,  4096, 128, "dense-4096x4096-g128"),
    (11008, 4096, 128, "ffn-up-11008x4096-g128"),
    (4096, 11008, 128, "ffn-down-4096x11008-g128"),
    (14336, 4096, 128, "ffn-up-14336x4096-g128"),
    (1792,  2304,  32, "moe-gemm1-1792x2304-g32"),
    (2304,   896,  32, "moe-gemm2-2304x896-g32"),
]
M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
IT = int(os.environ.get("PROBE_IT", "50"))
WARMUP = int(os.environ.get("PROBE_WARMUP", "20"))


def pack_ours(w):  # (N,K) int4 -> (N, K//8) int32, low nibble first along K (HIP native)
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_triton(w):  # (N,K) int4 -> b_q (K, N//8) int32, 8 N-values per int32 (Triton native)
    N, K = w.shape
    wt = w.t().contiguous()
    bq = torch.zeros(K, N // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        bq |= (wt[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return bq


def repack_triton_to_ours(bq, N, K):
    """UPPER-BOUND tax: translate Triton (K,N//8) -> HIP (N,K//8) entirely on-GPU.
    A separate pass that reads+writes the full weight (~2x weight HBM traffic) — strictly
    more than a fused in-register swizzle would move. Vectorized so it's a fair upper bound."""
    wt = torch.empty(K, N, dtype=torch.int32, device=bq.device)  # (K,N) nibbles
    for j in range(8):
        wt[:, j::8] = (bq >> (j * 4)) & 0xF
    w = wt.t().contiguous()  # (N,K)
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=bq.device)
    for j in range(8):
        wp |= (w[:, j::8] & 0xF) << (j * 4)
    return wp


def bench_eager(fn, it=IT, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def bench_graph(fn, it=IT, warmup=WARMUP):
    """Capture fn() into a HIP graph and time replay. Returns None if capture fails
    (e.g. an op that can't be captured) so the run keeps going."""
    # Lock in Triton autotune / JIT in eager first — autotune syncs, which is illegal mid-capture.
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
    except Exception as e:  # noqa: BLE001
        return None, f"capture_failed: {type(e).__name__}: {e}"
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it, "ok"


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA/HIP not available", file=sys.stderr)
        sys.exit(1)
    print(f"device={torch.cuda.get_device_name(0)}  it={IT} warmup={WARMUP}", flush=True)

    rows = []
    for (N, K, G, label) in SHAPES:
        # symmetric int4 weights in [0,15]; reference subtracts zp=8.
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
        wp_native = pack_ours(w)                 # (N, K//8) HIP native
        bq = pack_triton(w)                      # (K, N//8) Triton native
        sc = (torch.rand(N, K // G, dtype=torch.float16, device=DEV) * 0.02 + 0.001)
        tsc = sc.t().contiguous()                # (K//G, N) for Triton

        # ---- correctness: same weights, both layouts, must agree (advisor pt 3) ----
        x_chk = torch.randn(64, K, dtype=torch.float16, device=DEV) * 0.1
        wp_from_triton = repack_triton_to_ours(bq, N, K)
        repack_ok = bool(torch.equal(wp_from_triton, wp_native))
        out_native = OP(x_chk, wp_native, sc, EMPTY_ZP, 5).float()
        out_shared = OP(x_chk, wp_from_triton, sc, EMPTY_ZP, 5).float()
        bit_identical = bool(torch.equal(out_native, out_shared))
        max_abs = (out_native - out_shared).abs().max().item()
        tri = triton_w4a16_gemm(x_chk, bq, tsc, None, G, 8).float()
        # hip vs triton agreement (fp8 act-quant vs fp16 -> expect close, not exact)
        denom = tri.abs().mean().item() + 1e-6
        hip_vs_tri_rel = (out_native - tri).abs().mean().item() / denom

        # ---- repack tax: M-INDEPENDENT (weights don't depend on batch) -> measure ONCE ----
        # NOTE: this torch-strided repack is a *loose* upper bound, ~100x above the HBM floor
        # (launch/transpose-bound, many small kernels). The fused in-register swizzle (c) would
        # do a separate pass at all. Report it next to the analytic floor for honest framing.
        tax_us = bench_eager(lambda: repack_triton_to_ours(bq, N, K), it=10, warmup=3) * 1e6
        weight_bytes = N * K // 2  # int4
        floor_us = (2 * weight_bytes) / 1.5e12 * 1e6  # read+write @ ~1.5 TB/s
        print(f"\n## {label}  N={N} K={K} G={G}", flush=True)
        print(f"   repack(K,N//8)->(N,K//8) reproduces native packing: {repack_ok}", flush=True)
        print(f"   hip(shared-layout) bit-identical to hip(native): {bit_identical}  "
              f"(max_abs={max_abs:.2e})", flush=True)
        print(f"   hip(native) vs triton mean-rel-err: {hip_vs_tri_rel:.3e}", flush=True)
        print(f"   repack tax: torch-strided={tax_us:.0f}us (loose UB)  "
              f"HBM-floor~={floor_us:.1f}us  [M-independent]", flush=True)

        for M in M_LIST:
            x = torch.randn(M, K, dtype=torch.float16, device=DEV) * 0.1
            cands = {
                "triton":  lambda: triton_w4a16_gemm(x, bq, tsc, None, G, 8),
                "hip_v5":  lambda: OP(x, wp_native, sc, EMPTY_ZP, 5),
                "hip_v10": lambda: OP(x, wp_native, sc, EMPTY_ZP, 10),
            }
            rec = {"label": label, "N": N, "K": K, "G": G, "M": M,
                   "repack_tax_torch_us": round(tax_us, 1), "repack_floor_us": round(floor_us, 2)}
            for name, fn in cands.items():
                try:
                    e = bench_eager(fn) * 1e6  # us
                except Exception as ex:  # noqa: BLE001
                    rec[f"{name}_eager_us"] = None
                    rec[f"{name}_err"] = f"{type(ex).__name__}: {ex}"
                    continue
                rec[f"{name}_eager_us"] = round(e, 2)
                gres = bench_graph(fn)
                if gres[0] is None:
                    rec[f"{name}_graph_us"] = None
                    rec[f"{name}_graph_note"] = gres[1]
                else:
                    rec[f"{name}_graph_us"] = round(gres[0] * 1e6, 2)
            rec["repack_ok"] = repack_ok
            rec["bit_identical"] = bit_identical
            rows.append(rec)

            te, tg = rec.get("triton_eager_us"), rec.get("triton_graph_us")
            v10e, v10g = rec.get("hip_v10_eager_us"), rec.get("hip_v10_graph_us")
            v5e = rec.get("hip_v5_eager_us")
            win = "triton" if (te and v10e and te < v10e) else "v10"
            print(f"   M={M:<5} | triton e/g={te}/{tg}  v5 e={v5e}  "
                  f"v10 e/g={v10e}/{v10g}  -> {win}", flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out_path}  ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
