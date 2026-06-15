#!/usr/bin/env python3
"""
Unified W4A8 dense-kernel benchmark — every version v0..v17 + stock Triton, eager & HIP-graph.

Purpose: one harness, one table, to decide (a) whether the served kernels (v5/v10/v11) need
changing, (b) whether the gated v6 (b128 double-K) earns its keep or should be dropped, and
(c) to independently re-confirm the v15/16/17 register-direct small-M wall under THIS harness
(eager+graph) rather than trusting the PIECE2 notes alone.

Grounding (existing measured numbers we are filling/confirming, not re-deriving):
  - PIECE2: v13/v15/v16/v17 all converge ~108 GB/s at M=16-32 (dense small-M wall = starved GPU,
    not the weight read). v17 (true W4A16) is the most competitive: wins M=8 (0.90x), M=48 (0.80x).
  - VALIDATION: v11 GEMV wins M=1 (1.75x @ N=5120,K=17408), loses M>=4.
  - v10 wins M>=128, 2-6x at M>=256; v5 the any-gs fallback; "only loss is M=4-64".
  - v6 bit-exact to v5 (only LDS load width differs), mid-M 512-2048 band, never benchmarked.

ABI notes (verified against bindings.cpp + kernel.hip):
  - v0..v14 take the standard (N, K//8) int32 pack; the launcher act-quants x internally for all.
  - v15/16/17 take a pre-repacked w_rep (N//16, K//16, 32) int32 (Marlin lane-order) + N.
    scales/zeros stay in the STANDARD (N, K//group) / (N//8, K//group) layout (kernel.hip:2209/2177).
  - All versions take fp16 x at the python boundary. Symmetric -> w_zeros = empty (implicit zp=8).

Run (combined image, GPU window):
  docker run --rm --device /dev/kfd --device /dev/dri --group-add video --ipc host \
    --security-opt seccomp=unconfined --security-opt label=disable --cap-add CAP_SYS_PTRACE \
    --shm-size 16g -e HIP_VISIBLE_DEVICES=0 -e ROCR_VISIBLE_DEVICES=0 \
    -v /home/pat/code/vllm-gfx1201/.triton-cache-combined:/root/.triton \
    -v /home/pat/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
    -v /home/pat/code/vllm-gfx1201/profiling/all-versions-bench:/probe \
    -v /home/pat/code/vllm-gfx1201/w4a8_fp8_wmma:/w4a8 \
    --entrypoint bash vllm22-w4a8:combined \
    -lc 'source /app/.venv/bin/activate && cd /w4a8 && exec python3 /probe/bench_all_versions.py'
"""
import os, sys, json, time
import torch

import w4a8_fp8_wmma  # noqa: F401  installed package w/ compiled _C
sys.path.append("/w4a8")
sys.path.append(os.getcwd())
from triton_w4a16_ref import triton_w4a16_gemm

OPS = torch.ops.w4a8_fp8_wmma
DEV = "cuda"
EMPTY_ZP = torch.empty(0, dtype=torch.int32, device=DEV)  # symmetric -> implicit zp=8

FULL  = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SMALL = [1, 2, 4, 8, 16, 32, 64]
MIDLG = [64, 128, 256, 512, 1024, 2048, 4096]
COARSE = [1, 16, 256]   # retire-candidate kernels: a few points to confirm domination, not large-M

# version registry: (label, kind, m_list)
#   kind: "triton" | ("std", ver) standard-pack op | ("wrep", opname) w_rep op
# v0 (scalar golden) is omitted from timing — it's the correctness reference, never a perf candidate
# (v5 serves as the in-script anchor).
VERSIONS = [
    ("triton",          ("triton",),       FULL),
    ("v1_rocwmma",      ("std", 1),        COARSE),
    ("v2_tiled",        ("std", 2),        COARSE),
    ("v4_pipe",         ("std", 4),        COARSE),
    ("v5_wmma",         ("std", 5),        FULL),          # WIRED (any-gs fallback)
    ("v6_b128",         ("std", 6),        FULL),          # GATED (mid-M band) -- keep/drop decision
    ("v7_tiled",        ("std", 7),        COARSE),
    ("v8_dbuf",         ("std", 8),        COARSE),
    ("v9_dbuf2",        ("std", 9),        COARSE),
    ("v10_ashuffle",    ("std", 10),       FULL),          # WIRED (large-M prefill)
    ("v11_gemv",        ("std", 11),       SMALL),         # WIRED (decode M<=2)
    ("v12_splitk",      ("std", 12),       SMALL),
    ("v13_regdirect",   ("std", 13),       [1, 4, 8, 16]), # 5-15x slow: cap hard
    ("v14_nsplit",      ("std", 14),       SMALL),
    ("v15_marlin",      ("wrep", "mmq_fp8_gemm_v15"),  SMALL),
    ("v16_f16wmma",     ("wrep", "mmq_fp8_gemm_v16"),  SMALL),
    ("v17_w4a16",       ("wrep", "mmq_w4a16_gemm_v17"), SMALL),
]

# Shapes: dense crossover-cache + PIECE/VALIDATION shapes. (N, K, group, label)
SHAPES = [
    (4096,  4096, 128, "dense-4096x4096-g128"),
    (4096,  4096,  32, "dense-4096x4096-g32"),
    (11008, 4096,  32, "ffn-up-11008x4096-g32"),
    (5120, 17408, 128, "largeK-5120x17408-g128"),   # v11 GEMV showcase shape
]
IT = int(os.environ.get("PROBE_IT", "50"))
WARMUP = int(os.environ.get("PROBE_WARMUP", "15"))


def pack_ours(w):  # (N,K) int4 -> (N, K//8) int32, K-major (standard ABI for v0..v14)
    N, K = w.shape
    wp = torch.zeros(N, K // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        wp |= (w[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return wp


def pack_triton(w):  # (N,K) int4 -> (K, N//8) int32 (stock Triton ABI)
    N, K = w.shape
    wt = w.t().contiguous()
    bq = torch.zeros(K, N // 8, dtype=torch.int32, device=w.device)
    for j in range(8):
        bq |= (wt[:, j::8].to(torch.int32) & 0xF) << (j * 4)
    return bq


def pack_wrep(w):
    """(N,K) int4 -> w_rep (N//16, K//16, 32) int32, Marlin lane-order for v15/16/17.
    Matches kernel.hip:2193: bword = w_rep[n_tile, k_tile, lane]; nibble jj (low-first) is the
    weight at col = 16*n_tile + (lane&15), k = 16*k_tile + (lane>>4)*8 + jj."""
    N, K = w.shape
    assert N % 16 == 0 and K % 16 == 0, f"v15-17 need N,K %16==0 (got {N},{K})"
    nt, kt = N // 16, K // 16
    wr = torch.zeros(nt, kt, 32, dtype=torch.int32, device=w.device)
    col_base = torch.arange(nt, device=w.device) * 16
    kt_base = torch.arange(kt, device=w.device) * 16
    for lane in range(32):
        cols = col_base + (lane & 15)            # (nt,)
        khalf = (lane >> 4) * 8
        wcols = w[cols]                          # (nt, K)
        for jj in range(8):
            ks = kt_base + khalf + jj            # (kt,)
            nib = (wcols[:, ks] & 0xF).to(torch.int32)   # (nt, kt)
            wr[:, :, lane] |= nib << (jj * 4)
    return wr


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
        return None
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / it


def make_fn(kind, x, packs):
    wp, bq, tsc, sc, wr, N, G = packs
    if kind[0] == "triton":
        return lambda: triton_w4a16_gemm(x, bq, tsc, None, G, 8)
    if kind[0] == "std":
        ver = kind[1]
        return lambda: OPS.mmq_fp8_gemm(x, wp, sc, EMPTY_ZP, ver)
    if kind[0] == "wrep":
        op = getattr(OPS, kind[1])
        return lambda: op(x, wr, sc, EMPTY_ZP, N)
    raise ValueError(kind)


def main():
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("no GPU", file=sys.stderr); sys.exit(1)
    print(f"device={torch.cuda.get_device_name(0)} it={IT} warmup={WARMUP}", flush=True)

    rows = []
    for (N, K, G, label) in SHAPES:
        w = torch.randint(0, 16, (N, K), dtype=torch.int32, device=DEV)
        wp = pack_ours(w)
        bq = pack_triton(w)
        sc = (torch.rand(N, K // G, dtype=torch.float16, device=DEV) * 0.02 + 0.001)
        tsc = sc.t().contiguous()
        wr = pack_wrep(w) if (N % 16 == 0 and K % 16 == 0) else None
        packs = (wp, bq, tsc, sc, wr, N, G)
        print(f"\n## {label}  N={N} K={K} G={G}", flush=True)

        # ---- correctness anchors ----
        xc = torch.randn(32, K, dtype=torch.float16, device=DEV) * 0.1
        ref = OPS.mmq_fp8_gemm(xc, wp, sc, EMPTY_ZP, 5).float()       # v5 fp8 anchor
        tri = triton_w4a16_gemm(xc, bq, tsc, None, G, 8).float()
        rel_tri = ((ref - tri).abs().mean() / (tri.abs().mean() + 1e-6)).item()
        print(f"   v5 vs triton mean-rel: {rel_tri:.3e}", flush=True)
        if wr is not None:
            v15 = OPS.mmq_fp8_gemm_v15(xc, wr, sc, EMPTY_ZP, N).float()
            rel_v15 = ((ref - v15).abs().max()).item()
            print(f"   v15(w_rep) vs v5 max-abs: {rel_v15:.3e}  "
                  f"(==0 -> prepack bit-correct)", flush=True)

        for (vlabel, kind, mlist) in VERSIONS:
            for M in mlist:
                x = torch.randn(M, K, dtype=torch.float16, device=DEV) * 0.1
                fn = make_fn(kind, x, packs)
                rec = {"shape": label, "N": N, "K": K, "G": G, "M": M, "ver": vlabel}
                try:
                    rec["eager_us"] = round(bench_eager(fn) * 1e6, 2)
                except Exception as ex:  # noqa: BLE001
                    rec["eager_us"] = None
                    rec["err"] = f"{type(ex).__name__}: {str(ex)[:120]}"
                    rows.append(rec); continue
                g = bench_graph(fn)
                rec["graph_us"] = round(g * 1e6, 2) if g is not None else None
                rows.append(rec)
        # compact per-shape print: winner vs triton at each M
        by_m = {}
        for r in rows:
            if r["shape"] != label or r.get("eager_us") is None:
                continue
            by_m.setdefault(r["M"], {})[r["ver"]] = r["eager_us"]
        for M in sorted(by_m):
            d = by_m[M]
            tri_us = d.get("triton")
            best = min(((v, u) for v, u in d.items() if v != "triton"), key=lambda kv: kv[1], default=None)
            print(f"   M={M:<5} triton={tri_us}  best_custom={best}", flush=True)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_all.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
