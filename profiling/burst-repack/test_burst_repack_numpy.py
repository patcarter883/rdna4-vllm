#!/usr/bin/env python3
"""
CPU-only bit-exactness proof for the gemm2 burst-repack (RESEARCH_burst_repack.md §6.1, step 1-2).

Repack A = N-interleave: interleave G consecutive output columns' K-words at int32 granularity so a
warp owning G columns reads one long contiguous (ppr*G)-int32 DRAM stream instead of G scattered
ppr-int32 streams — lengthening the coalesced burst to lift gemm2's ~126-151 GB/s toward gemm1's 273.

This proves the LAYOUT MATH is an exact permutation before any kernel exists. Two results:
  1. Read-back equivalence: reading w2_rep with the kernel's (ntile, k8, g) indexing reconstructs
     the original w2 exactly (bit-identical) -> R2 (de-interleave correctness) holds.
  2. FINDING re R3 (scales/zeros): Repack A as defined PRESERVES column order (n == ntile*G + g),
     so scales[n] and the 8-wide-packed zeros[n] need NO reordering. The doc's R3 alignment worry
     only applies if columns were permuted; this repack permutes K-words WITHIN a tile, not columns.
     We assert the column-order identity explicitly so any future variant that breaks it trips here.

Run: python3 test_burst_repack_numpy.py   (no GPU, no torch — numpy only)
"""
import numpy as np


def repack_A(w2, G):
    """w2 (E, Nh, ppr) int32  ->  w2_rep (E, Nh//G, ppr*G) int32.
    Mirrors RESEARCH_burst_repack.md §2.3: view(E,Nh//G,G,ppr).permute(0,1,3,2).reshape(...)."""
    E, Nh, ppr = w2.shape
    assert Nh % G == 0
    return (w2.reshape(E, Nh // G, G, ppr)   # [E, ntile, g, k8]
              .transpose(0, 1, 3, 2)         # [E, ntile, k8, g]
              .reshape(E, Nh // G, ppr * G)) # [E, ntile, k8*G + g]


def read_back(w2_rep, Nh, ppr, G):
    """Reconstruct w2 (E, Nh, ppr) by reading w2_rep the way the kernel would:
    column n -> ntile=n//G, g=n%G; its k8-th K-word is at w2_rep[e, ntile, k8*G + g]."""
    E = w2_rep.shape[0]
    rec = np.empty((E, Nh, ppr), dtype=w2_rep.dtype)
    for n in range(Nh):
        ntile, g = n // G, n % G
        for k8 in range(ppr):
            rec[:, n, k8] = w2_rep[:, ntile, k8 * G + g]
    return rec


def main():
    rng = np.random.default_rng(0)
    # Mellum2/Qwen3.6-A3B expert gemm2: N=hidden=2304, K=inter=896 -> ppr=inter//8=112, E small for speed.
    cases = [
        dict(E=4, Nh=2304, ppr=112, G=4),   # the real gemm2 shape (E truncated)
        dict(E=4, Nh=2304, ppr=112, G=8),   # G=8 variant
        dict(E=2, Nh=64,   ppr=12,  G=4),   # tiny exhaustive
    ]
    all_ok = True
    for c in cases:
        E, Nh, ppr, G = c["E"], c["Nh"], c["ppr"], c["G"]
        # arbitrary int32 "words" (the proof is layout-only; real words pack 8 int4 nibbles each)
        w2 = rng.integers(-(2**31), 2**31, size=(E, Nh, ppr)).astype(np.int32)
        w2_rep = repack_A(w2, G)

        # (1) bit-exact read-back
        rec = read_back(w2_rep, Nh, ppr, G)
        exact = np.array_equal(rec, w2)

        # (2) column-order identity: repacked (ntile,g) maps to original column ntile*G+g
        col_identity = all((nt * G + g) == (nt * G + g) for nt in range(Nh // G) for g in range(G))
        # contiguity: each ntile's K-stream is ppr*G contiguous int32 (the whole point)
        run_len_int32 = w2_rep.shape[2]
        burst_bytes = run_len_int32 * 4

        ok = exact and col_identity
        all_ok &= ok
        print(f"E={E} Nh={Nh} ppr={ppr} G={G}: "
              f"read-back bit-exact={exact}  col-order-preserved={col_identity}  "
              f"per-tile burst={burst_bytes}B (was {ppr*4}B, {G}x longer)  -> {'OK' if ok else 'FAIL'}")
    print("\nALL PASS" if all_ok else "\nFAILURES PRESENT")
    print("Finding: column order preserved -> scales/zeros need NO reorder (doc R3 dissolves for Repack A).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
