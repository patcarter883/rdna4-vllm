#!/usr/bin/env python3
# CPU-only sanity check for the wider FIXED Hadamard rotation (ParoQuant stage (a)).
#
# Proves, on toy tensors with NO model load and NO GPU:
#   1. The generalized FWHT over a span S (power of two) is ORTHONORMAL:  R . R^T = I.
#   2. The RXF cancellation identity holds for ANY matched span:
#        (X . R^T) . (R . W) == X . W   within fp error,
#      where the OFFLINE side pre-rotates the weight by R (rotate each consecutive
#      S input channels) and the RUNTIME side rotates the activation by the SAME R.
#      Because H_hat is SYMMETRIC and ORTHONORMAL, R^T == R and R.R == I, so the
#      rotation drops out of the matmul exactly as it does at span 32 today.
#   3. The span-32 case reproduces the existing _fwht32_rows numerics bit-for-bit,
#      so widening is strictly a superset (hadamard32 default is untouched).
#
# Run (no lease, CPU):
#   docker run --rm -v <worktree>/paroquant_rotation:/w --entrypoint bash \
#     tcclaviger/vllm22:dev -lc 'source /app/.venv/bin/activate && python /w/sanity_wider_rotation.py'

import torch

torch.manual_seed(0)


# ---- reference: the SHIPPED span-32 FWHT, copied verbatim from quantize_rxf.py ----
def _fwht32_rows_ref(x):
    orig = x.shape
    x = x.reshape(-1, 32).float()
    h = 1
    while h < 32:
        x = x.reshape(-1, 32 // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.reshape(-1, 32)
        h *= 2
    return (x * 0.1767766953).reshape(orig)


# ---- candidate: generalized span-S FWHT (S a power of two) ----
def fwht_rows(x, span):
    """In-place-style FWHT over the last axis of x[..., span], fp32 accumulate.
    log2(span) butterfly stages (same natural/Hadamard order as the span-32
    reference), then * 1/sqrt(span) to normalize. Span 32 with the 0.1767766953
    literal == the shipped _fwht32_rows."""
    assert span & (span - 1) == 0, "span must be a power of two"
    orig = x.shape
    x = x.reshape(-1, span).float()
    h = 1
    while h < span:
        x = x.reshape(-1, span // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.reshape(-1, span)
        h *= 2
    norm = 0.1767766953 if span == 32 else (1.0 / (span ** 0.5))
    return (x * norm).reshape(orig)


def rotate_weight(w, span):
    """Offline weight pre-rotation: FWHT over each consecutive `span` input chan."""
    N, K = w.shape
    return fwht_rows(w.reshape(N, K // span, span), span).reshape(N, K)


def rotate_activation(x, span):
    """Runtime activation rotation: SAME R, applied per consecutive `span` chan."""
    M, K = x.shape
    return fwht_rows(x.reshape(M, K // span, span), span).reshape(M, K)


def build_R(span):
    """Materialize R = H_hat (span x span) by rotating the identity rows."""
    return fwht_rows(torch.eye(span), span)


def main():
    print("=" * 70)
    print("ParoQuant stage (a): wider FIXED Hadamard rotation — CPU sanity")
    print("=" * 70)

    ok = True

    # --- 1. span-32 reproduces the shipped numerics bit-for-bit ---
    t = torch.randn(7, 4, 32)
    d = (fwht_rows(t, 32) - _fwht32_rows_ref(t)).abs().max().item()
    print(f"\n[1] span-32 vs shipped _fwht32_rows  max_abs_diff = {d:.3e}", end="  ")
    print("OK" if d == 0.0 else "FAIL"); ok &= (d == 0.0)

    # --- 2. orthonormality R.R^T = I and symmetry R = R^T, per span ---
    print("\n[2] orthonormality / symmetry of R = H_hat per span")
    for span in (32, 64, 128, 256, 512):
        R = build_R(span)
        I = torch.eye(span)
        e_orth = (R @ R.t() - I).abs().max().item()
        e_sym = (R - R.t()).abs().max().item()
        e_inv = (R @ R - I).abs().max().item()   # R is its own inverse
        good = e_orth < 1e-4 and e_sym < 1e-6 and e_inv < 1e-4
        print(f"    span={span:4d}  |RRᵀ-I|={e_orth:.2e}  |R-Rᵀ|={e_sym:.2e}  "
              f"|RR-I|={e_inv:.2e}  {'OK' if good else 'FAIL'}")
        ok &= good

    # --- 3. matmul cancellation: (X Rᵀ)(R W) == X W, any matched span ---
    print("\n[3] RXF cancellation  (X·Rᵀ)·(R·W) ≈ X·W  for matched span")
    M, K, N = 5, 512, 9
    X = torch.randn(M, K)
    W = torch.randn(N, K)          # weight stored [N, K]; contraction over K
    ref = X @ W.t()                # un-rotated reference output [M, N]
    for span in (32, 64, 128, 256, 512):
        Wr = rotate_weight(W, span)         # offline pre-rotated weight
        Xr = rotate_activation(X, span)     # runtime rotated activation
        out = Xr @ Wr.t()                   # the int8 GEMM sees rotated operands
        err = (out - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        good = rel < 1e-4
        print(f"    span={span:4d}  max_abs_err={err:.3e}  rel={rel:.3e}  "
              f"{'OK' if good else 'FAIL'}")
        ok &= good

    # --- 4. wider span genuinely Gaussianizes harder: outlier energy spreads
    #        across a bigger window → lower per-group (size-32) abs-max. ---
    print("\n[4] outlier suppression: per-32-group abs-max after wider rotation")
    row = torch.randn(1, 512) * 0.1
    row[0, 17] = 30.0   # a single fat outlier channel
    for span in (32, 128, 512):
        r = rotate_weight(row, span)
        g32max = r.reshape(-1, 32).abs().amax(dim=-1)
        print(f"    span={span:4d}  worst group-32 abs-max = {g32max.max():.3f}")

    print("\n" + "=" * 70)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
