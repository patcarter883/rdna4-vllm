# SPDX-License-Identifier: Apache-2.0
#
# CPU NumPy mirror of the GENERALIZED RXF runtime activation rotation
# (rxf_kernels._fwht / _rxf_rotate_quant_int8_kernel butterfly), proving stage
# (a) — wider fixed Hadamard span — is correct WITHOUT a GPU (Triton JIT needs a
# device, so we mirror the kernel's exact stage sequence in NumPy instead).
#
# Three guarantees, for spans {32, 64, 128, 512}:
#   (1) the runtime mirror == a reference Hadamard-S transform (1/sqrt(S) * H_S),
#   (2) it CANCELS with the offline quantize_rxf._fwht_rows(span=S):
#         (X . R^T) . (R . W) ~= X . W      (R orthonormal & symmetric => R^T=R)
#   (3) span==32 is BIT-IDENTICAL to the original fixed 5-stage FWHT-32.
#
# Run in the image venv (no GPU): paste the PASS output.

import numpy as np

# Pull the offline reference straight from the (untouched) offline module so the
# cancellation test uses the REAL offline butterfly, not a re-derivation.
import importlib.util
import os

_HERE = os.path.dirname(os.path.realpath(__file__))


def _load_offline_fwht():
    """The REAL offline _fwht_rows(x, span) from quantize_rxf.py — extracted by
    source (the module has heavy unrelated imports, e.g. model_registry, that
    aren't on the path here; the rotation function itself only needs torch)."""
    import ast
    import textwrap
    src = open(os.path.join(_HERE, "quantize_rxf.py")).read()
    tree = ast.parse(src)
    fn_src = None
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("GROUP",
                                                         "ROTATION_SPAN"):
                    try:
                        consts[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
        if isinstance(node, ast.FunctionDef) and node.name == "_fwht_rows":
            fn_src = ast.get_source_segment(src, node)
    assert fn_src is not None, "could not locate offline _fwht_rows"
    import torch
    ns = {"torch": torch,
          "GROUP": consts.get("GROUP", 32),
          "ROTATION_SPAN": consts.get("ROTATION_SPAN", 32)}
    exec(textwrap.dedent(fn_src), ns)
    return ns["_fwht_rows"]


# ---------------------------------------------------------------------------
# Runtime mirror — EXACTLY rxf_kernels._fwht_stage / the int8 kernel butterfly.
# ---------------------------------------------------------------------------
def _runtime_stage(x, ROWS, S, NG2H, H):
    """Mirror of rxf_kernels._fwht_stage: reshape (ROWS,NG2H,2,H), permute to
    (ROWS,NG2H,H,2), split lo/hi on the last axis, join (lo+hi, lo-hi), permute
    back, reshape (ROWS,S)."""
    t = x.reshape(ROWS, NG2H, 2, H)
    t = np.transpose(t, (0, 1, 3, 2))          # (ROWS, NG2H, H, 2)
    lo = t[..., 0]                             # (ROWS, NG2H, H)
    hi = t[..., 1]
    t = np.stack([lo + hi, lo - hi], axis=-1)  # (ROWS, NG2H, H, 2)
    t = np.transpose(t, (0, 1, 3, 2))          # (ROWS, NG2H, 2, H)
    return t.reshape(ROWS, S)


def runtime_fwht_rows(x, S):
    """Mirror of the generalized runtime FWHT applied per consecutive S channels
    of each row. x: [.., K] with K % S == 0. log2(S) stages H=1..S/2,
    NG2H = S//(2H), norm = 0.1767766953 (S==32) else 1/sqrt(S)."""
    orig = x.shape
    rows = x.reshape(-1, S).astype(np.float64)
    ROWS = rows.shape[0]
    H = 1
    while H < S:
        rows = _runtime_stage(rows, ROWS, S, S // (2 * H), H)
        H *= 2
    norm = 0.1767766953 if S == 32 else (1.0 / np.sqrt(S))
    return (rows * norm).reshape(orig)


def _original_fwht32_rows(x):
    """The ORIGINAL fixed 5-stage FWHT-32 mirror (pre-generalization), to prove
    span==32 bit-identity. Hard-coded stage list + 0.1767766953, exactly as the
    shipped _fwht32_stage calls were."""
    orig = x.shape
    rows = x.reshape(-1, 32).astype(np.float64)
    ROWS = rows.shape[0]
    for NG2H, H in [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]:
        rows = _runtime_stage(rows, ROWS, 32, NG2H, H)
    return (rows * 0.1767766953).reshape(orig)


# ---------------------------------------------------------------------------
# Reference Hadamard-S (normalized Sylvester) for the equality check.
# ---------------------------------------------------------------------------
def hadamard_matrix(S):
    H = np.array([[1.0]])
    while H.shape[0] < S:
        H = np.block([[H, H], [H, -H]])
    return H


def ref_fwht_rows(x, S):
    """Reference: block-diagonal (1/sqrt(S) H_S) over each consecutive S
    channels. The runtime butterfly is the NATURAL (Hadamard) order, which is
    exactly the Sylvester matrix above (no bit-reversal), so this is the ground
    truth the runtime must equal."""
    Hn = hadamard_matrix(S) / np.sqrt(S)
    orig = x.shape
    blocks = x.reshape(-1, S).astype(np.float64)
    return (blocks @ Hn.T).reshape(orig)


def main():
    rng = np.random.default_rng(0)
    offline_fwht = _load_offline_fwht()
    import torch

    spans = [32, 64, 128, 512]
    N, M = 96, 40                # weight rows, activation rows
    all_ok = True

    print("=== RXF runtime wider-span rotation — CPU mirror sanity ===\n")

    for S in spans:
        K = S * 6                # 6 spans wide => crosses block boundaries
        X = rng.standard_normal((M, K))
        W = rng.standard_normal((N, K))

        # (1) runtime mirror == reference Hadamard-S
        r_run = runtime_fwht_rows(X, S)
        r_ref = ref_fwht_rows(X, S)
        e_ref = np.max(np.abs(r_run - r_ref)) / (np.max(np.abs(r_ref)) + 1e-30)

        # Orthonormality / symmetry of the implied R (one S-block).
        I = np.eye(S)
        R = runtime_fwht_rows(I, S)            # R applied row-wise to I => R^T
        ortho = np.max(np.abs(R @ R.T - I))
        symm = np.max(np.abs(R - R.T))

        # (2) cancellation with the OFFLINE _fwht_rows(span=S):
        #     activation rotated by R (runtime), weight rotated by R (offline);
        #     R symmetric+orthonormal => (X R)(R W)^T = X W^T.
        Xr = runtime_fwht_rows(X, S)
        Wr = offline_fwht(
            torch.from_numpy(W.reshape(N, K // S, S)), S
        ).reshape(N, K).double().numpy()
        lhs = Xr @ Wr.T
        rhs = X @ W.T
        e_cancel = np.max(np.abs(lhs - rhs)) / (np.max(np.abs(rhs)) + 1e-30)

        # fp64 control: rotate the weight with the SAME runtime mirror (fp64)
        # instead of the fp32 offline — isolates that the ~1e-7 above is purely
        # the offline fp32 accumulate, not a span/ordering mismatch.
        Wr64 = runtime_fwht_rows(W.reshape(N, K // S, S), S).reshape(N, K)
        e_cancel64 = (np.max(np.abs((Xr @ Wr64.T) - rhs))
                      / (np.max(np.abs(rhs)) + 1e-30))

        # mirror==ref: ~1e-15 for S!=32; ~2e-11 at S==32 because the runtime
        # uses the literal 0.1767766953 (= 1/sqrt(32) to 10 digits) — the
        # bit-identity guard literal, not a logic gap.
        # cancel: offline _fwht_rows runs in fp32 (.float()), so the round-trip
        # carries fp32 roundoff (~1e-7) — same level as sanity_wider_rotation.py
        # (rel <= 4e-7). The R-matrix tests above are exact, so this is purely
        # the offline fp32 accumulate, not a cancellation error.
        # fp64 control floor: exact (1e-12) for S!=32; at S==32 the literal
        # 0.1767766953 (1/sqrt(32) truncated to 10 digits) sets a ~4e-11 floor —
        # the intentional bit-identity guard, identical to the mirror==ref gap.
        cancel64_tol = 1e-12 if S != 32 else 1e-9
        ok = (e_ref < 1e-9 and ortho < 1e-9 and symm < 1e-12
              and e_cancel < 4e-7 and e_cancel64 < cancel64_tol)
        all_ok &= ok
        print(f"S={S:4d}  K={K:5d}  "
              f"mirror==ref rel={e_ref:.2e}  "
              f"R.R^T-I={ortho:.2e}  R-R^T={symm:.2e}  "
              f"cancel rel={e_cancel:.2e} (offline fp32)  "
              f"cancel64 rel={e_cancel64:.2e} (fp64)  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # (3) span==32 bit-identity vs the ORIGINAL fixed 5-stage FWHT-32.
    Xi = rng.standard_normal((M, 32 * 6))
    e_id = np.max(np.abs(
        runtime_fwht_rows(Xi, 32) - _original_fwht32_rows(Xi)))
    id_ok = e_id == 0.0
    all_ok &= id_ok
    print(f"\nspan==32 bit-identity vs original fixed 5-stage: "
          f"abs-diff={e_id:.1e}  [{'PASS' if id_ok else 'FAIL'}]")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
