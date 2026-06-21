# SPDX-License-Identifier: Apache-2.0
#
# CPU sanity for the ACTIVATION-CONDITIONING Givens fit (ParoQuant stage c, the
# STRATEGIC PIVOT) — proving the new objective WITHOUT a GPU. Stage (b) (data-blind
# weight-quant MSE) was a measured PPL null: Hadamard is already near-optimal for
# WEIGHT incoherence. The pivot (cf. DFlash) is to fit R to the REAL per-token int8
# ACTIVATION-quant error instead — flatten activation outlier spikes before the
# int8 cast. This test exercises that fit (score="activation") and checks:
#   (1) the fitted R is ORTHONORMAL (R·Rᵀ = I) — cancellation precondition,
#   (2) CANCELLATION still holds: (R x)·(R w) = x·w (the rotation is fit on
#       activations but R still cancels through the real GEMM — orthonormal),
#   (3) the activation objective MOVES: per-block int8 act-MSE fitted <= Hadamard
#       (GUARANTEED by the Hadamard-init / commit-only-improving descent),
#   (4) the FAITHFUL per-TOKEN metric (full-row absmax scale, the real runtime)
#       is reported — fitted should be <= Hadamard on structured activation
#       outliers (a surrogate, not guaranteed; the served PPL A/B is the arbiter).
#
# Offline functions are lifted from the REAL quantize_rxf.py by source so the test
# exercises the shipped code, not a re-derivation.
#
# Run in the image venv (no GPU):
#   docker run --rm -v <pq>:/work --entrypoint bash vllm22-w4a8:combined \
#     -lc 'source /app/.venv/bin/activate && python /work/sanity_givens_activation.py'

import ast
import math
import os
import textwrap

import torch

_HERE = os.path.dirname(os.path.realpath(__file__))

_WANTED = [
    "_fwht_rows", "_apply_rotation_rows", "_givens_quant_mse",
    "_rotated_importance", "_act_int8_quant_mse", "fit_givens_rotation",
]


def _load_offline():
    src = open(os.path.join(_HERE, "quantize_rxf.py")).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "math": math, "GROUP": 32}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("GROUP", "ROTATION_SPAN"):
                    try:
                        ns[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    want = set(_WANTED)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            exec(textwrap.dedent(ast.get_source_segment(src, node)), ns)
            want.discard(node.name)
    assert not want, f"could not locate offline functions: {want}"
    return ns


_NL = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.float32)


def _hadamard(S):
    H = torch.tensor([[1.0]])
    while H.shape[0] < S:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(S)


def runtime_givens_rotate(x, R):
    """Mirror of rxf_kernels.invoke_rxf_givens_rotate: block-diagonal x @ R.T."""
    M, K = x.shape
    S = R.shape[0]
    return (x.reshape(M, K // S, S).float() @ R.float().t()).reshape(M, K)


def _act_int8_row_mse(X, R):
    """FAITHFUL per-TOKEN int8 activation-quant MSE — the REAL runtime
    (rxf_kernels._rxf_rotate_quant_int8_kernel): rotate the row by block-diagonal
    R, scale = absmax(full row)/127, round to int8, dequant, mean squared error.
    X: [M, K]. Returns scalar."""
    Xr = runtime_givens_rotate(X, R)                          # [M, K]
    absmax = Xr.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = absmax / 127.0
    deq = torch.round(Xr / scale).clamp_(-127, 127) * scale
    return ((Xr - deq) ** 2).mean().item()


def _outlier_activation(M, K, span, n_spikes=5, seed=0):
    """Calibration-like activations with a few large outlier channels AND a
    correlated heavy-tailed bulk, so a data-blind Hadamard's uniform spreading is
    NOT optimal and a data-aware fit can do better."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(M, K, generator=g)
    spikes = torch.randperm(K, generator=g)[:n_spikes]
    # token-dependent spike magnitude (real outliers are not constant)
    mag = 12.0 + 18.0 * torch.rand(M, n_spikes, generator=g)
    X[:, spikes] *= mag
    return X, spikes


def main():
    off = _load_offline()
    fit = off["fit_givens_rotation"]
    apply_rows = off["_apply_rotation_rows"]
    act_mse = off["_act_int8_quant_mse"]

    torch.manual_seed(0)
    span = 32
    K = span * 8                  # 8 rotation blocks wide
    M = 1024                      # activation rows (tokens)
    N = 256                       # weight rows (for the cancellation check)
    all_ok = True

    print("=== RXF activation-conditioning Givens fit — CPU sanity ===\n")

    X, spikes = _outlier_activation(M, K, span, seed=1)
    W = torch.randn(N, K)         # arbitrary weight for cancellation

    blocks = X.reshape(-1, span).float()     # activation blocks to fit on
    H = _hadamard(span)

    # ---- fit R on the ACTIVATION objective ----
    R_c = fit(blocks, _NL, span=span, score="activation", seed=0)

    # (1) orthonormality
    I = torch.eye(span)
    ortho = (R_c @ R_c.t() - I).abs().max().item()
    ok1 = ortho < 1e-5
    all_ok &= ok1
    print(f"[orthonormal]   R·Rᵀ−I = {ortho:.2e}  [{'PASS' if ok1 else 'FAIL'}]")

    # (2) cancellation through the real GEMM: (R x)·(R w) = x·w
    Xr = runtime_givens_rotate(X, R_c)
    Wr = apply_rows(W.reshape(N, K // span, span), R_c).reshape(N, K)
    lhs = Xr @ Wr.t()
    rhs = X @ W.t()
    cancel = (lhs - rhs).abs().max().item() / (rhs.abs().max().item() + 1e-30)
    ok2 = cancel < 1e-4
    all_ok &= ok2
    print(f"[cancellation]  (Rx)·(Rw)=x·w rel = {cancel:.2e}  "
          f"[{'PASS' if ok2 else 'FAIL'}]")

    # (3) per-BLOCK int8 activation MSE — the fitted objective (guaranteed <= H)
    mse_none_b = act_mse(blocks)
    mse_had_b = act_mse(apply_rows(blocks, H))
    mse_fit_b = act_mse(apply_rows(blocks, R_c))
    print(f"\nper-block int8 activation-quant MSE (the fit objective):")
    print(f"  no-rotate        {mse_none_b:.4e}")
    print(f"  fixed Hadamard   {mse_had_b:.4e}  "
          f"({mse_none_b/mse_had_b:.2f}x vs none)")
    print(f"  fitted Givens-c  {mse_fit_b:.4e}  "
          f"({mse_had_b/mse_fit_b:.3f}x vs Hadamard)")
    ok3 = mse_fit_b <= mse_had_b * 1.001     # commit-only-improving ⇒ <= Hadamard
    all_ok &= ok3
    print(f"  fitted <= Hadamard (guaranteed): [{'PASS' if ok3 else 'FAIL'}]")

    # (4) FAITHFUL per-TOKEN int8 MSE (real runtime: full-row absmax scale)
    mse_none_r = _act_int8_row_mse(X, I)
    mse_had_r = _act_int8_row_mse(X, H)
    mse_fit_r = _act_int8_row_mse(X, R_c)
    print(f"\nper-TOKEN int8 activation-quant MSE (REAL runtime, full-row scale):")
    print(f"  no-rotate        {mse_none_r:.4e}")
    print(f"  fixed Hadamard   {mse_had_r:.4e}  "
          f"({mse_none_r/mse_had_r:.2f}x vs none)")
    print(f"  fitted Givens-c  {mse_fit_r:.4e}  "
          f"({mse_had_r/mse_fit_r:.3f}x vs Hadamard)")
    # per-token is a SURROGATE target of the per-block fit; report, soft-gate.
    ok4 = mse_fit_r <= mse_had_r * 1.05
    all_ok &= ok4
    print(f"  fitted <~ Hadamard (surrogate, informative): "
          f"[{'PASS' if ok4 else 'WARN'}]")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
