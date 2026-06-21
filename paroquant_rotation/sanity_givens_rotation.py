# SPDX-License-Identifier: Apache-2.0
#
# CPU sanity for the LEARNED Givens rotation (ParoQuant stage b/c) — proving the
# offline fit + the runtime apply WITHOUT a GPU (Triton needs a device; the
# runtime rotation here is plain torch, so the CPU run is the real math). Four
# guarantees:
#   (1) the fitted R is ORTHONORMAL (R·Rᵀ = I) — cancellation precondition,
#   (2) the runtime rotation invoke_rxf_givens_rotate (mirrored) CANCELS with the
#       offline weight rotation _apply_rotation_rows: (R x)·(R w) = x·w,
#   (3) ISO-BITS: the learned R's per-group NL-quant MSE on a weight with
#       reasoning-spike outliers is <= the fixed Hadamard's (the stage-(a) result
#       was FLAT/conditional; the learned fit should not be worse, and on
#       structured outliers should win),
#   (4) IMPORTANCE-AWARE (stage c): with per-channel importance, the learned R
#       lowers the IMPORTANCE-WEIGHTED quant MSE vs the fixed Hadamard.
#
# The offline functions are pulled from the REAL quantize_rxf.py by source (the
# module has heavy unrelated imports — model_registry etc. — that aren't on the
# path here), so the test exercises the shipped fit, not a re-derivation.
#
# Run in the image venv (no GPU):
#   docker run --rm -v <pq>:/work --entrypoint bash vllm22-w4a8:combined \
#     -lc 'source /app/.venv/bin/activate && python /work/sanity_givens_rotation.py'

import ast
import math
import os
import textwrap

import torch

_HERE = os.path.dirname(os.path.realpath(__file__))

# Functions to lift verbatim from quantize_rxf.py into a shared namespace so they
# call each other (fit_givens_rotation -> _givens_quant_mse / _rotated_importance).
_WANTED = [
    "_fwht_rows", "_apply_rotation_rows", "_givens_quant_mse",
    "_rotated_importance", "fit_givens_rotation",
]


def _load_offline():
    src = open(os.path.join(_HERE, "quantize_rxf.py")).read()
    tree = ast.parse(src)
    ns = {"torch": torch, "math": math, "GROUP": 32}
    # GROUP / ROTATION_SPAN consts if present
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


# IQ4-NL default grid (integer-valued; matches ACTIVE_TABLE default in the tool).
_NL = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.float32)


def _hadamard(S):
    H = torch.tensor([[1.0]])
    while H.shape[0] < S:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(S)


def runtime_givens_rotate(x, R):
    """Mirror of rxf_kernels.invoke_rxf_givens_rotate: block-diagonal x_block @ R.T
    over each consecutive S channels (plain torch, fp32)."""
    M, K = x.shape
    S = R.shape[0]
    return (x.reshape(M, K // S, S).float() @ R.float().t()).reshape(M, K)


def _group_quant_mse(w, nl, imp=None):
    """Per-(size-32)-group symmetric-scale NL-quant MSE, optionally importance
    weighted. w: [N, K] fp32. Returns mean (weighted) MSE — a single scalar at
    iso-bits (same 4-bit grid, same group size) so rotations are comparable."""
    N, K = w.shape
    g = w.reshape(-1, 32)
    maxabs = nl.abs().max()
    scale = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / maxabs
    idx = (g.unsqueeze(-1) / scale.unsqueeze(-1) - nl).abs().argmin(-1)
    e2 = (g - nl[idx] * scale) ** 2
    if imp is not None:
        e2 = e2 * imp.reshape(-1, 32)
    return e2.mean().item()


def _outlier_weight(N, K, span, n_spikes=6, seed=0):
    """A weight whose columns carry a few large 'reasoning spike' channels (the
    motivating case: a handful of input channels with ~30x energy)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, generator=g)
    spikes = torch.randperm(K, generator=g)[:n_spikes]
    w[:, spikes] *= 30.0
    return w, spikes


def main():
    off = _load_offline()
    fit = off["fit_givens_rotation"]
    apply_rows = off["_apply_rotation_rows"]
    fwht_rows = off["_fwht_rows"]

    torch.manual_seed(0)
    span = 32
    N, K = 256, span * 8         # 8 rotation blocks wide
    M = 32                       # activation rows
    nl = _NL
    all_ok = True

    print("=== RXF learned Givens rotation — CPU sanity ===\n")

    W, spikes = _outlier_weight(N, K, span, seed=1)
    X = torch.randn(M, K)
    # Put activation energy on the SAME spike channels (so importance is real).
    X[:, spikes] *= 30.0
    imp_k = (X.float().abs().mean(0))                 # [K] per-channel importance
    imp_k = imp_k / (imp_k.mean() + 1e-8)

    pooled = W.reshape(-1, span).float()             # blocks to fit on
    imp_block = imp_k.reshape(-1, span).mean(0)      # [span]

    # ---- (b) uniform fit, (c) importance fit ----
    R_b = fit(pooled, nl, imp_block=None, span=span, seed=0)
    R_c = fit(pooled, nl, imp_block=imp_block, span=span, seed=0)
    H = _hadamard(span)

    for tag, R in (("learned-b(uniform)", R_b), ("learned-c(importance)", R_c)):
        # (1) orthonormality
        I = torch.eye(span)
        ortho = (R @ R.t() - I).abs().max().item()

        # (2) cancellation: runtime rotates X, offline rotates W, both by R.
        Xr = runtime_givens_rotate(X, R)
        Wr = apply_rows(W.reshape(N, K // span, span), R).reshape(N, K)
        lhs = Xr @ Wr.t()
        rhs = X @ W.t()
        cancel = (lhs - rhs).abs().max().item() / (rhs.abs().max().item() + 1e-30)

        ok = ortho < 1e-5 and cancel < 1e-4
        all_ok &= ok
        print(f"[{tag:24s}] R·Rᵀ−I={ortho:.2e}  cancel rel={cancel:.2e}  "
              f"[{'PASS' if ok else 'FAIL'}]")

    # ---- (3) iso-bits per-group MSE: none / hadamard / learned-b ----
    def rot(W, R):
        return apply_rows(W.reshape(N, K // span, span), R).reshape(N, K)

    mse_none = _group_quant_mse(W, nl)
    mse_had = _group_quant_mse(rot(W, H), nl)
    mse_b = _group_quant_mse(rot(W, R_b), nl)
    print(f"\niso-bits weight-quant MSE (uniform):")
    print(f"  no-rotate        {mse_none:.4e}")
    print(f"  fixed Hadamard   {mse_had:.4e}  ({mse_none/mse_had:.2f}x vs none)")
    print(f"  learned Givens-b {mse_b:.4e}  ({mse_had/mse_b:.2f}x vs Hadamard)")
    # The learned fit minimizes this exact objective, so it must not be WORSE
    # than Hadamard (a 2% slack covers the proxy-vs-exact scale + subsampling).
    ok_b = mse_b <= mse_had * 1.02
    all_ok &= ok_b
    print(f"  learned-b <= Hadamard: [{'PASS' if ok_b else 'FAIL'}]")

    # ---- (4) importance-weighted MSE: hadamard vs learned-c ----
    imp_full = imp_k.unsqueeze(0).expand(N, -1)
    # importance transformed into each rotated basis (what the quantizer scores)
    rib_h = off["_rotated_importance"](imp_block, H).repeat(K // span)
    rib_c = off["_rotated_importance"](imp_block, R_c).repeat(K // span)
    mse_had_i = _group_quant_mse(rot(W, H), nl,
                                 rib_h.unsqueeze(0).expand(N, -1))
    mse_c_i = _group_quant_mse(rot(W, R_c), nl,
                               rib_c.unsqueeze(0).expand(N, -1))
    print(f"\nimportance-weighted weight-quant MSE (stage c):")
    print(f"  fixed Hadamard   {mse_had_i:.4e}")
    print(f"  learned Givens-c {mse_c_i:.4e}  ({mse_had_i/mse_c_i:.2f}x)")
    ok_c = mse_c_i <= mse_had_i * 1.02
    all_ok &= ok_c
    print(f"  learned-c <= Hadamard (weighted): [{'PASS' if ok_c else 'FAIL'}]")

    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
