"""Torchless numpy cross-check of _awq_moe_to_op_layout (moe_experts.py).

Faithfully re-implements the conversion's integer ops in numpy and verifies the
op-layout weights dequant-equal the AWQ source — the same property
test_moe_experts.py::test_conversion checks, but runnable without torch/GPU.
Catches axis/transpose/shift/rev-order/sign bugs in the layout conversion.
"""
import numpy as np

REV = [0, 4, 1, 5, 2, 6, 3, 7]  # _REVERSE_AWQ_PACK_ORDER


# ---- forward AWQ pack (mirror test_moe_experts.awq_pack / awq_pack_zeros) ----
def awq_pack(vals):  # (E, R, C) int -> (E, R, C//8) int32, AWQ bit order
    E, R, C = vals.shape
    packed = np.zeros((E, R, C // 8), dtype=np.int32)
    v = vals.astype(np.int32)
    for i in range(8):
        packed |= (v[:, :, i::8] & 0xF) << (REV[i] * 4)
    return packed


# ---- conversion under test: numpy port of _awq_to_op_layout_single ----------
def awq_to_op_layout_single(qw_e, sc_e, qz_e, pf, size_bits, rev, shifts):
    mask = (1 << size_bits) - 1
    K, Np = qw_e.shape
    N = Np * pf
    Gz = qz_e.shape[0]

    uw = (qw_e[..., None] >> shifts) & mask          # (K, Np, pf)
    uw = uw[:, :, rev].reshape(K, N)                 # (K, N) natural channels
    uw = np.ascontiguousarray(uw.T).astype(np.int32) # (N, K)
    w_packed = np.zeros((N, K // pf), dtype=np.int32)
    for j in range(pf):
        w_packed |= (uw[:, j::pf] & mask) << (j * size_bits)

    scales_op = np.ascontiguousarray(sc_e.T)         # (N, G)

    uz = (qz_e[..., None] >> shifts) & mask          # (G, Np, pf)
    uz = uz[:, :, rev].reshape(Gz, N)                # (G, N)
    uz = np.ascontiguousarray(uz.T).astype(np.int32) # (N, G)
    zeros_op = np.zeros((N // pf, Gz), dtype=np.int32)
    for j in range(pf):
        zeros_op |= (uz[j::pf, :] & mask) << (j * size_bits)

    return w_packed, scales_op, zeros_op


def awq_to_op_layout(qweight, scales, qzeros, group_size, size_bits=4):
    pf = 32 // size_bits
    E, K, Np = qweight.shape
    rev = np.array(REV[:pf], dtype=np.int64)
    shifts = np.arange(0, 32, size_bits, dtype=np.int32)
    wl, sl, zl = [], [], []
    for e in range(E):
        w, s, z = awq_to_op_layout_single(
            qweight[e], scales[e], qzeros[e], pf, size_bits, rev, shifts)
        wl.append(w); sl.append(s); zl.append(z)
    return np.stack(wl), np.stack(sl), np.stack(zl)


# ---- dequant helpers (mirror test_moe_experts) ------------------------------
def dequant_true(qvals, zeros, scales, group_size):  # (E,K,N)
    zexp = np.repeat(zeros, group_size, axis=1)
    sexp = np.repeat(scales.astype(np.float64), group_size, axis=1)
    return (qvals.astype(np.float64) - zexp.astype(np.float64)) * sexp


def unpack_op_weight(w_packed, scales_op, zeros_op, group_size):  # (E,N,K)
    E, N, Kp = w_packed.shape
    K = Kp * 8
    shifts = (np.arange(8, dtype=np.int32) * 4)
    nib = (w_packed[..., None] >> shifts) & 0xF        # (E,N,K//8,8)
    nib = nib.reshape(E, N, K).astype(np.float64)
    nidx = np.arange(N)
    zshift = ((nidx % 8) * 4).reshape(1, N, 1)
    zp = (zeros_op[:, nidx // 8, :] >> zshift) & 0xF   # (E,N,G)
    zp = np.repeat(zp.astype(np.float64), group_size, axis=2)
    sc = np.repeat(scales_op.astype(np.float64), group_size, axis=2)
    return (nib - zp) * sc


def run_case(E, K, N, g, rng):
    G = K // g
    qvals = rng.integers(0, 16, (E, K, N)).astype(np.int32)
    zeros = rng.integers(0, 16, (E, G, N)).astype(np.int32)
    # fp16-ish scales (use float32 to emulate; conversion stores fp16 but the
    # integer layout is what we validate, so scale precision is incidental)
    scales = (np.abs(rng.standard_normal((E, G, N))).astype(np.float32) * 0.02
              + 0.001)
    qweight = awq_pack(qvals)
    qzeros = awq_pack(zeros)

    w_p, s_p, z_p = awq_to_op_layout(qweight, scales, qzeros, g)
    assert w_p.shape == (E, N, K // 8), (w_p.shape, (E, N, K // 8))
    assert s_p.shape == (E, N, K // g)
    assert z_p.shape == (E, N // 8, K // g)

    W_true = dequant_true(qvals, zeros, scales, g)          # (E,K,N)
    W_op = unpack_op_weight(w_p, s_p, z_p, g)               # (E,N,K)
    diff = np.abs(W_true.transpose(0, 2, 1) - W_op).max()
    ok = diff < 1e-4
    print(f"  conv E={E} K={K} N={N} g={g}: max|diff|={diff:.2e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    rng = np.random.default_rng(0)
    cases = [(4, 256, 128, 32), (8, 512, 256, 128), (2, 1024, 512, 32),
             (3, 256, 64, 16), (1, 128, 128, 64)]
    res = [run_case(*c, rng=rng) for c in cases]
    print("=" * 50)
    ok = all(res)
    print("ALL PASSED" if ok else f"FAIL {sum(1 for r in res if not r)}/{len(res)}")
    return ok


if __name__ == "__main__":
    # Exit non-zero on failure so CI can gate on this (see .github/workflows/ci.yml).
    raise SystemExit(0 if main() else 1)
