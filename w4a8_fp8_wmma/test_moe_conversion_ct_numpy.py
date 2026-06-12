"""Torchless numpy cross-check of _ct_moe_to_op_layout (moe_experts.py).

The compressed-tensors MoE path (CompressedTensorsWNA16MoEMethod, used on
gfx1201 for the real target models like Qwen3.6-35B-A3B) registers GPTQ-convention
weights: (E, K//8, N) int32 packed along the INPUT dim K, natural nibble order,
symmetric uint4b8 (stored q in [0,15] = signed (q-8)). This re-implements the
converter's integer ops in numpy and verifies the op-layout weights dequant-equal
the source ((q-8)*scale) -- runnable without torch/GPU. Catches axis/transpose/
shift/packing bugs. (The op applies the implicit zp=8; the converter only repacks.)
"""
import numpy as np


# ---- forward GPTQ pack (natural order, packed along K) ----------------------
def gptq_pack(vals):  # (E, K, N) int -> (E, K//8, N) int32
    E, K, N = vals.shape
    packed = np.zeros((E, K // 8, N), dtype=np.int32)
    v = vals.astype(np.int32)
    for j in range(8):
        packed |= (v[:, j::8, :] & 0xF) << (j * 4)  # nibble j = input row k8*8+j
    return packed


# ---- converter under test: numpy port of _ct_to_op_layout_single ------------
def ct_to_op_single(qw_e, sc_e, pf, size_bits, shifts):
    mask = (1 << size_bits) - 1
    Kp, N = qw_e.shape
    K = Kp * pf
    uw = (qw_e[:, None, :] >> shifts.reshape(1, pf, 1)) & mask   # (Kp, pf, N)
    uw = uw.reshape(K, N)
    w_kn = np.ascontiguousarray(uw.T).astype(np.int32)          # (N, K)
    w_packed = np.zeros((N, K // pf), dtype=np.int32)
    for j in range(pf):
        w_packed |= (w_kn[:, j::pf] & mask) << (j * size_bits)
    scales_op = np.ascontiguousarray(sc_e.T)                    # (N, K//g)
    return w_packed, scales_op


def ct_to_op(qweight, scales, size_bits=4):  # (E,K//pf,N),(E,K//g,N)
    pf = 32 // size_bits
    E = qweight.shape[0]
    shifts = np.arange(0, 32, size_bits, dtype=np.int32)
    wl, sl = [], []
    for e in range(E):
        w, s = ct_to_op_single(qweight[e], scales[e], pf, size_bits, shifts)
        wl.append(w); sl.append(s)
    return np.stack(wl), np.stack(sl)


# ---- dequant (symmetric uint4b8: (q-8)*scale) -------------------------------
def dequant_true_sym(qvals, scales, group_size):  # (E,K,N)
    sexp = np.repeat(scales.astype(np.float64), group_size, axis=1)
    return (qvals.astype(np.float64) - 8.0) * sexp


def unpack_op_weight_sym(w_packed, scales_op, group_size):  # (E,N,K)
    E, N, Kp = w_packed.shape
    K = Kp * 8
    shifts = (np.arange(8, dtype=np.int32) * 4)
    nib = (w_packed[..., None] >> shifts) & 0xF
    nib = nib.reshape(E, N, K).astype(np.float64)
    sc = np.repeat(scales_op.astype(np.float64), group_size, axis=2)
    return (nib - 8.0) * sc


def run_case(E, K, N, g, rng):
    G = K // g
    qvals = rng.integers(0, 16, (E, K, N)).astype(np.int32)
    scales = (np.abs(rng.standard_normal((E, G, N))).astype(np.float32) * 0.02
              + 0.001)
    qweight = gptq_pack(qvals)                    # (E, K//8, N)

    w_p, s_p = ct_to_op(qweight, scales, 4)
    assert w_p.shape == (E, N, K // 8), (w_p.shape, (E, N, K // 8))
    assert s_p.shape == (E, N, K // g)

    W_true = dequant_true_sym(qvals, scales, g)          # (E,K,N)
    W_op = unpack_op_weight_sym(w_p, s_p, g)             # (E,N,K)
    diff = np.abs(W_true.transpose(0, 2, 1) - W_op).max()
    ok = diff < 1e-4
    print(f"  ct-conv E={E} K={K} N={N} g={g}: max|diff|={diff:.2e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    rng = np.random.default_rng(0)
    # shapes mimic per-expert w13 (K=hidden, N=2*inter) and w2 (K=inter, N=hidden)
    cases = [(4, 256, 128, 32), (8, 512, 256, 32), (2, 1024, 512, 32),
             (3, 256, 128, 16), (1, 2304, 1792, 32)]  # last ~ Qwen3.6 w13 g32
    res = [run_case(*c, rng=rng) for c in cases]
    print("=" * 50)
    print("ALL PASSED" if all(res) else f"FAIL {sum(1 for r in res if not r)}/{len(res)}")


if __name__ == "__main__":
    main()
