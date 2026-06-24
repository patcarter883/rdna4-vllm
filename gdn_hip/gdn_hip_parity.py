"""Task 4-#19 — numeric parity for the native gdn_hip HIP kernels (GPU).

Each gdn_hip op is checked against a pure-torch reference of the EXACT math it implements (the fla
gated-delta-rule recurrence + depthwise causal conv + gated RMSNorm). Both run fp32, so a faithful
kernel matches to ~1e-4; a real indexing/LDS/register bug shows as a large max|Δ|. This recurrent
reference is also the oracle for the future chunked-HIP prefill.

Run inside the combined ROCm image UNDER a 1-card lease (executes HIP kernels):
    .../gpu-lease.sh -n 1 -- bash -c 'docker run ... python /engine/tools/gdn_hip_parity.py'
(The gdn_hip_C*.so must be built first: cd gdn_hip && GPU_ARCHS=gfx1201 python setup.py build_ext --inplace)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import gdn_hip  # loads the .so + registers torch.ops.gdn_hip.*

DEV = "cuda"
torch.manual_seed(0)

# Real GDN geometry (Qwen3.5/3.6): num_k_heads=16, num_v_heads=32, head_k_dim=head_v_dim=128.
H, HV, K, V = 16, 32, 128, 128
SCALE = K ** -0.5


def _report(name: str, got: torch.Tensor, ref: torch.Tensor, tol: float = 2e-3) -> bool:
    d = (got.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().mean().item()
    ok = d <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} max|Δ|={d:.3e}  (ref |·|~{scale:.3e})")
    return ok


def _softplus(x):
    return torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)


def ref_step(S, q, k, v, g, beta):
    """One recurrence step. S:[V,K], q,k:[K], v:[V] -> o:[V], updates S."""
    S = S * torch.exp(g)
    v = v - S @ k          # [V]
    v = v * beta
    S = S + torch.outer(v, k)
    o = S @ q              # [V]
    return o, S


def check_decode() -> bool:
    B = 4
    num_slots = 8
    q = torch.randn(B, H, K, device=DEV)
    k = torch.randn(B, H, K, device=DEV)
    v = torch.randn(B, HV, V, device=DEV)
    a = torch.randn(B, HV, device=DEV)
    b = torch.randn(B, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    # slots: mix of valid + one NULL (0) to exercise the skip
    idx = torch.tensor([1, 0, 3, 5], dtype=torch.long, device=DEV)

    ref_state = state.clone()
    ref_out = torch.zeros(B, HV, V, device=DEV)
    for bi in range(B):
        slot = int(idx[bi])
        if slot <= 0:
            continue
        for hv in range(HV):
            hq = hv // (HV // H)
            qn = F.normalize(q[bi, hq], dim=-1, eps=1e-6) * SCALE
            kn = F.normalize(k[bi, hq], dim=-1, eps=1e-6)
            g = -torch.exp(A_log[hv]) * _softplus(a[bi, hv] + dt_bias[hv])
            beta = torch.sigmoid(b[bi, hv])
            o, S = ref_step(ref_state[slot, hv], qn, kn, v[bi, hv].clone(), g, beta)
            ref_out[bi, hv] = o
            ref_state[slot, hv] = S

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.gdn_decode(q, k, v, a, b, A_log, dt_bias, got_state, idx, SCALE, 1)
    ok = _report("gdn_decode.out", got_out, ref_out)
    ok &= _report("gdn_decode.state", got_state[idx[idx > 0]], ref_state[idx[idx > 0]])
    return ok


def check_prefill() -> bool:
    lens = [5, 3]
    N = len(lens)
    T = sum(lens)
    num_slots = 6
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32, device=DEV)
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV)
    b = torch.randn(T, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV)
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)

    ref_state = state.clone()
    ref_out = torch.zeros(T, HV, V, device=DEV)
    for n in range(N):
        slot = int(idx[n])
        bos = int(cu[n])
        for hv in range(HV):
            hq = hv // (HV // H)
            S = ref_state[slot, hv].clone() if has_init[n] else torch.zeros(V, K, device=DEV)
            for t in range(bos, int(cu[n + 1])):
                qn = F.normalize(q[t, hq], dim=-1, eps=1e-6) * SCALE
                kn = F.normalize(k[t, hq], dim=-1, eps=1e-6)
                g = -torch.exp(A_log[hv]) * _softplus(a[t, hv] + dt_bias[hv])
                beta = torch.sigmoid(b[t, hv])
                o, S = ref_step(S, qn, kn, v[t, hv].clone(), g, beta)
                ref_out[t, hv] = o
            ref_state[slot, hv] = S

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.gdn_prefill(q, k, v, a, b, A_log, dt_bias, cu, idx, has_init,
                                            got_state, SCALE, 1)
    ok = _report("gdn_prefill.out", got_out, ref_out)
    ok &= _report("gdn_prefill.state", got_state[idx], ref_state[idx])
    return ok


def check_prefill_chunked() -> bool:
    """Chunked prefill vs the recurrent kernel (the validated oracle), on sequences spanning several
    GDN_CHUNK=32 chunks + a partial final chunk. Mild decay (A_log~-2) so gamma doesn't underflow —
    the regime where the chunked ratio formulation and the recurrent step form are comparable."""
    lens = [40, 70]  # 40 = 1.25 chunks; 70 = 2.19 chunks
    N, T = len(lens), sum(lens)
    num_slots = 6
    cu = torch.tensor([0, 40, 110], dtype=torch.int32, device=DEV)
    q = torch.randn(T, H, K, device=DEV)
    k = torch.randn(T, H, K, device=DEV)
    v = torch.randn(T, HV, V, device=DEV)
    a = torch.randn(T, HV, device=DEV)
    b = torch.randn(T, HV, device=DEV)
    A_log = torch.randn(HV, device=DEV) * 0.5 - 2.0  # exp(A_log)~0.05-0.3 -> mild per-token decay
    dt_bias = torch.randn(HV, device=DEV)
    state = torch.randn(num_slots, HV, V, K, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)

    st_ref = state.clone()
    out_ref = torch.ops.gdn_hip.gdn_prefill(q, k, v, a, b, A_log, dt_bias, cu, idx, has_init,
                                            st_ref, SCALE, 1)
    st_ch = state.clone()
    out_ch = torch.ops.gdn_hip.gdn_prefill_chunked(q, k, v, a, b, A_log, dt_bias, cu, idx, has_init,
                                                   st_ch, SCALE, 1)
    ok = _report("prefill_chunked.out (vs recurrent)", out_ch, out_ref, tol=5e-3)
    ok &= _report("prefill_chunked.state", st_ch[idx], st_ref[idx], tol=5e-3)
    return ok


def check_conv_update() -> bool:
    B, C, W = 4, 256, 4
    num_slots = 6
    x = torch.randn(B, C, device=DEV)
    weight = torch.randn(C, W, device=DEV)
    bias = torch.randn(C, device=DEV)
    state = torch.randn(num_slots, C, W - 1, device=DEV)
    idx = torch.tensor([1, 0, 3, 5], dtype=torch.long, device=DEV)

    ref_state = state.clone()
    ref_out = torch.zeros(B, C, device=DEV)
    for bi in range(B):
        slot = int(idx[bi])
        win = torch.zeros(C, W, device=DEV)
        if slot > 0:
            win[:, : W - 1] = ref_state[slot]
        win[:, W - 1] = x[bi]
        acc = (win * weight).sum(-1) + bias
        ref_out[bi] = F.silu(acc)
        if slot > 0:
            ref_state[slot] = win[:, 1:]  # roll left, append new at tail

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.causal_conv1d_update(x, weight, bias, got_state, idx, 1)
    ok = _report("conv1d_update.out", got_out, ref_out)
    ok &= _report("conv1d_update.state", got_state[idx[idx > 0]], ref_state[idx[idx > 0]])
    return ok


def check_conv_fwd() -> bool:
    lens = [5, 3]
    N, T, C, W = 2, 8, 256, 4
    num_slots = 6
    cu = torch.tensor([0, 5, 8], dtype=torch.int32, device=DEV)
    x = torch.randn(T, C, device=DEV)
    weight = torch.randn(C, W, device=DEV)
    bias = torch.randn(C, device=DEV)
    state = torch.randn(num_slots, C, W - 1, device=DEV)
    idx = torch.tensor([1, 4], dtype=torch.long, device=DEV)
    has_init = torch.tensor([1, 0], dtype=torch.uint8, device=DEV)

    ref_state = state.clone()
    ref_out = torch.zeros(T, C, device=DEV)
    for n in range(N):
        slot = int(idx[n])
        hist = ref_state[slot].clone() if has_init[n] else torch.zeros(C, W - 1, device=DEV)
        for t in range(int(cu[n]), int(cu[n + 1])):
            win = torch.cat([hist, x[t].unsqueeze(-1)], dim=-1)  # [C, W]
            acc = (win * weight).sum(-1) + bias
            ref_out[t] = F.silu(acc)
            hist = win[:, 1:]
        ref_state[slot] = hist

    got_state = state.clone()
    got_out = torch.ops.gdn_hip.causal_conv1d_fwd(x, weight, bias, cu, idx, has_init, got_state, 1)
    ok = _report("conv1d_fwd.out", got_out, ref_out)
    ok &= _report("conv1d_fwd.state", got_state[idx], ref_state[idx])
    return ok


def check_rmsnorm_gated() -> bool:
    M, D = 64, 128
    x = torch.randn(M, D, device=DEV)
    z = torch.randn(M, D, device=DEV)
    weight = torch.randn(D, device=DEV)
    eps = 1e-5
    inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    ref = x * inv * weight * F.silu(z)
    got = torch.ops.gdn_hip.rmsnorm_gated(x, z, weight, eps)
    return _report("rmsnorm_gated", got, ref)


def main() -> None:
    assert torch.cuda.is_available(), "needs a GPU (run under a lease)"
    print(f"=== gdn_hip parity vs torch reference (device={torch.cuda.get_device_name()}) ===")
    results = {
        "gdn_decode": check_decode(),
        "gdn_prefill": check_prefill(),
        "gdn_prefill_chunked": check_prefill_chunked(),
        "causal_conv1d_update": check_conv_update(),
        "causal_conv1d_fwd": check_conv_fwd(),
        "rmsnorm_gated": check_rmsnorm_gated(),
    }
    print("=" * 50)
    allok = all(results.values())
    for n, ok in results.items():
        print(f"  {n:24s} {'PASS' if ok else 'FAIL'}")
    print("\nRESULT:", "ALL PASS — gdn_hip numerics faithful" if allok else "FAIL (see above)")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
