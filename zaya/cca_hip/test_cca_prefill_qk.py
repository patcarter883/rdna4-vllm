"""Numeric validation of the fused CCA PREFILL kernel (conv + grouped-means +
per-head RMS-norm + new-conv-state) vs an fp32 eager reference that mirrors the
flat-buffer prefill path in cca.py (scatter -> two-stage conv -> gather + means +
norm + last-TP-columns state).

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
    python test_cca_prefill_qk.py
"""
import math

import torch
import cca_op  # noqa: F401

torch.manual_seed(0)

# ZAYA CCA dims
C, d, K0, K1, TP = 1280, 128, 2, 2, 2
num_q, num_k = 8, 2
gqa = num_q // num_k
latent_q = num_q * d          # 1024
latent_k = num_k * d          # 256
H = num_q + num_k             # 10
Lmid = TP - K0 + 2
sqrt_d = math.sqrt(d)
eps = 1e-12
dev = "cuda"

# Scenarios stress every window/state path:
#  - long seg whose last token reads no cache; short segs that read cache;
#  - a 1-token seg with NO init (all-zero predecessors);
#  - a 1-token seg WITH init (one token both reads 2 cached cols AND writes state);
#  - a 2-token seg with init; a >TP seg with NO init (first TP tokens zero-padded).
SCENARIOS = [
    ([5, 1, 3, 2], [True, False, True, True]),
    ([1, 2, 7], [True, True, False]),
    ([1, 4], [False, True]),
]
NB = 32
tp = TP

# fixed weights across scenarios
w0 = torch.randn(C, K0, device=dev)
b0 = torch.randn(C, device=dev)
w1 = torch.randn(C, d, K1, device=dev)
b1 = torch.randn(C, device=dev)
temp_eff = torch.rand(num_k, device=dev) + 0.5
w1t = w1.view(H, d, d, K1).permute(0, 2, 1, 3).contiguous()


def flat_conv(x):  # [C, flat_len] -> [C, flat_len - TP]
    xw = x.unfold(-1, K0, 1)                                # [C, Lm, K0]
    mid = (xw * w0[:, None, :]).sum(-1) + b0[:, None]       # [C, Lm]
    g = C // d
    midw = mid.view(g, d, mid.shape[-1]).unfold(-1, K1, 1)  # [g, d, Lout, K1]
    w1g = w1.view(g, d, d, K1)                              # [g, o, di, k]
    out = torch.einsum("godk,gdtk->got", w1g, midw) + b1.view(g, d, 1)
    return out.reshape(C, -1)


def run(seq_lens, has_init_list):
    has_init = torch.tensor(has_init_list, device=dev)
    R = len(seq_lens)
    P = sum(seq_lens)
    qk_new = torch.randn(P, C, device=dev)
    conv_states0 = torch.randn(NB, C, tp, device=dev)
    req_idx = torch.arange(R, device=dev)
    seq_lens_t = torch.tensor(seq_lens, device=dev)
    token_req = torch.repeat_interleave(req_idx, seq_lens_t, output_size=P)
    token_flat = torch.arange(P, device=dev)
    qsl = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                     torch.cumsum(seq_lens_t, 0)])          # [R+1]
    # distinct slots
    state_indices = (torch.arange(R, device=dev) * 5 + 2).to(torch.long)

    init_states = conv_states0[state_indices].clone()
    init_states = torch.where(has_init.view(-1, 1, 1), init_states,
                              init_states.new_zeros(()))     # [R, C, TP]

    # ---- eager reference: flat-buffer two-stage conv (mirrors cca.py) ----
    flat_len = P + R * tp
    flat_in = qk_new.new_zeros((C, flat_len))
    token_pos = token_flat + (token_req + 1) * tp
    out_pos = token_flat + token_req * tp
    seg_starts = qsl[:-1] + req_idx * tp
    pad_off = torch.arange(tp, device=dev)
    flat_in[:, token_pos] = qk_new.t()
    state_pos = (seg_starts.unsqueeze(1) + pad_off).reshape(-1)
    flat_in[:, state_pos] = init_states.permute(1, 0, 2).reshape(C, -1)

    conv_out = flat_conv(flat_in)[:, out_pos].t()            # [P, C]

    query = conv_out[:, :latent_q].view(P, num_q, d).clone()
    key = conv_out[:, latent_q:].view(P, num_k, d).clone()
    qpre = qk_new[:, :latent_q].view(P, num_q, d)
    kbase = qk_new[:, latent_q:].view(P, num_k, d)
    qpre_g = qpre.view(P, num_k, gqa, d)
    query.view(P, num_k, gqa, d).add_(qpre_g, alpha=0.5).add_(kbase.unsqueeze(2), alpha=0.5)
    key.add_(qpre_g.mean(dim=2), alpha=0.5).add_(kbase, alpha=0.5)
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + eps) * sqrt_d
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + eps) * sqrt_d
    key = key * temp_eff.view(1, num_k, 1)
    ref_qk = torch.cat([query.reshape(P, latent_q), key.reshape(P, latent_k)], dim=-1)

    new_state_pos = ((qsl[1:] + req_idx * tp).unsqueeze(1) + pad_off).reshape(-1)
    new_states = flat_in[:, new_state_pos].reshape(C, R, tp).permute(1, 0, 2)
    ref_states = conv_states0.clone()
    ref_states[state_indices] = new_states

    # ---- kernel ----
    seg_pos = (token_flat - qsl[:-1][token_req]).to(torch.int32)
    req_id = token_req.to(torch.int32)
    slot = state_indices[token_req]
    is_last = (token_flat == (qsl[1:] - 1)[token_req])

    ker_states = conv_states0.clone()
    ker_qk = torch.ops.zaya_cca.cca_prefill_qk(
        qk_new.contiguous(), ker_states, init_states.contiguous(),
        seg_pos, req_id, slot, is_last, w0, b0, w1t, b1, temp_eff,
        num_q, gqa, latent_q, sqrt_d)

    qk_rel = (ker_qk - ref_qk).abs().max().item() / ref_qk.abs().max().item()
    st_err = (ker_states[state_indices] - ref_states[state_indices]).abs().max().item()
    mask = torch.ones(NB, dtype=torch.bool, device=dev)
    mask[state_indices] = False
    untouched = (ker_states[mask] - conv_states0[mask]).abs().max().item()
    ok = qk_rel < 1e-4 and st_err < 1e-5 and untouched < 1e-7
    print(f"  seq_lens={seq_lens} init={has_init_list} P={P} R={R} | "
          f"qk_rel={qk_rel:.2e} st={st_err:.1e} untouched={untouched:.1e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


all_ok = all(run(sl, hi) for sl, hi in SCENARIOS)
print("ALL PASS" if all_ok else "SOME FAILED")
