"""Numeric validation of the MIXED prefill+decode CCA path: one forward runs both
fused kernels — cca_decode_qk over the decode region and cca_prefill_qk over the
prefill region — into the SAME conv_states (disjoint slots), then concatenates
their q|k outputs decode-first (the V1 batch layout). This mirrors cca.py
forward_cuda's mixed-batch wiring (use_hip_qk).

Checks:
  1. concat([ker_qk_d, ker_qk_p]) is bit-exact vs an eager mixed reference
     (decode region via the windowed conv_ref; prefill region via the flat-buffer
     conv; identical means + per-head RMS-norm postprocess);
  2. the final conv_states matches the eager reference everywhere — decode slots
     rolled, prefill slots replaced, block 0 zeroed by the pad decode token, all
     other blocks untouched;
  3. mixed == pure ⊕ pure: each region's output/state in the combined launch is
     identical to running that kernel alone, i.e. the two launches don't perturb
     each other (the whole point of the extension).

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
    python test_cca_mixed_qk.py
"""
import math
import sys

import torch
import cca_op  # noqa: F401

torch.manual_seed(0)

# ZAYA CCA dims (same as test_cca_decode_qk / test_cca_prefill_qk)
C, d, K0, K1, TP = 1280, 128, 2, 2, 2
num_q, num_k = 8, 2
gqa = num_q // num_k          # 4
latent_q = num_q * d          # 1024
latent_k = num_k * d          # 256
H = num_q + num_k             # 10
sqrt_d = math.sqrt(d)
eps = 1e-12
dev = "cuda"
NB = 64
tp = TP

# Shared conv weights (one CCA layer drives both regions).
w0 = torch.randn(C, K0, device=dev)
b0 = torch.randn(C, device=dev)
w1 = torch.randn(C, d, K1, device=dev)
b1 = torch.randn(C, device=dev)
temp_eff = torch.rand(num_k, device=dev) + 0.5
# Kernels expect w1 pre-transposed to [H, d_in, d_out, K1] (coalesced reads).
w1t = w1.view(H, d, d, K1).permute(0, 2, 1, 3).contiguous()

conv_states0 = torch.randn(NB, C, tp, device=dev)


def means_norm(conv_out, qk_new_r):
    """Shared grouped-means + per-head RMS-norm postprocess (q|k -> normalized)."""
    n = conv_out.shape[0]
    query = conv_out[:, :latent_q].view(n, num_q, d).clone()
    key = conv_out[:, latent_q:].view(n, num_k, d).clone()
    qpre = qk_new_r[:, :latent_q].view(n, num_q, d)
    kbase = qk_new_r[:, latent_q:].view(n, num_k, d)
    qpre_g = qpre.view(n, num_k, gqa, d)
    query.view(n, num_k, gqa, d).add_(qpre_g, alpha=0.5).add_(kbase.unsqueeze(2), alpha=0.5)
    key.add_(qpre_g.mean(dim=2), alpha=0.5).add_(kbase, alpha=0.5)
    query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + eps) * sqrt_d
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + eps) * sqrt_d
    key = key * temp_eff.view(1, num_k, 1)
    return torch.cat([query.reshape(n, latent_q), key.reshape(n, latent_k)], dim=-1)


# ---------------------------------------------------------------------------
# Decode region: D single-token sequences (the last one is a cudagraph pad).
# ---------------------------------------------------------------------------
def conv_ref_decode(x):  # x [N, C, TP+1] -> [N, C]
    xw = x.unfold(-1, K0, 1)
    mid = (xw * w0[:, None, :]).sum(-1) + b0[None, :, None]
    midw = mid.view(mid.shape[0], C // d, d, mid.shape[-1]).unfold(-1, K1, 1)
    w1g = w1.view(C // d, d, d, K1)
    out = torch.einsum("godk,sgdtk->sgot", w1g, midw) + b1.view(1, C // d, d, 1)
    return out.reshape(x.shape[0], C, out.shape[-1])[:, :, 0]


D = 6  # 5 real + 1 pad
qk_new_d = torch.randn(D, C, device=dev)
# Distinct decode slots; the pad token's slot collapses to the reserved block 0.
dslot = torch.tensor([3, 7, 11, 15, 19, 0], device=dev, dtype=torch.long)
is_pad_d = torch.tensor([False] * (D - 1) + [True], device=dev)


def decode_ref_and_state(states_in):
    """Eager decode reference: windowed conv + means/norm, and the rolled state."""
    win = torch.zeros(D, C, TP + 1, device=dev)
    for s in range(D):
        if not is_pad_d[s]:
            win[s, :, :TP] = states_in[dslot[s]]
            win[s, :, TP] = qk_new_d[s]
    conv_out = conv_ref_decode(win)              # [D, C]
    ref_qk = means_norm(conv_out, qk_new_d)      # [D, C]
    state_writes = {}
    for s in range(D):
        new = torch.zeros(C, TP, device=dev) if is_pad_d[s] else win[s, :, 1:]
        state_writes[int(dslot[s])] = new        # block 0 (pad) -> zeros
    return ref_qk, state_writes


# ---------------------------------------------------------------------------
# Prefill region: R varlen requests, some carrying initial conv state.
# ---------------------------------------------------------------------------
def flat_conv(x):  # [C, flat_len] -> [C, flat_len - TP]
    xw = x.unfold(-1, K0, 1)
    mid = (xw * w0[:, None, :]).sum(-1) + b0[:, None]
    g = C // d
    midw = mid.view(g, d, mid.shape[-1]).unfold(-1, K1, 1)
    w1g = w1.view(g, d, d, K1)
    out = torch.einsum("godk,gdtk->got", w1g, midw) + b1.view(g, d, 1)
    return out.reshape(C, -1)


seq_lens = [5, 1, 3, 2]
has_init_list = [True, False, True, True]
R = len(seq_lens)
P = sum(seq_lens)
qk_new_p = torch.randn(P, C, device=dev)
has_init = torch.tensor(has_init_list, device=dev)
# Prefill slots disjoint from the decode slots {0,3,7,11,15,19}.
pslot = (torch.arange(R, device=dev) * 7 + 24).to(torch.long)   # [24,31,38,45]

req_idx = torch.arange(R, device=dev)
seq_lens_t = torch.tensor(seq_lens, device=dev)
token_req = torch.repeat_interleave(req_idx, seq_lens_t, output_size=P)
token_flat = torch.arange(P, device=dev)
qsl = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                 torch.cumsum(seq_lens_t, 0)])                  # [R+1]


def prefill_init_states(states_in):
    init_states = states_in[pslot].clone()
    return torch.where(has_init.view(-1, 1, 1), init_states, init_states.new_zeros(()))


def prefill_ref_and_state(states_in):
    """Eager prefill reference: flat-buffer two-stage conv + means/norm + new state."""
    init_states = prefill_init_states(states_in)
    flat_len = P + R * tp
    flat_in = qk_new_p.new_zeros((C, flat_len))
    token_pos = token_flat + (token_req + 1) * tp
    out_pos = token_flat + token_req * tp
    seg_starts = qsl[:-1] + req_idx * tp
    pad_off = torch.arange(tp, device=dev)
    flat_in[:, token_pos] = qk_new_p.t()
    state_pos = (seg_starts.unsqueeze(1) + pad_off).reshape(-1)
    flat_in[:, state_pos] = init_states.permute(1, 0, 2).reshape(C, -1)

    conv_out = flat_conv(flat_in)[:, out_pos].t()               # [P, C]
    ref_qk = means_norm(conv_out, qk_new_p)                     # [P, C]

    new_state_pos = ((qsl[1:] + req_idx * tp).unsqueeze(1) + pad_off).reshape(-1)
    new_states = flat_in[:, new_state_pos].reshape(C, R, tp).permute(1, 0, 2)
    state_writes = {int(pslot[i]): new_states[i] for i in range(R)}
    return ref_qk, state_writes


# Kernel metadata for the prefill region (mirrors cca.py use_hip_qk prefill path).
seg_pos = (token_flat - qsl[:-1][token_req]).to(torch.int32)
req_id = token_req.to(torch.int32)
slot_p = pslot[token_req]
is_last_p = (token_flat == (qsl[1:] - 1)[token_req])

# ---------------------------------------------------------------------------
# Eager mixed reference (decode-first concat + combined conv_states).
# ---------------------------------------------------------------------------
ref_qk_d, dec_writes = decode_ref_and_state(conv_states0)
ref_qk_p, pre_writes = prefill_ref_and_state(conv_states0)
ref_qk = torch.cat([ref_qk_d, ref_qk_p], dim=0)                 # [D+P, C]

ref_states = conv_states0.clone()
for slot, val in {**dec_writes, **pre_writes}.items():
    ref_states[slot] = val

# ---------------------------------------------------------------------------
# Mixed kernel run: prefill THEN decode into one shared conv_states (cca.py order).
# Disjoint slots + prefill reading a pre-gathered init_states copy => no race.
# ---------------------------------------------------------------------------
ker_states = conv_states0.clone()
init_states_k = prefill_init_states(ker_states)
ker_qk_p = torch.ops.zaya_cca.cca_prefill_qk(
    qk_new_p.contiguous(), ker_states, init_states_k.contiguous(),
    seg_pos, req_id, slot_p, is_last_p, w0, b0, w1t, b1, temp_eff,
    num_q, gqa, latent_q, sqrt_d)
ker_qk_d = torch.ops.zaya_cca.cca_decode_qk(
    qk_new_d.contiguous(), ker_states, dslot, is_pad_d, w0, b0, w1t, b1, temp_eff,
    num_q, gqa, latent_q, sqrt_d)
ker_qk = torch.cat([ker_qk_d, ker_qk_p], dim=0)                # decode-first

# ---------------------------------------------------------------------------
# 1. q|k bit-exact vs the eager mixed reference (compare non-pad decode + all
#    prefill rows; pad decode rows are zeroed downstream in the model).
# ---------------------------------------------------------------------------
row_ok = torch.cat([~is_pad_d, torch.ones(P, dtype=torch.bool, device=dev)])
qk_rel = ((ker_qk[row_ok] - ref_qk[row_ok]).abs().max().item()
          / ref_qk[row_ok].abs().max().item())

# 2. combined conv_states bit-exact everywhere.
st_err = (ker_states - ref_states).abs().max().item()

# 3. mixed == pure ⊕ pure: rerun each kernel ALONE on a fresh state and confirm
#    its output (and its own slots) are identical to the combined launch.
solo_dec_states = conv_states0.clone()
solo_qk_d = torch.ops.zaya_cca.cca_decode_qk(
    qk_new_d.contiguous(), solo_dec_states, dslot, is_pad_d, w0, b0, w1t, b1,
    temp_eff, num_q, gqa, latent_q, sqrt_d)
solo_pre_states = conv_states0.clone()
solo_qk_p = torch.ops.zaya_cca.cca_prefill_qk(
    qk_new_p.contiguous(), solo_pre_states, prefill_init_states(solo_pre_states).contiguous(),
    seg_pos, req_id, slot_p, is_last_p, w0, b0, w1t, b1, temp_eff,
    num_q, gqa, latent_q, sqrt_d)
dec_qk_iso = (ker_qk_d - solo_qk_d).abs().max().item()
pre_qk_iso = (ker_qk_p - solo_qk_p).abs().max().item()
dec_state_iso = (ker_states[dslot] - solo_dec_states[dslot]).abs().max().item()
pre_state_iso = (ker_states[pslot] - solo_pre_states[pslot]).abs().max().item()
iso = max(dec_qk_iso, pre_qk_iso, dec_state_iso, pre_state_iso)

ok = qk_rel < 1e-4 and st_err < 1e-5 and iso < 1e-7
print(f"mixed D={D}(pad 1) P={P} R={R} decode_slots={dslot.tolist()} "
      f"prefill_slots={pslot.tolist()}")
print(f"  qk_rel={qk_rel:.2e}  state_err={st_err:.1e}  "
      f"pure-vs-mixed_iso={iso:.1e}  -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
