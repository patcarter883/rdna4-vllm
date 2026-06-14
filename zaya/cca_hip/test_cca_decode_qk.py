"""Numeric validation of the full-fusion CCA decode kernel (conv + grouped-means
+ per-head RMS-norm + qk write) vs an fp32 eager reference replicating cca.py.

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
    python test_cca_decode_qk.py
"""
import math

import torch
import cca_op  # noqa: F401

torch.manual_seed(0)

# ZAYA CCA dims
C, d, K0, K1, TP = 1280, 128, 2, 2, 2
num_q, num_k = 8, 2
gqa = num_q // num_k  # 4
latent_q = num_q * d  # 1024
latent_k = num_k * d  # 256
Lmid = TP - K0 + 2
S, NB = 16, 64
sqrt_d = math.sqrt(d)
eps = 1e-12
dev = "cuda"

qk_new = torch.randn(S, C, device=dev)
conv_states = torch.randn(NB, C, TP, device=dev)
w0 = torch.randn(C, K0, device=dev)
b0 = torch.randn(C, device=dev)
w1 = torch.randn(C, d, K1, device=dev)
b1 = torch.randn(C, device=dev)
temp_eff = torch.rand(num_k, device=dev) + 0.5  # already-clamped/exp'd temp

slot = torch.arange(1, S + 1, device=dev, dtype=torch.long)
is_pad = torch.zeros(S, device=dev, dtype=torch.bool)
slot[-1] = 0
is_pad[-1] = True


def conv_ref(x):  # x [N, C, TP+1] -> [N, C]
    xw = x.unfold(-1, K0, 1)
    mid = (xw * w0[:, None, :]).sum(-1) + b0[None, :, None]
    midw = mid.view(mid.shape[0], C // d, d, mid.shape[-1]).unfold(-1, K1, 1)
    w1g = w1.view(C // d, d, d, K1)
    out = torch.einsum("godk,sgdtk->sgot", w1g, midw) + b1.view(1, C // d, d, 1)
    return out.reshape(x.shape[0], C, out.shape[-1])[:, :, 0]


# ---- eager reference ----
ref_states = conv_states.clone()
win = torch.zeros(S, C, TP + 1, device=dev)
for s in range(S):
    if not is_pad[s]:
        win[s, :, :TP] = conv_states[slot[s]]
        win[s, :, TP] = qk_new[s]
conv_out = conv_ref(win)  # [S, C] (q|k)

query = conv_out[:, :latent_q].view(S, num_q, d).clone()
key = conv_out[:, latent_q:].view(S, num_k, d).clone()
qpre = qk_new[:, :latent_q].view(S, num_q, d)
kbase = qk_new[:, latent_q:].view(S, num_k, d)
# means
qpre_g = qpre.view(S, num_k, gqa, d)
query.view(S, num_k, gqa, d).add_(qpre_g, alpha=0.5).add_(kbase.unsqueeze(2), alpha=0.5)
key.add_(qpre_g.mean(dim=2), alpha=0.5).add_(kbase, alpha=0.5)
# rms-norm
query = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + eps) * sqrt_d
key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + eps) * sqrt_d
key = key * temp_eff.view(1, num_k, 1)
ref_qk = torch.cat([query.reshape(S, latent_q), key.reshape(S, latent_k)], dim=-1)
for s in range(S):
    new = torch.zeros(C, TP, device=dev) if is_pad[s] else win[s, :, 1:]
    ref_states[slot[s]] = new

# ---- kernel ----
# Kernel expects w1 transposed to [H, d_in, d_out, K1] for coalesced reads.
H = C // d
w1t = w1.view(H, d, d, K1).permute(0, 2, 1, 3).contiguous()
ker_states = conv_states.clone()
ker_qk = torch.ops.zaya_cca.cca_decode_qk(
    qk_new.contiguous(), ker_states, slot, is_pad, w0, b0, w1t, b1, temp_eff,
    num_q, gqa, latent_q, sqrt_d)

# pad rows are masked downstream in the model; compare non-pad rows only
nz = ~is_pad
qk_err = (ker_qk[nz] - ref_qk[nz]).abs().max().item()
qk_rel = qk_err / ref_qk[nz].abs().max().item()
st_err = (ker_states - ref_states).abs().max().item()
print(f"qk     max abs err = {qk_err:.3e}  (rel {qk_rel:.3e})")
print(f"state  max abs err = {st_err:.3e}")
print("PASS" if (qk_rel < 1e-4 and st_err < 1e-5) else "FAIL")
