"""Standalone numeric validation of the CCA conv+state HIP op vs an fp32 eager
reference (the exact cca.py decode math). Run in-container on gfx1201:

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
    python test_cca_kernel.py
"""
import torch
import cca_op  # noqa: F401  (imports zaya_cca_C + registers op + fake meta)

torch.manual_seed(0)

# ZAYA CCA dims
C, d, K0, K1, TP = 1280, 128, 2, 2, 2
g = C // d  # 10
Lmid = TP - K0 + 2  # 2
S = 16            # decode batch
NB = 64           # state blocks


def conv_ref(x, w0, b0, w1, b1):
    # x: [N, C, TP+1] -> [N, C, 1]  (exact cca.py _conv_qk_decode, fp32)
    xw = x.unfold(-1, K0, 1)                       # [N,C,Lmid,K0]
    mid = (xw * w0[:, None, :]).sum(-1) + b0[None, :, None]
    midw = mid.view(mid.shape[0], g, d, mid.shape[-1]).unfold(-1, K1, 1)
    w1g = w1.view(g, d, d, K1)
    out = torch.einsum("godk,sgdtk->sgot", w1g, midw) + b1.view(1, g, d, 1)
    return out.reshape(x.shape[0], C, out.shape[-1])


dev = "cuda"
qk_new = torch.randn(S, C, device=dev)
conv_states = torch.randn(NB, C, TP, device=dev)
w0 = torch.randn(C, K0, device=dev)
b0 = torch.randn(C, device=dev)
w1 = torch.randn(C, d, K1, device=dev)
b1 = torch.randn(C, device=dev)

# slots: tokens 0..S-2 use distinct non-zero slots; last token is a pad (slot 0)
slot = torch.arange(1, S + 1, device=dev, dtype=torch.long)
is_pad = torch.zeros(S, device=dev, dtype=torch.bool)
slot[-1] = 0
is_pad[-1] = True

# ---- eager reference ----
ref_states = conv_states.clone()
win = torch.zeros(S, C, TP + 1, device=dev)
for s in range(S):
    if is_pad[s]:
        continue
    win[s, :, :TP] = conv_states[slot[s]]
    win[s, :, TP] = qk_new[s]
ref_out = conv_ref(win, w0, b0, w1, b1)[:, :, 0]  # [S, C]
for s in range(S):
    new = torch.zeros(C, TP, device=dev) if is_pad[s] else win[s, :, 1:]
    ref_states[slot[s]] = new

# ---- kernel ----
ker_states = conv_states.clone()
ker_out = torch.ops.zaya_cca.conv_state_decode(
    qk_new.contiguous(), ker_states, slot, is_pad, w0, b0, w1, b1)

out_err = (ker_out - ref_out).abs().max().item()
st_err = (ker_states - ref_states).abs().max().item()
out_rel = out_err / ref_out.abs().max().clamp_min(1e-9).item()
print(f"qk_out   max abs err = {out_err:.3e}  (rel {out_rel:.3e})")
print(f"state    max abs err = {st_err:.3e}")
ok = out_err < 1e-3 and st_err < 1e-5
print("PASS" if ok else "FAIL")
