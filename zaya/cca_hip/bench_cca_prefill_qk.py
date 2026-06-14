"""Latency: fused CCA prefill kernel vs the eager flat-conv path it replaces.

Eager path = what cca.py runs per CCA layer on prefill: scatter into a flat conv
buffer -> two-stage nn.Conv1d (depthwise + grouped, MIOpen) -> gather -> eager
grouped-means + fp32 RMS-norm. The kernel folds all of it into one launch.

    GPU_ARCHS=gfx1201 python bench_cca_prefill_qk.py [P]
"""
import math
import sys

import torch
import torch.nn as nn
import cca_op  # noqa: F401

torch.manual_seed(0)
C, d, K0, K1, TP = 1280, 128, 2, 2, 2
num_q, num_k = 8, 2
gqa = num_q // num_k
latent_q, latent_k = num_q * d, num_k * d
H = num_q + num_k
sqrt_d, eps = math.sqrt(d), 1e-12
dev = "cuda"

P = int(sys.argv[1]) if len(sys.argv) > 1 else 2048   # single-request prefill len
NB = 64

qk_new = torch.randn(P, C, device=dev)
conv_states0 = torch.randn(NB, C, TP, device=dev)
w0 = torch.randn(C, K0, device=dev); b0 = torch.randn(C, device=dev)
w1 = torch.randn(C, d, K1, device=dev) * 0.05; b1 = torch.randn(C, device=dev)
temp_eff = torch.rand(num_k, device=dev) + 0.5
w1t = w1.view(H, d, d, K1).permute(0, 2, 1, 3).contiguous()

# single request, has_init
slot0 = 5
init_states = conv_states0[slot0:slot0 + 1].clone()           # [1, C, TP]
seg_pos = torch.arange(P, device=dev, dtype=torch.int32)
req_id = torch.zeros(P, device=dev, dtype=torch.int32)
slot = torch.full((P,), slot0, device=dev, dtype=torch.long)
is_last = torch.zeros(P, device=dev, dtype=torch.bool); is_last[-1] = True

# eager conv module (matches cca.py conv_qk)
conv = nn.Sequential(
    nn.Conv1d(C, C, K0, groups=C),
    nn.Conv1d(C, C, K1, groups=H),
).to(dev)
with torch.no_grad():
    conv[0].weight.copy_(w0.unsqueeze(1)); conv[0].bias.copy_(b0)
    conv[1].weight.copy_(w1); conv[1].bias.copy_(b1)
temp_v = temp_eff.view(1, num_k, 1)


def eager():
    flat_len = P + TP
    flat_in = qk_new.new_zeros((C, flat_len))
    flat_in[:, TP:] = qk_new.t()
    flat_in[:, :TP] = init_states[0]
    flat_out = conv(flat_in.unsqueeze(0))[0]                   # [C, P]
    conv_out = flat_out.t()                                    # [P, C]
    q = conv_out[:, :latent_q].view(P, num_q, d).float()
    k = conv_out[:, latent_q:].view(P, num_k, d).float()
    qpre = qk_new[:, :latent_q].view(P, num_k, gqa, d)
    kbase = qk_new[:, latent_q:].view(P, num_k, d)
    q.view(P, num_k, gqa, d).add_(qpre, alpha=0.5).add_(kbase.unsqueeze(2), alpha=0.5)
    k.add_(qpre.mean(2), alpha=0.5).add_(kbase, alpha=0.5)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + eps) * sqrt_d
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + eps) * sqrt_d * temp_v
    return q, k


def kernel():
    ks = conv_states0.clone()
    return torch.ops.zaya_cca.cca_prefill_qk(
        qk_new, ks, init_states, seg_pos, req_id, slot, is_last,
        w0, b0, w1t, b1, temp_eff, num_q, gqa, latent_q, sqrt_d)


def time_fn(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters   # us


us_e = time_fn(eager)
us_k = time_fn(kernel)
print(f"P={P}  prefill (per CCA layer call):")
print(f"  eager flat-conv path : {us_e:8.2f} us")
print(f"  fused prefill kernel : {us_k:8.2f} us   ({us_e / us_k:.2f}x faster)")
