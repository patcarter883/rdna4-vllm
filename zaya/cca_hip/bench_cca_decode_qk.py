"""Standalone latency benchmark for the fused CCA decode kernel.

Mirrors the ZAYA decode shapes (S=16 tokens, 10 heads, d=128). Times the kernel
in a serial loop (each call's conv_states write feeds the next, matching the
autoregressive decode dependency) and reports us/call via hip events.

    GPU_ARCHS=gfx1201 python bench_cca_decode_qk.py
"""
import math
import sys

import torch
import cca_op  # noqa: F401  (registers torch.ops.zaya_cca)

torch.manual_seed(0)

# ZAYA CCA dims
C, d, K0, K1, TP = 1280, 128, 2, 2, 2
num_q, num_k = 8, 2
gqa = num_q // num_k
latent_q = num_q * d
S = 16
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
sqrt_d = math.sqrt(d)
dev = "cuda"

qk_new = torch.randn(S, C, device=dev)
conv_states = torch.randn(NB, C, TP, device=dev)
w0 = torch.randn(C, K0, device=dev)
b0 = torch.randn(C, device=dev)
w1 = torch.randn(C, d, K1, device=dev) * 0.05  # keep state bounded over the loop
b1 = torch.randn(C, device=dev)
temp_eff = torch.rand(num_k, device=dev) + 0.5
# kernel expects w1 transposed to [H, d_in, d_out, K1] for coalesced reads
H = C // d
w1 = w1.view(H, d, d, K1).permute(0, 2, 1, 3).contiguous()

# scatter the 16 decode slots across the cache (cache-realistic, non-contiguous)
slot = torch.randint(1, NB, (S,), device=dev, dtype=torch.long)
is_pad = torch.zeros(S, device=dev, dtype=torch.bool)


def once():
    return torch.ops.zaya_cca.cca_decode_qk(
        qk_new, conv_states, slot, is_pad, w0, b0, w1, b1, temp_eff,
        num_q, gqa, latent_q, sqrt_d)


def time_loop(iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        once()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters  # us/call


# warmup
for _ in range(50):
    once()
torch.cuda.synchronize()

print(f"NB={NB}  grid=({num_q+num_k},{S})={(num_q+num_k)*S} blocks  blockDim={d}")

# (a) end-to-end dispatched (includes python/torch op launch overhead)
us_disp = min(time_loop(2000) for _ in range(3))
print(f"dispatched : {us_disp:.2f} us/call")

# (b) HIP-graph capture: back-to-back kernel execution, no host dispatch
#     (matches production, where decode runs under cudagraph). Capture REPS
#     kernel execs into one graph, replay, divide.
REPS = 100
g = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
with torch.cuda.graph(g):
    for _ in range(REPS):
        once()
# warmup replays
for _ in range(10):
    g.replay()
torch.cuda.synchronize()

NREPLAY = 50
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(NREPLAY):
    g.replay()
end.record()
torch.cuda.synchronize()
us_graph = start.elapsed_time(end) * 1000.0 / (NREPLAY * REPS)
print(f"graphed    : {us_graph:.2f} us/call  <-- production-relevant kernel time")
