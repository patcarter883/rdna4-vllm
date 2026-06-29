"""Stage 3 parity — the assembled DeepMemory.forward must match an INDEPENDENT reference.

The module path: analytic surprise (matmul backward) + parallel cumprod/cumsum scan.
The reference:   autograd surprise (torch.autograd.grad) + naive sequential for-loop scan.
Two disjoint implementations of the same math -> agreement to ~1e-10 (fp64) is a real correctness gate,
not a tautology. Also checks retrieve() against a hand-rolled gelu(q@W1)@W2.

CPU-only, no GPU. Run: python deep_mem/stage3_parity.py  (from titans/), or python stage3_parity.py.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deep_memory import DeepMemory  # noqa: E402
from deep_mem_analytic import autograd_surprise, gelu  # noqa: E402
from store_recurrence import store_sequential  # noqa: E402


def reference_forward(mem: DeepMemory, x, state):
    """Recompute mem.forward via the OTHER code path (autograd surprise + sequential scan)."""
    B, N, _ = x.shape
    H, d, h, cs = mem.heads, mem.head_dim, mem.hidden, mem.chunk_size
    B2 = B * H
    # same ingest-path causal conv as mem.forward (shared layer; parity tests the surprise+scan math,
    # the conv is deterministic preprocessing both paths apply)
    x, _ = mem._causal_conv(x, state.conv_state)
    pad = (-N) % cs
    if pad:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad))
    Np = x.shape[1]
    T = Np // cs

    k = mem._heads(mem.to_k(x))
    v = mem._heads(mem.to_v(x))
    lw = torch.nn.functional.softplus(mem.to_lw(x)).permute(0, 2, 1).reshape(B2, Np).clone()
    if pad:
        lw[:, Np - pad:] = 0.0

    rep = x.reshape(B, T, cs, mem.dim).mean(dim=2)
    theta = (mem.theta_max * torch.sigmoid(mem.to_theta(rep))).permute(1, 0, 2).reshape(T, B2, 1)
    eta = torch.sigmoid(mem.to_eta(rep)).permute(1, 0, 2).reshape(T, B2, 1)
    alpha = torch.sigmoid(mem.to_alpha(rep)).permute(1, 0, 2).reshape(T, B2, 1)

    # per-chunk surprise via autograd at the window-start weights (state.W1/W2)
    g1, g2 = [], []
    for c in range(T):
        kc = k[:, c * cs:(c + 1) * cs]
        vc = v[:, c * cs:(c + 1) * cs]
        lwc = lw[:, c * cs:(c + 1) * cs]
        gW1, gW2 = autograd_surprise(kc, vc, lwc, state.W1, state.W2)
        g1.append(gW1.reshape(B2, d * h))
        g2.append(gW2.reshape(B2, h * d))
    g1 = torch.stack(g1, 0)   # [T,B2,P1]
    g2 = torch.stack(g2, 0)

    # sequential scan (the for-loop reference), final weights
    M1 = store_sequential(g1, theta, eta, alpha, state.W1.reshape(B2, d * h))[-1]
    M2 = store_sequential(g2, theta, eta, alpha, state.W2.reshape(B2, h * d))[-1]
    return M1.reshape(B2, d, h), M2.reshape(B2, h, d)


def _parity():
    torch.manual_seed(0)
    B, N, dim, H, cs = 2, 37, 32, 4, 8     # N not a multiple of cs -> exercises padding
    mem = DeepMemory(dim=dim, heads=H, chunk_size=cs, expansion=4.0).double()
    # randomize the gate projections so gates aren't trivially constant (real per-chunk variation)
    for lin in (mem.to_theta, mem.to_eta, mem.to_alpha, mem.to_lw):
        torch.nn.init.normal_(lin.weight, std=0.5)
    # randomize the causal conv away from its identity init so the parity actually exercises the conv
    if mem.conv is not None:
        torch.nn.init.normal_(mem.conv.weight, std=0.5)
        torch.nn.init.normal_(mem.conv.bias, std=0.1)
    x = torch.randn(B, N, dim, dtype=torch.float64)
    state = mem.init_state(B)

    newW1, newW2 = mem.forward(x, state).W1, mem.forward(x, state).W2
    refW1, refW2 = reference_forward(mem, x, state)
    dW1 = (newW1 - refW1).abs().max().item()
    dW2 = (newW2 - refW2).abs().max().item()
    print(f"[stage3] forward  vs autograd+sequential ref  max|dW1|={dW1:.2e}  max|dW2|={dW2:.2e}")

    # retrieve vs hand-rolled gelu(q@W1)@W2
    q = torch.randn(B, 5, dim, dtype=torch.float64)
    out = mem.retrieve(q, state)
    qh = mem._heads(mem.to_q(q))
    man = (gelu(torch.bmm(qh, state.W1)) @ state.W2)
    man = man.reshape(B, H, 5, mem.head_dim).permute(0, 2, 1, 3).reshape(B, 5, dim)
    dR = (out - man).abs().max().item()
    print(f"[stage3] retrieve vs hand-rolled gelu(q@W1)@W2  max|d|={dR:.2e}")

    ok = dW1 < 1e-9 and dW2 < 1e-9 and dR < 1e-9
    print(f"[stage3] {'PASS' if ok else 'FAIL'}: assembled DeepMemory matches the independent reference "
          f"({'graph-free module is correct -> ready to drop into M2' if ok else 'assembly mismatch'})")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _parity() else 1)
