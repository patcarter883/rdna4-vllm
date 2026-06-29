"""Deep-memory surprise: closed-form analytic backward (the correctness oracle + OOM fix).

The Titans deep memory is a depth-2 MemoryMLP whose weights are per-sequence state:
    pred = gelu(k @ W1) @ W2        W1:[B,d,h]  W2:[B,h,d]   (B = batch*heads)
The "surprise" stored per chunk is the gradient of the weighted MSE loss wrt those weights:
    L = sum_n  lw_n * mean_d( (pred_n - v_n)^2 )
    grad wrt (W1, W2)
lucidrains computes this with torch.func grad+vmap (create_graph) -> the retained graph is what
OOMs at >~6 segments. Here we compute the SAME gradient as explicit matmuls (standard MLP backprop),
which (a) retains no autograd graph -> memory-bounded, (b) is differentiable wrt the adapter's outer
params via plain ops, (c) is the math the Triton kernel will implement. Validated bit-parity vs autograd.
"""
import math
import torch
import torch.nn.functional as F


def gelu(x):
    return F.gelu(x)  # exact (erf) gelu, matching MemoryMLP


def gelu_prime(a):
    Phi = 0.5 * (1.0 + torch.erf(a / math.sqrt(2.0)))
    phi = torch.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    return Phi + a * phi


def mlp_forward(k, W1, W2):
    a = torch.bmm(k, W1)          # [B,n,h]
    z = gelu(a)
    pred = torch.bmm(z, W2)       # [B,n,d]
    return pred, a, z


def analytic_surprise(k, v, lw, W1, W2):
    """Returns (gW1[B,d,h], gW2[B,h,d], pred[B,n,d]) — no autograd graph."""
    pred, a, z = mlp_forward(k, W1, W2)
    D = pred.shape[-1]
    g_pred = (2.0 / D) * lw.unsqueeze(-1) * (pred - v)        # dL/dpred  [B,n,d]
    gW2 = torch.bmm(z.transpose(1, 2), g_pred)               # [B,h,d]
    dz = torch.bmm(g_pred, W2.transpose(1, 2))               # dL/dz [B,n,h]
    da = dz * gelu_prime(a)                                  # dL/da [B,n,h]
    gW1 = torch.bmm(k.transpose(1, 2), da)                   # [B,d,h]
    return gW1, gW2, pred


def autograd_surprise(k, v, lw, W1, W2):
    W1 = W1.clone().requires_grad_(True)
    W2 = W2.clone().requires_grad_(True)
    a = torch.bmm(k, W1)
    z = gelu(a)
    pred = torch.bmm(z, W2)
    loss = (lw * (pred - v).pow(2).mean(dim=-1)).sum()
    gW1, gW2 = torch.autograd.grad(loss, (W1, W2))
    return gW1, gW2


def _parity():
    torch.manual_seed(0)
    B, n, d, h = 4, 16, 32, 128   # B = batch*heads
    dt = torch.float64
    k = torch.randn(B, n, d, dtype=dt)
    v = torch.randn(B, n, d, dtype=dt)
    lw = torch.rand(B, n, dtype=dt)
    W1 = torch.randn(B, d, h, dtype=dt) * 0.1
    W2 = torch.randn(B, h, d, dtype=dt) * 0.1
    gW1a, gW2a, _ = analytic_surprise(k, v, lw, W1, W2)
    gW1b, gW2b = autograd_surprise(k, v, lw, W1, W2)
    dW1 = (gW1a - gW1b).abs().max().item()
    dW2 = (gW2a - gW2b).abs().max().item()
    print(f"[deep-mem] analytic vs autograd surprise  max|dW1|={dW1:.2e}  max|dW2|={dW2:.2e}")
    ok = dW1 < 1e-10 and dW2 < 1e-10
    print(f"[deep-mem] {'PASS' if ok else 'FAIL'}: closed-form surprise is exact "
          f"({'graph-free oracle ready for the kernel' if ok else 'math mismatch'})")
    return ok


if __name__ == "__main__":
    _parity()
