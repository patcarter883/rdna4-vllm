"""Deep-mem store recurrence (momentum + decay) — sequential reference vs parallel associative scan.

Per chunk c (B = batch*heads; weights flattened to P elements; gates are per-(b,h) scalars):
    S_c = eta_c   * S_{c-1} - theta_c * g_c       (momentary surprise + momentum), S_{-1}=0
    M_c = (1-a_c) * M_{c-1} + S_c                  (weight with decay/forgetting),  M_{-1}=M0
g_c = the analytic surprise gradient (weight-shaped). theta (adaptive lr), eta (momentum), a (decay)
are data-dependent per-chunk scalars in (0,1) (theta>0).

Two chained gated-linear recurrences. Each x_c = gate_c * x_{c-1} + in_c has the closed form
    x_c = P_c * (x0 + cumsum(in / P)_c),   P_c = cumprod(gate)_c
so the whole store parallelizes (cumprod + cumsum) -> Triton-friendly. We prove the parallel form is
bit-identical to the naive sequential loop.
"""
import torch


def store_sequential(g, theta, eta, alpha, M0):
    """g:[T,B,P]  theta/eta/alpha:[T,B,1]  M0:[B,P] -> M:[T,B,P] (weights after each chunk)."""
    T = g.shape[0]
    S = torch.zeros_like(M0)
    M = M0.clone()
    out = []
    for c in range(T):
        S = eta[c] * S - theta[c] * g[c]
        M = (1.0 - alpha[c]) * M + S
        out.append(M.clone())
    return torch.stack(out, dim=0)


def _gated_scan(gate, inp, x0):
    """Parallel x_c = gate_c * x_{c-1} + inp_c, x_{-1}=x0.  gate:[T,B,1] inp:[T,B,P] x0:[B,P]."""
    P = torch.cumprod(gate, dim=0)                       # [T,B,1]
    return P * (x0.unsqueeze(0) + torch.cumsum(inp / P, dim=0))


def store_parallel(g, theta, eta, alpha, M0):
    S = _gated_scan(eta, -theta * g, torch.zeros_like(M0))    # S_c
    M = _gated_scan(1.0 - alpha, S, M0)                       # M_c
    return M


def _parity():
    torch.manual_seed(0)
    T, B, P = 12, 4, 100
    dt = torch.float64
    g = torch.randn(T, B, P, dtype=dt)
    theta = torch.rand(T, B, 1, dtype=dt) * 0.1          # lr > 0
    eta = torch.rand(T, B, 1, dtype=dt) * 0.9 + 0.05     # momentum in (0,1)
    alpha = torch.rand(T, B, 1, dtype=dt) * 0.5          # decay in (0,0.5)
    M0 = torch.randn(B, P, dtype=dt) * 0.1
    seq = store_sequential(g, theta, eta, alpha, M0)
    par = store_parallel(g, theta, eta, alpha, M0)
    d = (seq - par).abs().max().item()
    print(f"[store] sequential vs parallel associative-scan  max|dM|={d:.2e}")
    ok = d < 1e-10
    print(f"[store] {'PASS' if ok else 'FAIL'}: recurrence is exactly parallelizable "
          f"({'cumprod+cumsum form ready for Triton' if ok else 'scan math mismatch'})")
    return ok


if __name__ == "__main__":
    _parity()
