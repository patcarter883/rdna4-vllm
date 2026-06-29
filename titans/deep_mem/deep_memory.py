"""Stage 3 — the graph-free deep Titans memory module (the OOM fix).

Assembles the two parity-exact foundations into a drop-in replacement for lucidrains `NeuralMemory`:
  - analytic surprise         (deep_mem_analytic.analytic_surprise)  — closed-form MLP backward, no
                                                                        torch.func create_graph
  - chunk store recurrence    (store_recurrence._gated_scan)         — momentum + decay, parallel scan
  - retrieve                  (mlp_forward)                          — pred = gelu(q@W1)@W2

Why this fixes the OOM (M2 finding): lucidrains computes per-chunk surprise with `torch.func.grad`
under `create_graph=True`, which retains the *inner* backward graph so the outer LM loss can backprop
through it — that doubled graph is what OOMs past ~6 segments. Here the surprise is plain forward
matmuls (single first-order graph wrt the adapter params, no second-order graph), and the cross-segment
carry is sequential `forward` calls each scanning only this segment's few chunks — so the scan never
grows and the per-segment graph is small. Memory scales to many segments.

Memory state = per-(batch·head) depth-2 MLP weights (W1,W2) + the momentum carries (S1,S2). The
adapter's projections/gates (this module's nn.Parameters) are the trained params; W/S are test-time
STATE folded by the surprise update.

Shapes (B=batch, H=heads, d=head_dim=dim//H, h=hidden=d*expansion, B2=B*H):
  token proj  k,v,q : [B,N,dim] -> [B2,N,d]
  weights     W1:[B2,d,h]  W2:[B2,h,d]      momentum  S1:[B2,d,h]  S2:[B2,h,d]
  per-chunk gates θ/η/α : [T,B2,1]   per-token loss-weight lw : [B2,N]
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # works both as a package import and when deep_mem/ is directly on sys.path
    from .deep_mem_analytic import analytic_surprise, mlp_forward
    from .store_recurrence import _gated_scan
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from deep_mem_analytic import analytic_surprise, mlp_forward
    from store_recurrence import _gated_scan


@dataclass
class DeepMemState:
    """Test-time memory state, carried across segments. All fp32, no autograd graph at segment start."""
    W1: torch.Tensor  # [B2,d,h]
    W2: torch.Tensor  # [B2,h,d]
    S1: torch.Tensor  # [B2,d,h] momentum
    S2: torch.Tensor  # [B2,h,d] momentum
    conv_state: torch.Tensor = None  # [B, K-1, dim] last input tokens for the streaming causal conv
                                     # (None when conv_kernel<=1)

    def detach(self):
        cs = self.conv_state.detach() if self.conv_state is not None else None
        return DeepMemState(self.W1.detach(), self.W2.detach(),
                            self.S1.detach(), self.S2.detach(), cs)


def _scan_final(g, theta, eta, alpha, M0, S0):
    """One window of the store recurrence; returns (M_T, S_T) — the carried finals.
        S_c = eta_c   * S_{c-1} - theta_c * g_c       (S_{-1}=S0)
        M_c = (1-a_c) * M_{c-1} + S_c                 (M_{-1}=M0)
    g:[T,B2,P]  theta/eta/alpha:[T,B2,1]  M0/S0:[B2,P].  T is per-segment (small) -> cumprod stable.
    """
    Sfull = _gated_scan(eta, -theta * g, S0)        # [T,B2,P]
    Mfull = _gated_scan(1.0 - alpha, Sfull, M0)     # [T,B2,P]
    return Mfull[-1], Sfull[-1]


class DeepMemory(nn.Module):
    """Graph-free deep Titans memory. API mirrors what TitansMemoryAdapter needs:
        init_state(batch) -> DeepMemState
        retrieve(q, state) -> [B,K,dim]
        forward(x, state)  -> DeepMemState     (ingest a segment; surprise store update)
    """

    def __init__(self, dim=512, heads=4, chunk_size=16, expansion=4.0,
                 theta_max=1.0, init_scale=1.0, conv_kernel=4, conv_init="identity"):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim, self.heads = dim, heads
        self.head_dim = dim // heads
        self.hidden = int(self.head_dim * expansion)
        self.chunk_size = chunk_size
        self.theta_max = theta_max
        self.conv_kernel = conv_kernel

        # Depthwise CAUSAL conv1d over the dim channels, applied to the INGEST tokens before to_k/to_v/
        # to_q so a value token's projections can see its preceding key (canonical GDN/Mamba short conv).
        # DeepMemory ingests context-free token embeddings with no other temporal mixing, so without this
        # it provably cannot bind interleaved [k,v] pairs (recall_mqar). Identity-initialised (last tap=1)
        # -> behaves as no-conv at step 0 and learns the mixing. NOT applied to retrieve() queries (they
        # are independent probes). State = the last K-1 ingested tokens, carried across segments.
        if conv_kernel and conv_kernel > 1:
            self.conv = nn.Conv1d(dim, dim, kernel_size=conv_kernel, groups=dim, bias=True, padding=0)
            with torch.no_grad():
                if conv_init == "identity":   # out[t] = x[t] at init -> behaves as no-conv, learns mixing
                    self.conv.weight.zero_()
                    self.conv.weight[:, :, -1] = 1.0
                    self.conv.bias.zero_()
                elif conv_init == "mix":      # box filter: out[t] = mean of last K tokens -> a value token
                    self.conv.weight.fill_(1.0 / conv_kernel)   # already sees its preceding key at step 0
                    self.conv.bias.zero_()
                elif conv_init == "random":   # standard small-random short conv (GDN/Mamba default)
                    pass                      # nn.Conv1d default kaiming-uniform init
                else:
                    raise ValueError(f"unknown conv_init {conv_init!r}")
        else:
            self.conv = None

        # token -> memory key / value / query (per-head split after projection)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, dim, bias=False)

        # per-token loss weight (adaptive surprise weighting) and per-chunk gates from chunk reps.
        # gate projection weights zero-init -> gates start at their bias (stable, constant); the
        # data-dependence is learned. biases chosen for a gentle initial write.
        self.to_lw = nn.Linear(dim, heads)
        self.to_theta = nn.Linear(dim, heads)   # adaptive lr  (0, theta_max) via theta_max*sigmoid
        self.to_eta = nn.Linear(dim, heads)     # momentum     (0,1) via sigmoid
        self.to_alpha = nn.Linear(dim, heads)   # decay        (0,1) via sigmoid
        for lin, b in ((self.to_theta, -2.0), (self.to_eta, 0.0), (self.to_alpha, -2.0), (self.to_lw, 0.0)):
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, b)

        # learned initial memory weights (per head); broadcast across batch at init_state
        self.W1_0 = nn.Parameter(torch.randn(heads, self.head_dim, self.hidden)
                                 * (init_scale / math.sqrt(self.head_dim)))
        self.W2_0 = nn.Parameter(torch.randn(heads, self.hidden, self.head_dim)
                                 * (init_scale / math.sqrt(self.hidden)))

    # ---- helpers ----------------------------------------------------------
    def _heads(self, t):
        """[B,N,dim] -> [B*H,N,head_dim] with flat index b*H+h."""
        B, N, _ = t.shape
        return (t.reshape(B, N, self.heads, self.head_dim)
                 .permute(0, 2, 1, 3).reshape(B * self.heads, N, self.head_dim))

    def init_state(self, batch):
        H, d, h = self.heads, self.head_dim, self.hidden
        W1 = self.W1_0.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * H, d, h).contiguous()
        W2 = self.W2_0.unsqueeze(0).expand(batch, -1, -1, -1).reshape(batch * H, h, d).contiguous()
        S1 = torch.zeros_like(W1)
        S2 = torch.zeros_like(W2)
        conv_state = None
        if self.conv is not None:
            conv_state = self.W1_0.new_zeros(batch, self.conv_kernel - 1, self.dim)
        return DeepMemState(W1, W2, S1, S2, conv_state)

    # ---- ingest-path causal conv -----------------------------------------
    def _causal_conv(self, x, conv_state):
        """x:[B,N,dim] -> (mixed[B,N,dim], new_conv_state[B,K-1,dim]). Streaming-causal: the carried
        conv_state (last K-1 tokens of the previous segment) is prepended as left context so out[t]
        depends only on x[t-K+1 .. t] across the segment boundary. No detach here — the carry threads
        grad like W1/W2 do; DeepMemState.detach() handles truncation."""
        if self.conv is None:
            return x, None
        B, N, _ = x.shape
        K = self.conv_kernel
        if conv_state is None:
            conv_state = x.new_zeros(B, K - 1, self.dim)
        xp = torch.cat([conv_state, x], dim=1)                    # [B, K-1+N, dim]
        y = self.conv(xp.transpose(1, 2)).transpose(1, 2)        # valid conv -> [B, N, dim]
        new_state = xp[:, -(K - 1):, :]                          # last K-1 input tokens for next segment
        return y, new_state

    # ---- retrieve ---------------------------------------------------------
    def retrieve(self, q, state):
        """q:[B,K,dim] -> [B,K,dim].  out = gelu(q@W1)@W2 with the current memory weights."""
        B, K, _ = q.shape
        qh = self._heads(self.to_q(q))                       # [B2,K,d]
        pred, _, _ = mlp_forward(qh, state.W1, state.W2)     # [B2,K,d]
        return (pred.reshape(B, self.heads, K, self.head_dim)
                    .permute(0, 2, 1, 3).reshape(B, K, self.dim))

    # ---- ingest (surprise store update) -----------------------------------
    def forward(self, x, state):
        """x:[B,N,dim] (fp32) -> new DeepMemState. Surprise of every chunk is taken at the incoming
        window-start weights (state.W*), then folded by the momentum+decay scan."""
        B, N, _ = x.shape
        H, d, h, cs = self.heads, self.head_dim, self.hidden, self.chunk_size
        B2 = B * H

        # temporal mixing on the ingest tokens BEFORE k/v/q projections (so a value token can see its
        # preceding key); conv_state carries the streaming left-context across segments
        x, new_conv_state = self._causal_conv(x, state.conv_state)

        # pad up to a whole number of chunks; padded tokens get lw=0 (contribute no surprise)
        pad = (-N) % cs
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Np = x.shape[1]
        T = Np // cs

        k = self._heads(self.to_k(x))                        # [B2,Np,d]
        v = self._heads(self.to_v(x))
        lw = F.softplus(self.to_lw(x))                       # [B,Np,H]
        lw = lw.permute(0, 2, 1).reshape(B2, Np).clone()
        if pad:
            lw[:, Np - pad:] = 0.0

        # chunk reps (mean over chunk) -> per-chunk gates [T,B2,1]
        rep = x.reshape(B, T, cs, self.dim).mean(dim=2)      # [B,T,dim]
        def gate(lin, act):
            g = act(lin(rep))                                # [B,T,H]
            return g.permute(1, 0, 2).reshape(T, B2, 1)
        theta = gate(self.to_theta, lambda z: self.theta_max * torch.sigmoid(z))
        eta = gate(self.to_eta, torch.sigmoid)
        alpha = gate(self.to_alpha, torch.sigmoid)

        # per-chunk analytic surprise at the window-start weights, all chunks batched over B2*T
        kc = k.reshape(B2 * T, cs, d)
        vc = v.reshape(B2 * T, cs, d)
        lwc = lw.reshape(B2 * T, cs)
        W1e = state.W1.unsqueeze(1).expand(-1, T, -1, -1).reshape(B2 * T, d, h)
        W2e = state.W2.unsqueeze(1).expand(-1, T, -1, -1).reshape(B2 * T, h, d)
        gW1, gW2, _ = analytic_surprise(kc, vc, lwc, W1e, W2e)   # [B2*T,d,h], [B2*T,h,d]

        g1 = gW1.reshape(B2, T, d * h).permute(1, 0, 2)      # [T,B2,P1]
        g2 = gW2.reshape(B2, T, h * d).permute(1, 0, 2)      # [T,B2,P2]

        M1, S1 = _scan_final(g1, theta, eta, alpha, state.W1.reshape(B2, d * h), state.S1.reshape(B2, d * h))
        M2, S2 = _scan_final(g2, theta, eta, alpha, state.W2.reshape(B2, h * d), state.S2.reshape(B2, h * d))

        return DeepMemState(M1.reshape(B2, d, h), M2.reshape(B2, h, d),
                            S1.reshape(B2, d, h), S2.reshape(B2, h, d), new_conv_state)
