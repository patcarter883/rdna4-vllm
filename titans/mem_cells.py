"""Drop-in linear-attention memory cells for the MQAR expressiveness ladder.

One module, one flag — `DecoupledGateCell(channelwise=...)` — gives BOTH linear arms with a
guaranteed one-variable difference (gate dimensionality only), because GDN-2 (arXiv 2605.22791)
*reduces to* the scalar gated-delta cell when its gates are tied to scalars (paper Eq. 10 limits):

  channelwise=False  -> arm 2  "gated-delta / scalar gate"      (b,w,alpha are per-head scalars)
  channelwise=True   -> arm 3  "GDN-2 / decoupled channel gates" (b,w,alpha are per-channel vectors)

Recurrence (GDN-2 Eq. 9, the read-intermediate form), per (batch, head), state S in R^{d_k x d_v}:
  S_bar = Diag(alpha_t) S_{t-1}              # channel-wise decay on the key axis
  e_t   = b_t (.) k_t                        # erase key (key-side gate)
  r_t   = S_bar^T e_t                        # what the erase-key currently reads  (in R^{d_v})
  z_t   = w_t (.) v_t                        # write value (value-side gate)
  S_t   = S_bar + k_t (z_t - r_t)^T          # decoupled erase/write delta update
  o_t   = S_t^T q_t                          # retrieval

Gates (Eq. 11-12): b_t=sigmoid(W_b x), w_t=sigmoid(W_w x), alpha_t=exp(-exp(a)*softplus(W_f x)).
Scalar arm projects each gate to 1 dim/head and broadcasts (= the paper's KDA/GatedDeltaNet limit).

This is a CORRECTNESS-FIRST pure-torch sequential scan (no FLA/KDA chunked kernel) — sufficient to
get the MQAR recall numbers for the gate. The fast chunkwise-WY path is downstream work, not built
here. It quacks like titans_pytorch.NeuralMemory.forward so it slots into the MAC memory slot
unchanged: forward(qkv[3,b,n,dim], state=None, prev_weights=None) -> (retrieved[b,n,dim], cache).
"""

from __future__ import annotations

from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

# minimal stand-in for NeuralMemState; MAC only reads `.updates` when neural_mem_weight_residual
# is on (we keep it OFF for these arms), and otherwise just stores the cache in a list.
CellCache = namedtuple("CellCache", ["updates"])


class DecoupledGateCell(nn.Module):
    def __init__(self, dim, heads=4, head_dim=32, channelwise=False, l2norm_qk=True):
        super().__init__()
        self.h, self.d = heads, head_dim
        self.inner = heads * head_dim
        self.channelwise = channelwise
        self.l2norm_qk = l2norm_qk

        self.to_q = nn.Linear(dim, self.inner, bias=False)
        self.to_k = nn.Linear(dim, self.inner, bias=False)
        self.to_v = nn.Linear(dim, self.inner, bias=False)
        self.to_o = nn.Linear(self.inner, dim, bias=False)

        gate_out = self.inner if channelwise else heads      # per-channel vs per-head scalar
        self.to_b = nn.Linear(dim, gate_out, bias=True)      # erase gate (key-side)
        self.to_f = nn.Linear(dim, gate_out, bias=True)      # forget/decay pre-activation
        if channelwise:
            self.to_w = nn.Linear(dim, gate_out, bias=True)  # independent write gate (value-side)
        else:
            self.to_w = None                                 # scalar arm ties write to erase (= single gate)
        # learnable per-(channel|head) decay scale `a` in alpha = exp(-exp(a)*softplus(W_f x))
        self.a = nn.Parameter(torch.zeros(gate_out))

    def _heads(self, t):                                      # [b,n,inner] -> [b,n,h,d]
        b, n, _ = t.shape
        return t.reshape(b, n, self.h, self.d)

    def _gate(self, lin, x):                                  # -> [b,n,h,d] (broadcast scalar arm)
        g = lin(x)
        if self.channelwise:
            return self._heads(g)
        return g.unsqueeze(-1)                               # [b,n,h,1] broadcasts over d

    def forward(self, qkv, state=None, prev_weights=None, **kwargs):
        x = qkv[0] if (torch.is_tensor(qkv) and qkv.ndim == 4) else qkv  # MAC stacks 3 identical views
        b, n, _ = x.shape
        q = self._heads(self.to_q(x)); k = self._heads(self.to_k(x)); v = self._heads(self.to_v(x))
        if self.l2norm_qk:
            q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1)

        bg = torch.sigmoid(self._gate(self.to_b, x))                     # erase gate  [b,n,h,*]
        wg = torch.sigmoid(self._gate(self.to_w, x)) if self.channelwise else bg  # write gate
        a = self.a.reshape(self.h, self.d) if self.channelwise else self.a.unsqueeze(-1)  # [h,d] or [h,1]
        alpha = torch.exp(-torch.exp(a) * F.softplus(self._gate(self.to_f, x)))   # decay [b,n,h,*]

        S = x.new_zeros(b, self.h, self.d, self.d)            # [b,h,d_k,d_v]
        outs = []
        for t in range(n):
            kt, vt, qt = k[:, t], v[:, t], q[:, t]            # [b,h,d]
            Sbar = alpha[:, t].unsqueeze(-1) * S              # decay rows (key axis): [b,h,d_k,1]*S
            et = bg[:, t] * kt                                # [b,h,d_k]
            rt = einsum("bhkv,bhk->bhv", Sbar, et)            # S_bar^T e  -> [b,h,d_v]
            zt = wg[:, t] * vt                                # [b,h,d_v]
            S = Sbar + einsum("bhk,bhv->bhkv", kt, zt - rt)   # delta update
            outs.append(einsum("bhkv,bhk->bhv", S, qt))       # S^T q -> [b,h,d_v]
        o = torch.stack(outs, dim=1).reshape(b, n, self.inner)
        return self.to_o(o), CellCache(updates=None)


def swap_memory_cells(model, cell_factory):
    """Replace each NeuralMemory in a built MAC with cell_factory() (same dim slot). Returns count.
    MAC layer tuple = [mem_hyper_conn, attn_hc, ff_hc, mem_qkv_selector, mem, attn, ff]; index 4=mem."""
    n = 0
    for layer in model.layers:
        if layer[4] is not None and type(layer[4]).__name__ == "NeuralMemory":
            layer[4] = cell_factory()
            n += 1
    return n
