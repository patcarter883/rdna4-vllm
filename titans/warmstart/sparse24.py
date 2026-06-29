"""2:4-by-design (SR-STE) structured sparsity for the CAM serve-weight de-risk (build step 0).

The serve-path tap projections (to_q/to_k/to_v/to_o of the GatedMemoryTap) are the d_hub×d_hub
weights that SWMMAC accelerates. We need to know whether training those weights 2:4-sparse FROM INIT
(NOT prune-after-dense, the +25% PPL path) holds the v0 recall fidelity vs a dense reference.

SR-STE (Sparse-Refined Straight-Through Estimator, NVIDIA 2021):
  - forward uses the 2:4-masked weight W_s = M ⊙ W (M keeps the 2-of-every-4-along-input-dim largest
    |W|), so the kernel only ever sees a 2:4-legal weight;
  - backward is straight-through (grad flows to ALL of W, masked or not);
  - PLUS an SR-STE decay term lambda·(1−M)⊙W that pushes the pruned (masked-out) weights toward zero,
    so the mask stabilises and the dense->sparse gap does not reopen during training.

The 2:4 group runs along the INPUT (last) dimension of nn.Linear's [out, in] weight — in%4==0.
"""
import torch
import torch.nn as nn


def topk2of4_mask(w):
    """w: [out, in] (in%4==0). Returns a {0,1} mask keeping the 2 largest-|w| of every 4 along `in`."""
    out_f, in_f = w.shape
    g = w.abs().reshape(out_f, in_f // 4, 4)              # [out, in/4, 4]
    idx = g.topk(2, dim=-1).indices                       # indices of the 2 largest per group
    m = torch.zeros_like(g)
    m.scatter_(-1, idx, 1.0)
    return m.reshape(out_f, in_f)


class Mask24Linear(nn.Module):
    """bias-free Linear whose weight is 2:4-sparse-by-design (SR-STE). Same param shape as nn.Linear,
    so it loads/saves like one and the param count is identical (storage saving is a serve concern)."""

    def __init__(self, in_f, out_f, srste_lambda=2e-4):
        super().__init__()
        assert in_f % 4 == 0, f"2:4 needs in_features%4==0, got {in_f}"
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)   # match nn.Linear default
        self.srste_lambda = srste_lambda

    def masked_weight(self):
        w = self.weight
        m = topk2of4_mask(w.detach())                       # mask from |W| (no grad through the mask)
        w_s = w * m                                          # forward weight (2:4-legal)
        # straight-through: forward uses w_s, backward sees grad as if dense (grad flows to all of W).
        w_ste = w + (w_s - w).detach()
        if self.training and self.srste_lambda:
            # SR-STE decay on the PRUNED weights: add lambda·(1−M)⊙W into the forward graph so its
            # gradient is lambda·(1−M)⊙W — i.e. an L2 pull on masked-out weights toward zero. Detach the
            # value so it does not perturb the forward numerics (only the gradient), keeping the
            # gate=0 init an exact no-op like the dense tap.
            decay = (self.srste_lambda * ((1.0 - m) * w))
            w_ste = w_ste + (decay - decay.detach())
        return w_ste

    def forward(self, x):
        return torch.nn.functional.linear(x, self.masked_weight())

    @torch.no_grad()
    def sparsity_report(self):
        m = topk2of4_mask(self.weight)
        return 1.0 - m.mean().item()                        # fraction of weights that are zero (~0.5)
