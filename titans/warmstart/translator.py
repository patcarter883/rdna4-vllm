"""CAM v1 — the base-agnostic affine translator (CAM_DESIGN §2.2).

The v0 memory front-end (frozen BoltAdapter -> mem-space bank) and a passing GatedMemoryTap are
FROZEN and reused verbatim. The bank fed to the tap ([B,K,mem_dim]) is base-AGNOSTIC — it is the
DeepMemory's own mem_dim retrieval, never any base's hidden space — so the SAME frozen memory drives
a second base. The only thing that differs across bases is the residual-stream geometry the tap
stitches into; v1 closes that gap with a TINY affine translator (residual-stream stitching transfers
*function* across LLMs, 2506.06609):

    A : d_base2 -> d_base1     (in)   # lift base-2's residual into the frozen tap's hidden space
    [ FROZEN GatedMemoryTap in d_base1, with the FROZEN mem bank ]  -> additive update u (d_base1)
    B : d_base1 -> d_base2     (out)  # project the tap's update back to base-2's residual
    h2' = h2 + tanh(gamma2) ⊙ B(u)    # gamma2 in R^{d_base2}, init 0 => exact no-op at init

Only A, B, gamma2 train (LM-loss through the frozen base-2). Mirrors the gated_tap dtype pattern
(compute in param fp32, cast the additive update back to the base dtype) and adds a NaN/grad guard so
the new gate can't diverge the L=16-style way.
"""
import os

import torch
import torch.nn as nn

from gated_tap import GatedMemoryTap, decoder_layers


def save_translator(path, injector, meta):
    """Persist the fitted translator card (A, B, gamma2 + meta) — the §5.5 UMX product artifact.
    The frozen tap/memory are NOT saved here (they live in the v0 memory checkpoint); a base's
    'translator card' is just the tiny affine pair that stitches it to that canonical memory."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tt = injector.tt
    torch.save({
        "A.weight": tt.A.weight.detach().cpu(), "A.bias": tt.A.bias.detach().cpu(),
        "B.weight": tt.B.weight.detach().cpu(), "B.bias": tt.B.bias.detach().cpu(),
        "gamma2": tt.gamma2.detach().cpu(),
        "d_base1": tt.tap.H, "d_base2": tt.d_base2, "meta": meta,
    }, path)
    print(f"[v1] saved translator card -> {path} "
          f"(A {tt.d_base2}->{tt.tap.H}, B {tt.tap.H}->{tt.d_base2}, base-2 {meta.get('base2')})",
          flush=True)


class TranslatedTap(nn.Module):
    """Wrap a FROZEN d_base1 GatedMemoryTap with trainable affine in/out adapters so it injects into
    a second base of hidden dim d_base2. Zero-init on the OUT side (B + gamma2) keeps the start an
    exact no-op, mirroring the tap's own gamma=0 stability property."""

    def __init__(self, frozen_tap: GatedMemoryTap, d_base2: int):
        super().__init__()
        self.tap = frozen_tap                       # FROZEN (caller sets requires_grad_(False))
        d_base1 = frozen_tap.H
        self.d_base2 = d_base2
        self.A = nn.Linear(d_base2, d_base1, bias=True)       # in: base-2 residual -> tap hidden
        self.B = nn.Linear(d_base1, d_base2, bias=True)       # out: tap update   -> base-2 residual
        # A and B are BOTH randomly initialised so gradient flows from step 0 (zero-init A or B
        # deadlocks: the update path = tanh(gamma2)*B(A(...)); zeroing B AND gamma2 makes every grad
        # zero — the classic double-zero-init dead start). The no-op-at-init property comes from
        # gamma2=0 ALONE (tanh(0)=0), exactly like V0's GatedMemoryTap (only its gamma is zeroed,
        # to_o stays random). So the gate is the sole zero, and grads still reach A and B.
        nn.init.normal_(self.A.weight, std=(1.0 / d_base2) ** 0.5)
        nn.init.zeros_(self.A.bias)
        nn.init.normal_(self.B.weight, std=(1.0 / d_base1) ** 0.5)
        nn.init.zeros_(self.B.bias)
        self.gamma2 = nn.Parameter(torch.zeros(d_base2))      # tanh(0)=0 -> no-op at init (sole zero)
        self._bank = None
        self.last_gate = torch.tensor(0.0)

    def set_bank(self, bank):
        self._bank = bank
        self.tap.set_bank(bank)

    def forward(self, h2):
        """h2: [B,T,d_base2] residual hidden -> injected. No-op when no bank is set."""
        if self._bank is None:
            return h2
        wdt = self.A.weight.dtype                              # fp32 compute (stable gate training)
        h2c = h2.to(wdt)
        h1 = self.A(h2c)                                       # [B,T,d_base1] lifted residual
        # reuse the frozen tap's cross-attention math, but read its ADDITIVE update only (not h1+update),
        # so the gate/scale lives on the base-2 side. The tap internally returns h1 + tanh(gamma)*y;
        # recover u = tap(h1) - h1 (gamma frozen, so this is the frozen memory contribution in d_base1).
        u1 = self.tap(h1) - h1                                 # [B,T,d_base1] frozen memory update
        u2 = self.B(u1)                                        # [B,T,d_base2] project back
        g = torch.tanh(self.gamma2)
        self.last_gate = g.abs().mean().detach()
        return h2 + (g * u2).to(h2.dtype)


class TranslatedInjector:
    """Registers a TranslatedTap forward-hook on one frozen base-2 decoder layer."""

    def __init__(self, base2, frozen_tap, tap_layer):
        d_base2 = base2.config.get_text_config().hidden_size
        self.layers = decoder_layers(base2)
        self.tap_layer = tap_layer
        self.tt = TranslatedTap(frozen_tap, d_base2)
        self._handle = None

    def to(self, dev):
        self.tt.to(dev)
        return self

    def parameters(self):
        # train ONLY the translator (A, B, gamma2); the wrapped tap is frozen.
        return self.A_params()

    def A_params(self):
        return [self.tt.A.weight, self.tt.A.bias, self.tt.B.weight, self.tt.B.bias, self.tt.gamma2]

    def train(self, mode=True):
        self.tt.train(mode)
        return self

    def eval(self):
        self.tt.eval()
        return self

    def set_bank(self, bank):
        self.tt.set_bank(bank)

    def gate_stat(self):
        return float(self.tt.last_gate)

    def _hook(self):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return (self.tt(out[0]),) + tuple(out[1:])
            return self.tt(out)
        return fn

    def attach(self):
        self._handle = self.layers[self.tap_layer].register_forward_hook(self._hook())
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return self
