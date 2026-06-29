"""GatedMemoryTap — zero-init Memory-as-Gate (MAG) injection for the CAM v0 falsifier.

Replaces boltA's input-embeds prefix (MAC) with an additive, zero-initialised gated cross-attention
tap on one (or more) FROZEN decoder layer(s). At init the gate is exactly 0 -> the base is
bit-identical to baseline; training opens the gate only as far as LM loss rewards (the load-bearing
stability property, ref 2603.16413).

   q = Wq h_l ; k = Wk bank ; v = Wv bank
   a = softmax(q k^T / sqrt(d_head))         # cross-attention over the K memory slots
   y = Wo (a v)
   h_l' = h_l + tanh(gamma) ⊙ y              # gamma in R^H, init 0  => g=0 => exact no-op

Injected via a forward hook on base.model.layers[L] (HF decoder layers return a tuple; the post-hook
rewrites output[0]). The bank ([B,K,mem_dim], query-conditioned, leak-free) is stashed before each
base(...) call and read by the hook.
"""
import torch
import torch.nn as nn


def decoder_layers(base):
    """Locate the ModuleList of decoder layers across HF causal/image-text-to-text wrappers."""
    for path in ("model.layers", "model.model.layers", "language_model.model.layers",
                 "model.language_model.layers", "transformer.h"):
        obj, ok = base, True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and isinstance(obj, nn.ModuleList):
            return obj
    raise RuntimeError("could not locate decoder-layer ModuleList on the base model")


class GatedMemoryTap(nn.Module):
    """Zero-init gated cross-attention from the residual stream into the K-slot memory bank."""

    def __init__(self, base_hidden, mem_dim, n_heads=8):
        super().__init__()
        assert base_hidden % n_heads == 0, "base_hidden must divide n_heads"
        self.H, self.n_heads, self.d_head = base_hidden, n_heads, base_hidden // n_heads
        self.to_q = nn.Linear(base_hidden, base_hidden, bias=False)
        self.to_k = nn.Linear(mem_dim, base_hidden, bias=False)
        self.to_v = nn.Linear(mem_dim, base_hidden, bias=False)
        self.to_o = nn.Linear(base_hidden, base_hidden, bias=False)
        self.gamma = nn.Parameter(torch.zeros(base_hidden))   # gate logit; tanh(0)=0 -> no-op at init
        self._bank = None                                     # [B,K,mem_dim], set per-forward
        self.last_gate = torch.tensor(0.0)
        self.last_attn_entropy = torch.tensor(0.0)

    def set_bank(self, bank):
        self._bank = bank

    def _split(self, t):
        B, T, _ = t.shape
        return t.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)   # [B,nh,T,dh]

    def forward(self, h):
        """h: [B,T,H] residual hidden -> injected [B,T,H]. No-op when no bank is set.

        The tap params live in fp32 (stable gate training); the frozen base hidden is bf16. Compute
        the whole tap in the param dtype, then cast the additive update back to h.dtype so the base
        residual stream stays bf16 (and the gate=0 init is an exact no-op)."""
        bank = self._bank
        if bank is None:
            return h
        wdt = self.to_q.weight.dtype                           # tap compute dtype (fp32)
        h32 = h.to(wdt)
        bank = bank.to(wdt)
        q = self._split(self.to_q(h32))                        # [B,nh,T,dh]
        k = self._split(self.to_k(bank))                       # [B,nh,K,dh]
        v = self._split(self.to_v(bank))                       # [B,nh,K,dh]
        a = torch.softmax(q @ k.transpose(-1, -2) / (self.d_head ** 0.5), dim=-1)   # [B,nh,T,K]
        ctx = (a @ v).transpose(1, 2).reshape(h32.shape)       # [B,T,H]
        y = self.to_o(ctx)
        g = torch.tanh(self.gamma)                             # [H]; 0 at init
        self.last_gate = g.abs().mean().detach()
        self.last_attn_entropy = (-(a.clamp_min(1e-9).log() * a).sum(-1)).mean().detach()
        return h + (g * y).to(h.dtype)


class MAGInjector:
    """Registers GatedMemoryTap forward-hooks on a set of frozen decoder layers, sharing one bank."""

    def __init__(self, base, tap_layers, mem_dim, n_heads=8):
        H = base.config.get_text_config().hidden_size
        self.layers = decoder_layers(base)
        self.tap_layers = list(tap_layers)
        self.taps = nn.ModuleDict({str(L): GatedMemoryTap(H, mem_dim, n_heads) for L in self.tap_layers})
        self._handles = []

    def to(self, dev):
        self.taps.to(dev)
        return self

    def parameters(self):
        return self.taps.parameters()

    def train(self, mode=True):
        self.taps.train(mode)
        return self

    def eval(self):
        self.taps.eval()
        return self

    def set_bank(self, bank):
        for t in self.taps.values():
            t.set_bank(bank)

    def gate_stats(self):
        return {L: float(self.taps[str(L)].last_gate) for L in self.tap_layers}

    def _hook(self, tap):
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return (tap(out[0]),) + tuple(out[1:])
            return tap(out)
        return fn

    def attach(self):
        for L in self.tap_layers:
            self._handles.append(self.layers[L].register_forward_hook(self._hook(self.taps[str(L)])))
        return self

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return self
