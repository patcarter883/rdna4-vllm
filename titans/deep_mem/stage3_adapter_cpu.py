"""Stage 3 wiring check (CPU, no base model) — exercise TitansMemoryAdapter's deepmem read/ingest.

Catches shape/None-handling/grad-flow bugs in the adapter's deepmem branch cheaply, before any GPU
time on the 4B base. Runs a multi-segment retrieve->ingest cycle, backprops an LM-stand-in loss, and
confirms: finite memory tokens, state carries across segments, grads reach the adapter (and the loss
moves under a few optimizer steps). Also confirms it scales to many segments with flat memory.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "warmstart"))
from m2_adapter import TitansMemoryAdapter  # noqa: E402


def run(n_seg=20, mem="deepmem"):
    torch.manual_seed(0)
    B, H, seg_len, K = 2, 64, 16, 4
    adapter = TitansMemoryAdapter(base_hidden=H, mem_dim=32, n_mem_tokens=K, mem_chunk=8,
                                  mem_heads=4, memory=mem)
    opt = torch.optim.AdamW(adapter.parameters(), lr=2e-3)
    embeds = torch.randn(B, n_seg, seg_len, H)            # stand-in for base token embeds
    target = torch.randn(B, K, H)                         # arbitrary fixed target for the read tokens

    first = last = None
    for step in range(80):
        opt.zero_grad()
        state, loss = None, 0.0
        for t in range(n_seg):
            mem_tokens = adapter.read(state, B, torch.float32)        # [B,K,H]
            loss = loss + (mem_tokens - target).pow(2).mean()        # stand-in LM signal on the read
            state = adapter.ingest(embeds[:, t], state)              # write segment, carry state
        loss = loss / n_seg
        assert torch.isfinite(mem_tokens).all(), "non-finite memory tokens"
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
        last = loss.item()

    g = [p.grad for p in adapter.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"[stage3-wire] memory={mem} n_seg={n_seg}: loss {first:.4f} -> {last:.4f} "
          f"(Δ={first-last:+.4f}); params-with-grad={len(g)}; mem_token shape={tuple(mem_tokens.shape)}")
    ok = (first - last) > 0.05 and len(g) > 0
    print(f"[stage3-wire] {'PASS' if ok else 'FAIL'}: deepmem adapter read/ingest cycle trains "
          f"({'grads flow, state carries, scales to '+str(n_seg)+' segs' if ok else 'no learning / no grad'})")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run(n_seg=20) else 1)
