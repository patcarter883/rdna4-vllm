"""Stage 3 — ISOLATED memory-backend scaling on GPU (no 4B base).

The M2 OOM was the memory's retained autograd graph, not the base. This bench isolates exactly that:
build only the adapter's memory (real shapes: mem_dim=512, heads=4, chunk=16), run N segments of
retrieve->ingest with the FULL cross-segment graph retained (no detach), one backward, and record peak
GPU memory or the OOM point. Sweeps N for both backends.

Expected: lucidrains (torch.func create_graph) OOMs at small N (~the documented ~6); deepmem (graph-free
surprise) scales far higher at flat/low peak memory -> the stage-3 fix, isolated from the base.

Run via gpu-lease (1 card):
  scripts/gpu-lease.sh -n 1 --name titans-memscale -- \
    titans/warmstart/run_m2.sh titans-memscale --entry deep_mem/stage3_mem_scaling.py
(or invoke directly inside the container: python -u /work/deep_mem/stage3_mem_scaling.py)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "warmstart"))
from m2_adapter import TitansMemoryAdapter  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
B, H_BASE, SEG_LEN, K = 1, 2560, 64, 8        # H_BASE mimics the real Qwen3.5-4B embedding width
MEM_DIM, MEM_HEADS, MEM_CHUNK = 512, 4, 16    # real adapter memory config


def trial(memory, n_seg):
    """One backward over n_seg segments, full graph retained. -> peak GB, or 'OOM'."""
    torch.manual_seed(0)
    if DEV == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    adapter = TitansMemoryAdapter(base_hidden=H_BASE, mem_dim=MEM_DIM, n_mem_tokens=K,
                                  mem_chunk=MEM_CHUNK, mem_heads=MEM_HEADS, memory=memory).to(DEV)
    embeds = torch.randn(B, n_seg, SEG_LEN, H_BASE, device=DEV)
    target = torch.randn(B, K, H_BASE, device=DEV)
    try:
        state, loss = None, 0.0
        for t in range(n_seg):
            mt = adapter.read(state, B, torch.float32)           # exercise retrieve graph
            loss = loss + (mt - target).pow(2).mean()
            state = adapter.ingest(embeds[:, t], state)          # full memory graph retained (no detach)
        (loss / n_seg).backward()                                # the memory's create-graph cost lands here
        peak = torch.cuda.max_memory_allocated() / 1e9 if DEV == "cuda" else 0.0
        return f"{peak:.2f} GB"
    except torch.OutOfMemoryError:
        return "OOM"
    finally:
        del adapter
        if DEV == "cuda":
            torch.cuda.empty_cache()


def main():
    print(f"[mem-scale] device={DEV} mem_dim={MEM_DIM} heads={MEM_HEADS} chunk={MEM_CHUNK} seg_len={SEG_LEN}")
    seg_counts = [4, 8, 16, 32, 64]
    results = {}
    for memory in ("lucidrains", "deepmem"):
        row = []
        for n in seg_counts:
            r = trial(memory, n)
            row.append(r)
            print(f"[mem-scale] {memory:11s} n_seg={n:3d} -> {r}")
            if r == "OOM":
                row += ["OOM"] * (len(seg_counts) - len(row))     # once OOM, larger N also OOMs
                break
        results[memory] = row
    print("\n[mem-scale] summary (peak GPU mem, full cross-segment graph retained):")
    print("  n_seg     : " + "  ".join(f"{n:>6d}" for n in seg_counts))
    for memory in ("lucidrains", "deepmem"):
        print(f"  {memory:10s}: " + "  ".join(f"{v:>6s}" for v in results[memory]))
    deep_ok = "OOM" not in results["deepmem"]
    luc_oom = "OOM" in results["lucidrains"]
    print(f"\n[mem-scale] {'PASS' if deep_ok else 'CHECK'}: deepmem retains the full graph to "
          f"{seg_counts[-1]} segments without OOM"
          + (f"; lucidrains OOMs first (the documented memory-graph blowup the kernel fixes)." if luc_oom
             else "; lucidrains did not OOM in this range (raise n_seg to find its wall)."))


if __name__ == "__main__":
    main()
