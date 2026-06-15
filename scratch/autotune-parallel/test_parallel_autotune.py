"""GPU A/B for the parallel attention-autotune pre-warm.

Measures, on a COLD Triton cache each time:
  (1) SERIAL   = original profile_attn_tuning (compiles each config one-at-a-time)
  (2) PARALLEL = prewarm() [process pool] + original profile_attn_tuning (cache-hits)

If (2) << (1), the workers' tiny-input compiles are cache-hitting the serial loop
(the core assumption) and the win is real. Run inside :m3 on a free GPU (cuda:0)."""
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, "/work")
import attn_autotune_parallel as aap  # noqa: E402
from vllm.v1.attention.ops import attn_autotune as A  # noqa: E402

SHAPE = dict(
    num_kv_heads=8, num_q_heads=32, head_size=128, dtype=torch.bfloat16,
    kv_cache_dtype="auto", query_len=1,
    capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    max_model_len=4096, block_size=16, device="cuda:0", runs=3,
    disable_progress=True,
)
PW = dict(
    num_kv_heads=8, num_q_heads=32, head_size=128, dtype=torch.bfloat16,
    kv_cache_dtype="auto", capture_sizes=SHAPE["capture_sizes"],
    max_model_len=4096, block_size=16, query_len=1, prefill_query_len=2048,
    device="cuda:0",
)


def clear_cache():
    A._PROFILE_CACHE.clear()
    d = os.environ["TRITON_CACHE_DIR"]
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def main():
    # warm the GPU/HIP context + import paths once (not timed)
    torch.empty(8, device="cuda:0")
    torch.cuda.synchronize()

    clear_cache()
    t = time.perf_counter()
    A.profile_attn_tuning(**SHAPE)
    serial = time.perf_counter() - t
    print(f"SERIAL autotune (cold): {serial:.1f}s", flush=True)

    clear_cache()
    t = time.perf_counter()
    aap.prewarm(**PW)
    pw = time.perf_counter() - t
    A.profile_attn_tuning(**SHAPE)
    total = time.perf_counter() - t
    print(f"PARALLEL prewarm: {pw:.1f}s | prewarm+serial total: {total:.1f}s", flush=True)
    print(f"SPEEDUP: {serial/total:.2f}x  (serial {serial:.1f}s -> {total:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
