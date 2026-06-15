"""Synthetic A/B for the Triton-autotuner parallel-compile patch: a @triton.autotune
kernel with 20 configs, cold-compiled serial vs parallel. Verifies (a) speedup and
(b) SAME best config picked (parallel pre-compile must not change the tuning result)."""
import os
import shutil
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/work")
import triton_autotune_parallel as tap  # noqa: E402

CONFIGS = [triton.Config({"BLOCK_SIZE": bs}, num_warps=w, num_stages=s)
           for bs in (64, 128, 256, 512, 1024) for w in (1, 2) for s in (1, 2)]  # 20


@triton.autotune(configs=CONFIGS, key=["n"])
@triton.jit
def _k(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m)
    for _ in range(6):  # non-trivial body so each compile is a few seconds
        v = tl.sin(v) * 2.0 + tl.cos(v) * 0.5
    tl.store(y_ptr + offs, v, mask=m)


def call(n=1 << 16):
    x = torch.randn(n, device="cuda")
    y = torch.empty_like(x)
    _k[lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )](x, y, n)
    torch.cuda.synchronize()


def cold():
    _k.cache.clear()
    d = os.environ["TRITON_CACHE_DIR"]
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def main():
    torch.empty(8, device="cuda"); torch.cuda.synchronize()
    print(f"{len(CONFIGS)} configs", flush=True)

    cold()
    t = time.perf_counter(); call(); serial = time.perf_counter() - t
    best_serial = str(_k.best_config)
    print(f"SERIAL : {serial:.1f}s  best={best_serial}", flush=True)

    tap.install()
    cold()
    t = time.perf_counter(); call(); par = time.perf_counter() - t
    best_par = str(_k.best_config)
    print(f"PARALLEL: {par:.1f}s  best={best_par}", flush=True)
    print(f"SPEEDUP: {serial/par:.2f}x", flush=True)
    print(f"BEST-CONFIG MATCH: {best_serial == best_par}", flush=True)


if __name__ == "__main__":
    main()
