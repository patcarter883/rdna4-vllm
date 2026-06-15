"""Tune the parallel pre-warm: find where the ~14s goes (worker import floor vs
compile) and sweep worker counts. Run inside :m3 on a free GPU (cuda:0)."""
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, "/work")
import attn_autotune_parallel as aap  # noqa: E402
from vllm.v1.attention.ops import attn_autotune as A  # noqa: E402

PW = dict(
    num_kv_heads=8, num_q_heads=32, head_size=128, dtype=torch.bfloat16,
    kv_cache_dtype="auto", capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    max_model_len=4096, block_size=16, query_len=1, prefill_query_len=2048,
    device="cuda:0",
)


def clear_cache():
    A._PROFILE_CACHE.clear()
    d = os.environ["TRITON_CACHE_DIR"]
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)


def _import_only(_):
    t = time.perf_counter()
    import torch  # noqa
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa
    return time.perf_counter() - t


def measure_import_floor():
    import concurrent.futures as cf
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        inner = list(ex.map(_import_only, [0]))[0]
    wall = time.perf_counter() - t
    print(f"IMPORT FLOOR: spawn+import 1 worker = {wall:.1f}s wall "
          f"({inner:.1f}s inside the worker)", flush=True)


def main():
    torch.empty(8, device="cuda:0"); torch.cuda.synchronize()  # parent inits CUDA
    n_tasks = len(aap._enumerate_tasks(**PW))
    print(f"tasks to compile: {n_tasks}", flush=True)
    measure_import_floor()
    for w in (6, 8, 12, 16):
        os.environ["VLLM_ATTN_AUTOTUNE_PARALLEL_WORKERS"] = str(w)
        clear_cache()
        t = time.perf_counter()
        aap.prewarm(**PW)
        dt = time.perf_counter() - t
        print(f"WORKERS={w:2d}: prewarm {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
