# SPDX-License-Identifier: Apache-2.0
"""Parallel pre-warm for vLLM's unified-attention startup autotuner.

WHY: profile_attn_tuning (attn_autotune.py) compiles each candidate Triton config
SERIALLY — for each config the first launch() does a Triton->LLVM->ISA compile
(~seconds, single CPU core) then times it on the GPU (~microseconds). On a cold
Triton cache (every container without a warm cache, every kernel/.so rebuild) this
is ~20 min at GPU 3% / 1-of-16 cores — a recurring DEV pain.

WHAT: a process-pool pre-warm. The compiled kernel is keyed by (knobs, dtypes), NOT
by input sizes, so we compile every unique knob-combo in parallel worker processes
using TINY inputs. Each worker's compile writes to the shared /root/.triton cache;
the subsequent serial profile_attn_tuning then CACHE-HITS every compile and only does
the (already-serial, but now fast) GPU timing. Net: the ~45 cold compiles fan across
N cores instead of running one-at-a-time.

Correctness/safety:
  * Compiles are identical to what the serial loop would produce (same kernel, same
    knobs/dtypes) — we change WHEN/where the compile happens, never the result. The
    timing + final pick are still done serially by the original code.
  * Workers use spawn (fresh HIP context each); pool is reused so context init (~1-2s)
    is amortized across the configs each worker handles.
  * Tiny inputs (1 KV block, num_seqs=1, qlen=1) keep per-worker GPU work + VRAM
    minimal; the GPU kernel launches serialize on the one GPU but the LLVM compiles
    (in separate processes) overlap — that's the whole win.
  * Best-effort: any worker failure is swallowed (the serial loop will just compile
    that config itself, as today). Pre-warm can never change the tuning outcome.

Gate: only active when VLLM_ATTN_AUTOTUNE_PARALLEL=1 (default off until GPU-validated).
Workers: min(VLLM_ATTN_AUTOTUNE_PARALLEL_WORKERS or cpu-2, 8).
"""
from __future__ import annotations

import os


# ---- worker (runs in a spawned subprocess; must be top-level + self-contained) ----
def _compile_worker(task: dict) -> tuple[str, bool, str]:
    """Compile ONE attention config by running unified_attention once on tiny inputs.
    Returns (label, ok, err). Never raises (best-effort cache warming)."""
    label = task["label"]
    try:
        import torch
        import triton
        from vllm.v1.attention.ops.triton_unified_attention import unified_attention
        from vllm.v1.kv_cache_interface import KVQuantMode

        dev = task["device"]
        torch.cuda.set_device(dev)
        hs = task["head_size"]
        nqh, nkvh = task["num_q_heads"], task["num_kv_heads"]
        q_dtype = getattr(torch, task["q_dtype"])
        store_dtype = getattr(torch, task["store_dtype"])
        fp8 = task["fp8"]
        bs = task["block_size"]

        # TINY inputs: 1 KV block, 1 seq, qlen=1 (decode) or small qlen (prefill).
        # Same dtypes/knobs => same compiled kernel as the real (large) shapes.
        qlen = task.get("qlen", 1)
        L = bs  # one block of KV history is enough to compile
        num_blocks = 1
        k = torch.randn(num_blocks, bs, nkvh, hs, dtype=torch.bfloat16,
                        device=dev).to(store_dtype)
        v = torch.randn(num_blocks, bs, nkvh, hs, dtype=torch.bfloat16,
                        device=dev).to(store_dtype)
        rows = qlen
        q = torch.randn(rows, nqh, hs, dtype=torch.bfloat16, device=dev).to(q_dtype)
        out = torch.empty(rows, nqh, hs, dtype=torch.bfloat16, device=dev)
        block_table = torch.zeros(1, num_blocks, dtype=torch.int32, device=dev)
        cu_seqlens_q = torch.tensor([0, qlen], dtype=torch.int32, device=dev)
        seqused_k = torch.full((1,), L, dtype=torch.int32, device=dev)
        if fp8:
            q_descale = torch.ones((), dtype=torch.float32, device=dev)
            k_descale = torch.ones((1, nkvh), dtype=torch.float32, device=dev)
            v_descale = k_descale.clone()
            kvqm = KVQuantMode.FP8_PER_TENSOR
        else:
            q_descale = k_descale = v_descale = None
            kvqm = KVQuantMode.NONE
        scale = hs ** -0.5

        common = dict(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q, max_seqlen_q=qlen,
            seqused_k=seqused_k, max_seqlen_k=L, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=block_table, softcap=0.0,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            kv_quant_mode=kvqm, num_warps=task["warps"], num_stages=task["stages"],
            waves_per_eu=task["waves"],
        )
        if task["kind"] == "decode":
            seg = task["seg"]
            hp = triton.next_power_of_2(hs)
            so = torch.empty(rows, nqh, seg, hp, dtype=torch.float32, device=dev)
            sm = torch.empty(rows, nqh, seg, dtype=torch.float32, device=dev)
            se = torch.empty(rows, nqh, seg, dtype=torch.float32, device=dev)
            unified_attention(
                seq_threshold_3D=rows + 1, decode_query_len_threshold=qlen,
                num_par_softmax_segments=seg, softmax_segm_output=so,
                softmax_segm_max=sm, softmax_segm_expsum=se,
                tile_size_decode=task["tile"], **common)
        else:  # prefill / 2D
            unified_attention(
                seq_threshold_3D=None, num_par_softmax_segments=None,
                softmax_segm_output=None, softmax_segm_max=None,
                softmax_segm_expsum=None, tile_size_2d=task["tile"], **common)
        torch.cuda.synchronize()
        return (label, True, "")
    except Exception as e:  # best-effort: serial loop will compile it instead
        return (label, False, repr(e)[:200])


def _enumerate_tasks(*, num_kv_heads, num_q_heads, head_size, dtype, kv_cache_dtype,
                     capture_sizes, max_model_len, block_size, query_len,
                     prefill_query_len, device):
    """Build the unique compile-config task list (decode seg×waves + prefill
    tile×stages×waves), mirroring attn_autotune's knob rules and seg candidates."""
    import triton

    from vllm.platforms import current_platform
    from vllm.v1.attention.ops import attn_autotune as A

    fp8 = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")
    store_dtype = current_platform.fp8_dtype() if fp8 else dtype
    q_dtype = store_dtype
    tile = 32 if fp8 else 16
    warps = 1 if head_size <= 128 else 2
    stages = 1

    nqpkv = max(1, num_q_heads // num_kv_heads)
    block_m = 16 if nqpkv <= 16 else triton.next_power_of_2(nqpkv)
    block_q = max(1, block_m // nqpkv)
    C = A._occ_constant()

    buckets = sorted({1} | {int(s) for s in capture_sizes if int(s) >= 1})
    segs = set()
    for M in buckets:
        gb = A._grid_base(M, query_len, block_q, num_kv_heads)
        segs.update(A._seg_candidates(A._nearest_pow2(C / gb)))

    def dt_name(t):
        return str(t).replace("torch.", "")

    shape = dict(num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
                 head_size=head_size, q_dtype=dt_name(q_dtype),
                 store_dtype=dt_name(store_dtype), fp8=fp8,
                 block_size=block_size, device=str(device))

    tasks = []
    for seg in sorted(segs):
        for waves in A.WAVES_SWEEP:
            tasks.append({**shape, "kind": "decode", "seg": int(seg), "tile": tile,
                          "warps": warps, "stages": stages, "waves": waves,
                          "qlen": int(query_len),
                          "label": f"decode seg={seg} waves={waves}"})
    pq = max(1, min(int(prefill_query_len), int(max_model_len)))
    for t in A.PREFILL_TILE_SWEEP:
        for st in A.PREFILL_STAGES_SWEEP:
            for wv in A.PREFILL_WAVES_SWEEP:
                tasks.append({**shape, "kind": "prefill", "seg": 0, "tile": t,
                              "warps": warps, "stages": st, "waves": wv,
                              "qlen": min(pq, 8),
                              "label": f"prefill tile={t} stages={st} waves={wv}"})
    return tasks


def prewarm(**kwargs) -> None:
    """Parallel pre-warm of all unique attention compile-configs (best-effort)."""
    import concurrent.futures as cf
    import multiprocessing as mp

    from vllm.logger import init_logger
    logger = init_logger(__name__)

    tasks = _enumerate_tasks(**kwargs)
    n = len(tasks)
    workers = int(os.environ.get("VLLM_ATTN_AUTOTUNE_PARALLEL_WORKERS",
                                 min(8, max(1, (os.cpu_count() or 2) - 2))))
    logger.info_once("attn autotune parallel pre-warm: %d configs across %d workers",
                     n, workers, scope="global")
    ctx = mp.get_context("spawn")
    ok = 0
    try:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for label, good, err in ex.map(_compile_worker, tasks):
                ok += good
                if not good:
                    logger.debug("prewarm %s failed: %s", label, err)
    except Exception as e:  # never break startup; serial loop compiles as fallback
        logger.debug("attn autotune parallel pre-warm aborted: %r", e)
    logger.info_once("attn autotune parallel pre-warm: %d/%d configs cached",
                     ok, n, scope="global")


def install() -> None:
    """Monkeypatch profile_attn_tuning to run the parallel pre-warm first, then the
    original serial profile (which now cache-hits every compile). Idempotent;
    gated by VLLM_ATTN_AUTOTUNE_PARALLEL=1."""
    if os.environ.get("VLLM_ATTN_AUTOTUNE_PARALLEL", "0") != "1":
        return
    from vllm.v1.attention.ops import attn_autotune as A
    if getattr(A, "_parallel_prewarm_installed", False):
        return
    _orig = A.profile_attn_tuning

    def _wrapped(*, num_kv_heads, num_q_heads, head_size, dtype, kv_cache_dtype,
                 query_len, capture_sizes, max_model_len, block_size, device,
                 runs=5, disable_progress=False,
                 prefill_query_len=A.DEFAULT_PREFILL_QLEN):
        try:
            prewarm(num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
                    head_size=head_size, dtype=dtype, kv_cache_dtype=kv_cache_dtype,
                    capture_sizes=capture_sizes, max_model_len=max_model_len,
                    block_size=block_size, query_len=query_len,
                    prefill_query_len=prefill_query_len, device=device)
        except Exception:
            pass  # fall through to the unchanged serial profile
        return _orig(num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
                     head_size=head_size, dtype=dtype, kv_cache_dtype=kv_cache_dtype,
                     query_len=query_len, capture_sizes=capture_sizes,
                     max_model_len=max_model_len, block_size=block_size,
                     device=device, runs=runs, disable_progress=disable_progress,
                     prefill_query_len=prefill_query_len)

    A.profile_attn_tuning = _wrapped
    A._parallel_prewarm_installed = True
