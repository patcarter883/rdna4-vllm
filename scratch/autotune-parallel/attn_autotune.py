# SPDX-License-Identifier: Apache-2.0
"""Startup autotuner for the unified-attention (TRITON_ATTN) 3D decode path.

This is the ALWAYS-ON mechanism that replaces prebaked per-device JSON configs.
It runs once inside the attention metadata builder's ``__init__`` (before the
segm_* scratch is allocated and before CUDA-graph capture), profiling the exact
deployed shape / dtype / kv-cache-dtype / query_len at each cudagraph capture
size, and returns a per-batch knob table.

Design comes from an offline calibration sweep across 4 shapes x 3 head sizes x
bf16/fp8 x query_len 1/4 (see ShapesToTuneAttentionKernel.md). The takeaways:

  * num_stages           = 1                       (always)
  * tile_size_decode     = 16 bf16 / 32 fp8        (fp8 forces >=32)
  * num_warps            = 1 if head<=128 else 2   (Q-independent)
  * 2D path never wins for decode -> only the 3D path is profiled
  * segments follows occupancy:
        segments ~= clamp(nearest_pow2(C / grid_base), 16, 256), C = CUs * 64
    (within 1 pow2 step ~73% of the time -> sweep +-1 step to absorb residual)
  * waves_per_eu is the only genuinely empirical knob -> sweep {1,2,3,4}

So the search per capture size is {seg/2, seg, seg*2} x waves{1..4} = up to 12
configs; tile/stages/warps are fixed. Profiling compiles exactly the kernel
variants graph capture then reuses, so it doubles as a warm-up.

Returns a plain dict so the backend can wrap it into ResolvedAttnTuning without
a circular import:
    {"buckets": [M,...], "by_bucket": {M: {knobs}}, "max_segments": int}
"""
from __future__ import annotations

import json
import os
import statistics
import tempfile

import torch
import triton
from tqdm import tqdm

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)

# Occupancy constant: segments fill grid_base * segments workgroups to ~CUs*64.
# Empirically ~4096 on the 64-CU R9700; scale by the live CU count when known.
_OCC_BASE = 4096
_DEFAULT_CUS = 64

SEG_MIN, SEG_MAX = 16, 256
WAVES_SWEEP = (1, 2, 3, 4)

# Prefill (2D path) sweep. Prefill is compute-bound long causal GEMMs, a
# different regime from decode: the KV tile (tile_size_2d) is the dominant
# lever, plus pipeline stages and waves. warps is PINNED (head-size rule, same
# as decode) — it came out 1 for head<=128 and isn't worth a sweep dimension.
# segments is irrelevant (2D).
PREFILL_TILE_SWEEP = (32, 64, 128)
PREFILL_STAGES_SWEEP = (1, 2, 4)
PREFILL_WAVES_SWEEP = (1, 2, 4)
# Representative prefill query length if the scheduler chunk is unknown.
DEFAULT_PREFILL_QLEN = 2048
# Profile prefill over a long KV (not just the chunk) so the stages/tile
# pipelining benefit of long-context prefill is exposed — capped well below
# max_model_len to keep startup fast.
PREFILL_KV_LEN = 8192

# Per-process memo: profile each unique shape ONCE. Many attention builders
# (per layer / per KV-cache group) share a shape; without this they each re-time
# the same configs, and margin noise can bake different picks per builder.
_PROFILE_CACHE: dict = {}


def _nearest_pow2(x: float) -> int:
    """Nearest power of two to x (geometric rounding), clamped to [16, 256]."""
    if x <= SEG_MIN:
        return SEG_MIN
    if x >= SEG_MAX:
        return SEG_MAX
    lo = SEG_MIN
    while lo * 2 <= SEG_MAX:
        hi = lo * 2
        if x <= (lo * hi) ** 0.5:  # geometric midpoint
            return lo
        lo = hi
    return SEG_MAX


def _occ_constant() -> int:
    """Occupancy constant C for segments = nearest_pow2(C / grid_base).

    EMPIRICALLY FIT to 4096 on the R9700 (gfx1201) across the offline
    calibration sweep — see ShapesToTuneAttentionKernel.md. We use the fitted
    value directly rather than multi_processor_count*64 because RDNA4 reports an
    ambiguous MP/WGP count (CUs vs WGPs) that mis-centers the prediction and
    makes the +-1 step band miss the true optimum. For a non-R9700 GPU,
    recalibrate this constant. The +-1 step search around it absorbs the rest.
    """
    return _OCC_BASE


def _grid_base(num_seqs, query_len, block_q, num_kv_heads) -> int:
    rows = num_seqs * query_len
    total_q_blocks = rows // block_q + num_seqs
    return total_q_blocks * num_kv_heads


def _seg_candidates(pred: int) -> list[int]:
    """{pred/2, pred, pred*2} clamped to [16,256], deduped, ascending."""
    cands = {
        max(SEG_MIN, pred // 2),
        min(SEG_MAX, max(SEG_MIN, pred)),
        min(SEG_MAX, pred * 2),
    }
    return sorted(cands)


def _build_inputs(L, num_seqs, query_len, num_q_heads, num_kv_heads, head_size,
                  block_size, q_dtype, store_dtype, fp8, device):
    """Synthetic decode batch: num_seqs sequences x query_len query rows each,
    over a KV history of length L. Mirrors the production fp8/bf16 forward."""
    num_blocks = (L + block_size - 1) // block_size
    k = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                    dtype=torch.bfloat16, device=device).to(store_dtype)
    v = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                    dtype=torch.bfloat16, device=device).to(store_dtype)
    rows = num_seqs * query_len
    q = torch.randn(rows, num_q_heads, head_size,
                    dtype=torch.bfloat16, device=device).to(q_dtype)
    out = torch.empty(rows, num_q_heads, head_size,
                      dtype=torch.bfloat16, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device)
    block_table = block_table.unsqueeze(0).expand(num_seqs, num_blocks).contiguous()
    cu_seqlens_q = (torch.arange(num_seqs + 1, dtype=torch.int32, device=device)
                    * query_len)
    seqused_k = torch.full((num_seqs,), L, dtype=torch.int32, device=device)
    if fp8:
        q_descale = torch.ones((), dtype=torch.float32, device=device)
        k_descale = torch.ones((num_seqs, num_kv_heads), dtype=torch.float32,
                               device=device)
        v_descale = k_descale.clone()
        kv_quant_mode = KVQuantMode.FP8_PER_TENSOR
    else:
        q_descale = k_descale = v_descale = None
        kv_quant_mode = KVQuantMode.NONE
    return dict(q=q, k=k, v=v, out=out, block_table=block_table,
                cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, rows=rows,
                max_seqlen_q=query_len, max_seqlen_k=L, q_descale=q_descale,
                k_descale=k_descale, v_descale=v_descale,
                kv_quant_mode=kv_quant_mode)


def _make_segm(segments, rows, num_q_heads, head_size, device):
    hp = triton.next_power_of_2(head_size)
    return (
        torch.empty(rows, num_q_heads, segments, hp, dtype=torch.float32,
                    device=device),
        torch.empty(rows, num_q_heads, segments, dtype=torch.float32,
                    device=device),
        torch.empty(rows, num_q_heads, segments, dtype=torch.float32,
                    device=device),
    )


def _time_3d(inp, scale, segments, tile, warps, stages, waves, runs, device):
    rows = inp["rows"]
    so, sm, se = _make_segm(segments, rows, inp["q"].shape[1],
                            inp["q"].shape[2], device)

    def launch():
        unified_attention(
            q=inp["q"], k=inp["k"], v=inp["v"], out=inp["out"],
            cu_seqlens_q=inp["cu_seqlens_q"], max_seqlen_q=inp["max_seqlen_q"],
            seqused_k=inp["seqused_k"], max_seqlen_k=inp["max_seqlen_k"],
            softmax_scale=scale, causal=True, window_size=(-1, -1),
            block_table=inp["block_table"], softcap=0.0,
            q_descale=inp["q_descale"], k_descale=inp["k_descale"],
            v_descale=inp["v_descale"], kv_quant_mode=inp["kv_quant_mode"],
            # rows+1 keeps the use_3d gate open; query_len counts as decode.
            seq_threshold_3D=rows + 1,
            decode_query_len_threshold=inp["max_seqlen_q"],
            num_par_softmax_segments=segments,
            softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
            tile_size_decode=tile, num_warps=warps, num_stages=stages,
            waves_per_eu=waves,
        )

    launch()  # warmup / JIT compile
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.mean(ts)


def _time_2d(inp, scale, tile_2d, warps, stages, waves, runs):
    """Time the 2D-grid path (prefill / over-capacity). seq_threshold_3D=None
    forces use_3d=False; segments/segm buffers are unused. tile_size_2d
    overrides TILE_SIZE_PREFILL."""
    def launch():
        unified_attention(
            q=inp["q"], k=inp["k"], v=inp["v"], out=inp["out"],
            cu_seqlens_q=inp["cu_seqlens_q"], max_seqlen_q=inp["max_seqlen_q"],
            seqused_k=inp["seqused_k"], max_seqlen_k=inp["max_seqlen_k"],
            softmax_scale=scale, causal=True, window_size=(-1, -1),
            block_table=inp["block_table"], softcap=0.0,
            q_descale=inp["q_descale"], k_descale=inp["k_descale"],
            v_descale=inp["v_descale"], kv_quant_mode=inp["kv_quant_mode"],
            seq_threshold_3D=None, num_par_softmax_segments=None,
            softmax_segm_output=None, softmax_segm_max=None,
            softmax_segm_expsum=None,
            tile_size_2d=tile_2d, num_warps=warps, num_stages=stages,
            waves_per_eu=waves,
        )

    launch()  # warmup / JIT compile
    torch.cuda.synchronize()
    ts = []
    for _ in range(runs):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.mean(ts)


def profile_prefill(*, num_kv_heads, num_q_heads, head_size, store_dtype,
                    q_dtype, fp8, warps, prefill_qlen, max_model_len,
                    block_size, scale, device, runs, disable_progress=False):
    """Profile the 2D prefill path: one config (grid-saturated). warps is PINNED
    (head-size rule); sweeps tile_2d x stages x waves over a long-context causal
    prefill. Returns {tile_size_2d, num_warps, num_stages, waves_per_eu} or
    None."""
    qlen = max(1, min(int(prefill_qlen), int(max_model_len)))
    # KV length: a long context (>= PREFILL_KV_LEN) so the stages/tile
    # pipelining benefit of long-context prefill shows up — capped at
    # max_model_len, and below it to keep the profile fast. Per-tile efficiency
    # at this KV transfers to even longer contexts (more identical iterations).
    kv_len = max(qlen, min(int(max_model_len), PREFILL_KV_LEN))
    inp = _build_inputs(kv_len, 1, qlen, num_q_heads, num_kv_heads, head_size,
                        block_size, q_dtype, store_dtype, fp8, device)
    grid = [(t, st, wv)
            for t in PREFILL_TILE_SWEEP
            for st in PREFILL_STAGES_SWEEP
            for wv in PREFILL_WAVES_SWEEP]
    best = None
    bar = tqdm(grid, desc="Autotuning unified-attention (prefill, 2D)",
               disable=disable_progress, unit="cfg")
    for (tile_2d, stages, waves) in bar:
        try:
            ms = _time_2d(inp, scale, tile_2d, warps, stages, waves, runs)
        except Exception as e:  # compile/launch failure or OOM
            logger.debug("prefill autotune tile=%d stages=%d waves=%d "
                         "failed: %r", tile_2d, stages, waves, e)
            continue
        if best is None or ms < best[0]:
            best = (ms, tile_2d, stages, waves)
    if best is None:
        return None
    _, btile, bstages, bwaves = best
    logger.info_once(
        "prefill autotune (qlen=%d, kv=%d): tile_2d=%d warps=%d (pinned) "
        "stages=%d waves=%d -> %.4f ms", qlen, kv_len, btile, warps, bstages,
        bwaves, best[0], scope="global",
    )
    return {
        "tile_size_2d": btile,
        "num_warps": warps,
        "num_stages": bstages,
        "waves_per_eu": bwaves,
    }


def _dump_result(result, diag, *, num_kv_heads, num_q_heads, head_size,
                 dtype, kv_cache_dtype, query_len, prefill_query_len):
    """Dump the full per-shape profiling result to JSON for correctness review.
    Location: $VLLM_ATTN_AUTOTUNE_DUMP_DIR, else <tmp>/vllm_attn_autotune/.
    One file per shape; includes per-bucket grid_base / pred_seg / ms diags."""
    dump_dir = os.environ.get("VLLM_ATTN_AUTOTUNE_DUMP_DIR") or \
        os.path.join(tempfile.gettempdir(), "vllm_attn_autotune")
    try:
        os.makedirs(dump_dir, exist_ok=True)
        try:
            device_name = current_platform.get_device_name().replace(" ", "_")
        except Exception:
            device_name = "unknown"
        dt = str(dtype).replace("torch.", "")
        kvt = "fp8" if (isinstance(kv_cache_dtype, str)
                        and kv_cache_dtype.startswith("fp8")) else "bf16"
        payload = {
            "device_name": device_name,
            "num_kv_heads": num_kv_heads, "num_q_heads": num_q_heads,
            "head_size": head_size, "dtype": dt, "kvdtype": kvt,
            "query_len": query_len, "prefill_query_len": prefill_query_len,
            "max_segments": result["max_segments"],
            "decode": {str(M): {**result["by_bucket"][M], **diag.get(M, {})}
                       for M in result["buckets"]},
            "prefill": result.get("prefill"),
        }
        fname = (f"autotune_{device_name}_kv{num_kv_heads}_q{num_q_heads}"
                 f"_h{head_size}_{dt}_kv{kvt}_ql{query_len}.json")
        path = os.path.join(dump_dir, fname)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info_once("attn autotune results dumped to %s", path,
                         scope="global")
    except Exception as e:  # dumping must never break startup
        logger.debug("attn autotune result dump failed: %r", e)


def profile_attn_tuning(*, num_kv_heads, num_q_heads, head_size, dtype,
                        kv_cache_dtype, query_len, capture_sizes, max_model_len,
                        block_size, device, runs=5, disable_progress=False,
                        prefill_query_len=DEFAULT_PREFILL_QLEN):
    """Profile the 3D decode path at each capture size AND the 2D prefill path.
    Returns {"buckets", "by_bucket", "max_segments", "prefill"} or None on total
    failure."""
    norm_sizes = tuple(sorted({int(s) for s in capture_sizes if int(s) >= 1}))
    cache_key = (num_kv_heads, num_q_heads, head_size, str(dtype),
                 str(kv_cache_dtype), int(query_len), norm_sizes,
                 int(block_size), int(max_model_len), int(prefill_query_len),
                 str(device))
    cached = _PROFILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    fp8 = isinstance(kv_cache_dtype, str) and kv_cache_dtype.startswith("fp8")
    store_dtype = current_platform.fp8_dtype() if fp8 else dtype
    q_dtype = store_dtype
    tile = 32 if fp8 else 16
    warps = 1 if head_size <= 128 else 2
    stages = 1
    scale = head_size ** -0.5
    L = int(max_model_len)

    nqpkv = max(1, num_q_heads // num_kv_heads)
    block_m = 16 if nqpkv <= 16 else triton.next_power_of_2(nqpkv)
    block_q = max(1, block_m // nqpkv)
    C = _occ_constant()

    # Always tune M=1. Single-sequence decode runs at num_seqs=1, and the
    # spec-decode draft path runs M=1 even when the verify batch is num_seqs>1;
    # but the spec-decode capture remap can make the smallest captured bucket >1
    # (e.g. 3 at num_spec=2). Without an explicit M=1 bucket, a live num_seqs=1
    # request rounds UP to the smallest bucket and inherits a throughput-tuned
    # config instead of the latency one.
    buckets = sorted({1} | {int(s) for s in capture_sizes if int(s) >= 1})

    by_bucket: dict[int, dict] = {}
    diag: dict[int, dict] = {}
    max_segments = SEG_MIN
    bar = tqdm(buckets, desc="Autotuning unified-attention (decode, 3D)",
               disable=disable_progress, unit="bucket")
    for M in bar:
        gb = _grid_base(M, query_len, block_q, num_kv_heads)
        pred = _nearest_pow2(C / gb)
        seg_cands = _seg_candidates(pred)
        inp = _build_inputs(L, M, query_len, num_q_heads, num_kv_heads,
                            head_size, block_size, q_dtype, store_dtype, fp8,
                            device)
        best = None
        for seg in seg_cands:
            for waves in WAVES_SWEEP:
                try:
                    ms = _time_3d(inp, scale, seg, tile, warps, stages, waves,
                                  runs, device)
                except Exception as e:  # compile/launch failure for this combo
                    logger.debug("attn autotune M=%d seg=%d waves=%d failed: %r",
                                 M, seg, waves, e)
                    continue
                if best is None or ms < best[0]:
                    best = (ms, seg, waves)
        if best is None:
            # Fall back to the formula prediction, default waves.
            best = (float("nan"), pred, None)
        _, bseg, bwaves = best
        by_bucket[M] = {
            "num_par_softmax_segments": bseg,
            "tile_size_decode": tile,
            "num_warps": warps,
            "num_stages": stages,
            "waves_per_eu": bwaves,
        }
        max_segments = max(max_segments, bseg)
        diag[M] = {"grid_base": gb, "pred_seg": pred,
                   "avg_ms": round(best[0], 5)}
        logger.info_once(
            "attn autotune: M=%d grid_base=%d pred_seg=%d -> seg=%d waves=%s "
            "(tile=%d warps=%d) %.4f ms",
            M, gb, pred, bseg, bwaves, tile, warps, best[0], scope="global",
        )

    if not by_bucket:
        return None

    # 2D prefill path (one config; grid-saturated). None if it fails entirely.
    prefill = profile_prefill(
        num_kv_heads=num_kv_heads, num_q_heads=num_q_heads,
        head_size=head_size, store_dtype=store_dtype, q_dtype=q_dtype, fp8=fp8,
        warps=warps, prefill_qlen=prefill_query_len, max_model_len=max_model_len,
        block_size=block_size, scale=scale, device=device, runs=runs,
        disable_progress=disable_progress,
    )

    result = {
        "buckets": sorted(by_bucket),
        "by_bucket": by_bucket,
        "max_segments": int(max_segments),
        "prefill": prefill,
    }
    _PROFILE_CACHE[cache_key] = result
    _dump_result(result, diag, num_kv_heads=num_kv_heads,
                 num_q_heads=num_q_heads, head_size=head_size, dtype=dtype,
                 kv_cache_dtype=kv_cache_dtype, query_len=query_len,
                 prefill_query_len=prefill_query_len)
    return result
