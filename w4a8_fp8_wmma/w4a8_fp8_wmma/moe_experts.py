"""vLLM modular-MoE experts backend for the W4A8-FP8 WMMA grouped HIP op (gfx1201).

This is the MoE analogue of ``vllm_adapter.py`` (which hooks the *dense*
``_POSSIBLE_KERNELS[ROCM]`` path). An AWQ-4bit MoE model on ROCm routes its
expert layers through ``AWQMarlinMoEMethod``, which builds a *modular kernel* with
a ``FusedMoEExperts`` class chosen by an oracle (``select_wna16_moe_backend``).
``register_moe()`` installs a **surgical** hook on gfx12x: it wraps
``AWQMarlinMoEMethod.__init__`` to swap in our ``W4A8Fp8WmmaExperts`` (only for
supported AWQ-asym-4bit configs) and patches ``convert_to_wna16_moe_kernel_format``
/ ``make_wna16_moe_kernel`` in the ``awq_marlin`` namespace only. It deliberately
does NOT touch the shared ``int_wna16`` oracle, because GPTQ and compressed-tensors
MoE go through the same oracle and must stay on Marlin. Unsupported AWQ configs
(symmetric, odd group size, ...) keep the stock Marlin selection.

What the experts class does (``W4A8Fp8WmmaExperts.apply``): the whole gated MoE,
composed from our one validated grouped GEMM op
(``torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm``):

    moe_align_block_size(topk_ids, block_m, E)
      -> gemm1 (w13)  : grouped op, gathers x by sorted_token_ids//top_k
      -> silu_and_mul : (P, 2*inter) -> (P, inter)   [vLLM activation op]
      -> gemm2 (w2)   : grouped op, identity-gather (top_k=1, sorted_ids=arange)
      -> scatter-reduce: fp32 index_add over tokens, weighted by topk_weights

The grouped op outputs in the *padded-sorted* layout (one row per padded
(token, expert-slot)); we keep the entire chain in that layout and only fold
back to (M, K) at the final scatter-reduce, which avoids depending on Marlin's
internal gather/scatter conventions.

Weight layout: AWQ MoE weights arrive packed-along-output with AWQ bit order.
``_awq_moe_to_op_layout`` converts each of w13/w2 to our op's layout
``w_packed (E, N, K//8)`` / ``scales (E, N, K//group)`` fp16 /
``zeros (E, N//8, K//group)`` int32 — the same convention the dense AWQ path was
validated against (``_convert_awq_to_standard_format`` + the AutoGPTQ repack).
"""
import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# AWQ packs the 8 nibbles of each int32 in this logical order (matches
# vllm awq_marlin._REVERSE_AWQ_PACK_ORDER).
_REVERSE_AWQ_PACK_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


# --------------------------------------------------------------------------- #
# env / platform helpers
# --------------------------------------------------------------------------- #
def _on_gfx12x() -> bool:
    try:
        from vllm.platforms.rocm import on_gfx12x
        return on_gfx12x()
    except Exception:
        return False


def _moe_enabled() -> bool:
    """MoE path on iff the master flag and the MoE-specific flag are both set."""
    if os.environ.get("VLLM_ROCM_USE_W4A8_FP8_WMMA", "1") != "1":
        return False
    return os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE", "1") == "1"


# The env-selectable grouped-kernel names (the public surface above the torch
# ABI; __init__._MOE_KERNELS maps each to its opaque op id). "wmma" is the tiled
# WMMA default; the former v5-vs-v6 A-residence split is now the env knob
# VLLM_W4A8_MOE_A_IN_LDS (consumed in C++ make_moe_tile_config), NOT a name.
_VALID_MOE_KERNELS = ("scalar", "wmma", "gemv")


def _raise_if_removed_moe_env() -> None:
    """Hard-break for the removed numeric MOE_VERSION override. Called BOTH from
    _moe_kernel() AND once at hook-install time (register_moe*) — the latter is the
    load-bearing chokepoint: _moe_kernel() is also reached from the first-batch
    autotune A/B, whose try/except would SWALLOW a raise there (caching None ->
    silent stock fallthrough, exactly the silent mis-dispatch the hard-break
    forbids). Raising at register time fires loudly at server boot instead."""
    if "VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION" in os.environ:
        raise RuntimeError(
            "VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION is removed; set "
            "VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL=wmma|scalar|gemv instead "
            "(the old 5-vs-6 A-residence split is now VLLM_W4A8_MOE_A_IN_LDS).")


def _moe_kernel() -> str:
    """Grouped-kernel NAME (default "wmma" = tiled fp8 WMMA, the served default;
    "scalar" = golden reference for debug; "gemv" = decode GEMV). The decode path
    still auto-selects gemv per-batch (M<=GEMV_MAX_M) regardless; this only names
    the WMMA-vs-scalar default."""
    _raise_if_removed_moe_env()
    kernel = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL", "wmma")
    if kernel not in _VALID_MOE_KERNELS:
        raise RuntimeError(
            f"VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL={kernel!r} invalid; "
            f"use one of {list(_VALID_MOE_KERNELS)}")
    return kernel


def _force_mode() -> str:
    """Override the per-shape tuning gate. VLLM_ROCM_W4A8_FORCE:
    'auto' (default) crossover cache; 'on'/'1' always our grouped op; 'off'/'0'
    always stock Triton MoE. Same env/semantics as the dense path's gate."""
    v = os.environ.get("VLLM_ROCM_W4A8_FORCE", "auto").strip().lower()
    if v in ("on", "1", "true"):
        return "on"
    if v in ("off", "0", "false"):
        return "off"
    return "auto"


def _moe_min_m() -> int:
    """M-adaptive dispatch threshold: route batches with M < this to the stock
    Triton MoE (tiny-decode regime), and use our FP8-WMMA grouped op for M >=
    this. Mirrors the dense path's crossover so the MoE pathway is always >=
    stock. Tunable; 0 forces our op for all M. Default 64: micro-bench (block_m=64
    + dynamic LDS) shows we beat stock moe_wna16 at M>=64 (~0.84-0.98x) and only
    lose at very small decode batches (M<=32)."""
    return int(os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M", "64"))


# --------------------------------------------------------------------------- #
# M-adaptive dispatch via an AOT crossover cache (offline-tuned, O(1) lookup).
# moe_crossover_cache.json: "E,hidden,inter,group,top_k" -> [lo, hi] (the M-window
# where our op beats stock moe_wna16) or null (never). Unknown shape -> stock, so
# the MoE pathway is ALWAYS >= stock (no regression). Tuned by tune_moe_crossover.py.
# --------------------------------------------------------------------------- #
_MOE_XOVER = None


def _moe_cache_path():
    return os.environ.get(
        "VLLM_ROCM_W4A8_FP8_WMMA_MOE_CACHE",
        os.path.join(os.path.dirname(__file__), "moe_crossover_cache.json"))


def _moe_load_cache():
    global _MOE_XOVER
    if _MOE_XOVER is None:
        try:
            with open(_moe_cache_path()) as f:
                _MOE_XOVER = json.load(f)
        except (OSError, ValueError):
            _MOE_XOVER = {}
    return _MOE_XOVER


def _moe_crossover_for(E, hidden, inter, group, top_k):
    return _moe_load_cache().get(f"{E},{hidden},{inter},{group},{top_k}")


def _moe_autotune_enabled() -> bool:
    """On a MoE crossover-cache MISS, run a quick A/B microbench (our grouped op
    vs stock Triton moe_wna16) for that exact (sharded) expert shape, then persist
    the winning M-window(s) (subsequent loads are O(1)). VLLM_ROCM_W4A8_AUTOTUNE:
    'on' (default) / 'off'. Off -> unknown shapes stay stock, the prior behaviour.
    Shares the dense path's env so one switch governs both gates."""
    v = os.environ.get("VLLM_ROCM_W4A8_AUTOTUNE", "on").strip().lower()
    return v not in ("0", "off", "false", "no")


# M sweep + win margin for the MoE autotune A/B (matches tune_moe_crossover.py so
# the value the autotuner writes is what the AOT tuner would write).
_MOE_AUTOTUNE_MGRID = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
_MOE_AUTOTUNE_MARGIN = 0.02  # require ours/stock < 1-margin to count as a win


def _moe_winning_intervals(wins):
    """Pure selection (no GPU): given `wins` = list of (M, won) over the sweep
    grid, return the contiguous winning [lo, hi] intervals (or None if no win).

    The win region can be non-contiguous (a mid-M dip), so a single [min,max]
    would wrongly engage the loss zone; each proven-win run is stored separately.
    Mirrors tune_moe_crossover.tune_shape()'s interval extraction exactly."""
    intervals, run = [], None
    for m, w in wins:
        if w:
            run = [m, m] if run is None else [run[0], m]
        elif run is not None:
            intervals.append(run)
            run = None
    if run is not None:
        intervals.append(run)
    return intervals or None


def _moe_persist_crossover(E, hidden, inter, group, top_k, value):
    """Merge {key: value} into the MoE crossover cache JSON + in-process table.
    value is the [[lo,hi],...] window list or None (never engage). Best-effort:
    a write failure must not break load (we still use the value in-memory)."""
    key = f"{E},{hidden},{inter},{group},{top_k}"
    table = _moe_load_cache()
    table[key] = value
    try:
        with open(_moe_cache_path(), "w") as f:
            json.dump(table, f, indent=1, sort_keys=True)
    except OSError:
        pass


def _moe_autotune_crossover(layer, E, hidden, inter, group, top_k):
    """First-batch A/B microbench (GPU): time our grouped FP8-WMMA MoE op vs stock
    Triton moe_wna16 (`fused_experts`) across _MOE_AUTOTUNE_MGRID for THIS exact
    (sharded) expert shape, find the contiguous winning M-window(s), persist them,
    and return the window list. ROBUSTNESS: ANY failure raises (the caller's
    try/except caches None -> stock) so the served pathway never regresses.

    Reuses the weights already on `layer`: our op-layout copies (_w4a8_w13/..) for
    the OURS side and the WNA16 hook's still-registered original uint8 standard-
    format weights (w13_qweight/..) for the STOCK side -- exactly the two layouts
    tune_moe_crossover.py builds. Mirrors that tuner's measurement (grouped op vs
    fused_experts, 1-margin win rule, contiguous intervals), but at first-batch
    time for the actual served shape (top_k only materialises per-batch, so this is
    the natural seam; runs once per shape, then the cache makes it O(1))."""
    import time

    from vllm.model_executor.layers.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        int4_w4a16_moe_quant_config,
    )

    w13_op, w2_op = layer._w4a8_w13, layer._w4a8_w2
    w13_s, w2_s = layer._w4a8_w13s, layer._w4a8_w2s
    w13_z, w2_z = layer._w4a8_w13z, layer._w4a8_w2z
    dev = w13_op.device

    # Stock fused_experts reference: the original uint8 standard-format weights the
    # WNA16 hook leaves registered (it ran _orig_process first; our op-layout views
    # share their storage). Same quant_config the runtime stock path would use.
    w13_u8 = layer.w13_qweight.data
    w2_u8 = layer.w2_qweight.data
    qc = int4_w4a16_moe_quant_config(
        w1_scale=layer.w13_scales.data, w2_scale=layer.w2_scales.data,
        w1_zp=layer.w13_qzeros.data if w13_z is not None else None,
        w2_zp=layer.w2_qzeros.data if w2_z is not None else None,
        block_shape=[0, group])

    def _bench(fn, it=30, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / it

    wins = []
    for M in _MOE_AUTOTUNE_MGRID:
        x = (torch.randn(M, hidden, dtype=torch.float16, device=dev) * 0.5)
        tids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
        tw = torch.rand(M, top_k, dtype=torch.float32, device=dev)
        ours = lambda: _run_grouped_moe(
            x, w13_op, w2_op, w13_s, w2_s, w13_z, w2_z, tw, tids,
            MoEActivation.SILU, E, None, False, _moe_kernel(), out_dtype=x.dtype)
        stock = lambda: fused_experts(
            x, w13_u8, w2_u8, topk_weights=tw, topk_ids=tids,
            activation=MoEActivation.SILU, apply_router_weight_on_input=False,
            global_num_experts=E, expert_map=None, quant_config=qc)
        r = _bench(ours) / _bench(stock)
        wins.append((M, r < (1.0 - _MOE_AUTOTUNE_MARGIN)))
    window = _moe_winning_intervals(wins)
    _moe_persist_crossover(E, hidden, inter, group, top_k, window)
    return window


def _moe_should_engage(M, w13_op, w2_op, w13_s, topk_ids, layer=None) -> bool:
    """Whether to run our grouped op (vs falling back to stock) for this batch.
    Env VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M forces a simple M>=threshold (tuning /
    forcing). Otherwise consult the AOT crossover cache for this exact (sharded)
    expert shape: engage iff lo<=M<=hi; unknown shape -> stock (never regress).

    On a cache MISS (a shape that wasn't AOT-tuned) the grouped op would be DEAD
    WEIGHT (always-stock); so if VLLM_ROCM_W4A8_AUTOTUNE is on (default) and the
    caller passed `layer` (it carries the converted weights needed for the A/B),
    we autotune this exact shape ONCE on the first batch that hits it, persist the
    window, and use it. top_k only materialises per-batch, so first-batch (not
    load-time) is the natural seam. ROBUST: any autotune failure caches an empty
    window -> stock, so the served pathway is always >= stock, tuned or not."""
    force = _force_mode()
    if force == "on":
        return True
    if force == "off":
        return False
    env = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M")
    if env is not None:
        return M >= int(env)
    E = w13_op.size(0)
    hidden = w2_op.size(1)
    inter = w2_op.size(2) * 8
    top_k = topk_ids.size(1)
    group = hidden // w13_s.size(2)
    win = _moe_crossover_for(E, hidden, inter, group, top_k)
    if (win is None
            and layer is not None
            and _moe_autotune_enabled()
            and f"{E},{hidden},{inter},{group},{top_k}" not in _moe_load_cache()):
        try:
            win = _moe_autotune_crossover(layer, E, hidden, inter, group, top_k)
        except Exception:  # pragma: no cover - defensive; never regress
            _moe_persist_crossover(E, hidden, inter, group, top_k, None)
            win = None
    if not win:
        return False
    # list of proven-winning [lo, hi] intervals; engage iff M lands in one.
    return any(lo <= M <= hi for lo, hi in win)


def _choose_block_m(M: int, top_k: int, num_experts: int) -> int:
    """moe_align block size = our v5 tile M. block_m/16 = warps/block, so larger
    block_m -> higher occupancy (but more padding at low per-expert load). Env
    override VLLM_ROCM_W4A8_FP8_WMMA_MOE_BLOCK_M forces a fixed value (tuning)."""
    env = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_BLOCK_M")
    if env:
        return int(env)
    # Occupancy-first: block_m=64 = 4 warps/block, which dominates the per-expert
    # padding cost in micro-benchmarks (we win at M=64-512 with bm=64 but lose
    # with bm=16/32). The dispatch routes tiny-M decode to the stock kernel, so
    # the engaged path always has enough rows to justify the largest tile.
    return 64


# --------------------------------------------------------------------------- #
# AWQ MoE weight conversion -> our grouped-op layout
# --------------------------------------------------------------------------- #
def _awq_to_op_layout_single(
    qw_e: torch.Tensor,   # (K, N//pf) int32, AWQ-packed along output, AWQ order
    sc_e: torch.Tensor,   # (K//group, N)
    qz_e: torch.Tensor,   # (K//group, N//pf) int32, AWQ-packed along output
    pf: int,
    size_bits: int,
    rev: torch.Tensor,    # (pf,) long  reverse-awq order
    shifts: torch.Tensor, # (pf,) int32 = [0, b, 2b, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert ONE expert's AWQ tensors to our op layout. Per-expert (not
    batched over E) to bound the transient unpacked-weight memory."""
    mask = (1 << size_bits) - 1
    K, Np = qw_e.shape
    N = Np * pf
    Gz = qz_e.shape[0]                                  # num groups (= K // group)
    dev = qw_e.device

    # ---- qweight: unpack AWQ -> (K, N) uint4, fix AWQ nibble order ----
    uw = (qw_e.unsqueeze(-1) >> shifts) & mask         # (K, Np, pf)
    uw = uw[:, :, rev].reshape(K, N)                   # (K, N) natural channels
    uw = uw.t().contiguous().to(torch.int32)           # (N, K)
    # repack along K (input), standard nibble order: nibble j = input k8*pf + j
    w_packed = torch.zeros((N, K // pf), dtype=torch.int32, device=dev)
    for j in range(pf):
        w_packed |= (uw[:, j::pf] & mask) << (j * size_bits)

    # ---- scales: (G, N) -> (N, G) fp16 ----
    scales_op = sc_e.t().contiguous().to(torch.float16)

    # ---- qzeros: unpack AWQ -> (G, N) uint4 -> (N, G) -> pack along N ----
    uz = (qz_e.unsqueeze(-1) >> shifts) & mask         # (G, N//pf, pf)
    uz = uz[:, :, rev].reshape(Gz, N)                  # (G, N) natural channels
    uz = uz.t().contiguous().to(torch.int32)           # (N, G)
    zeros_op = torch.zeros((N // pf, Gz), dtype=torch.int32, device=dev)
    for j in range(pf):
        zeros_op |= (uz[j::pf, :] & mask) << (j * size_bits)

    return w_packed, scales_op, zeros_op


def _awq_moe_to_op_layout(
    qweight: torch.Tensor,  # (E, K, N//pf) int32
    scales: torch.Tensor,   # (E, K//group, N)
    qzeros: torch.Tensor,   # (E, K//group, N//pf) int32
    group_size: int,
    size_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched (looped over E) AWQ -> our op layout. Returns
    w_packed (E, N, K//pf) int32, scales (E, N, K//group) fp16,
    zeros (E, N//pf, K//group) int32."""
    pf = 32 // size_bits
    E, K, Np = qweight.shape
    dev = qweight.device
    rev = torch.tensor(_REVERSE_AWQ_PACK_ORDER[:pf], dtype=torch.long, device=dev)
    shifts = torch.arange(0, 32, size_bits, dtype=torch.int32, device=dev)

    wq_l, sc_l, zp_l = [], [], []
    for e in range(E):
        wpe, spe, zpe = _awq_to_op_layout_single(
            qweight[e], scales[e], qzeros[e], pf, size_bits, rev, shifts)
        wq_l.append(wpe)
        sc_l.append(spe)
        zp_l.append(zpe)
    return (torch.stack(wq_l).contiguous(),
            torch.stack(sc_l).contiguous(),
            torch.stack(zp_l).contiguous())


# --------------------------------------------------------------------------- #
# compressed-tensors MoE weight conversion -> our grouped-op layout
# --------------------------------------------------------------------------- #
# compressed-tensors pack-quantized MoE weights are GPTQ-convention (packed
# along the INPUT dim K, natural nibble order: nibble j = input row k8*pf+j),
# UNLIKE AWQ (packed along output, reverse bit order). Symmetric int4 is
# uint4b8 (stored q in [0,15] = signed (q-8)); the op's symmetric mode applies
# the implicit zp=8, so the converter just repacks the nibbles into (N, K//pf)
# and transposes the scales -- the SAME convention the dense GPTQ path was
# validated against (vllm_adapter._process_weights_after_loading GPTQ branch).
def _ct_to_op_layout_single(
    qw_e: torch.Tensor,   # (K//pf, N) int32, GPTQ-packed along K, natural order
    sc_e: torch.Tensor,   # (K//g, N)
    pf: int,
    size_bits: int,
    shifts: torch.Tensor,  # (pf,) int32 = [0, b, 2b, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ONE expert's compressed-tensors (GPTQ-style) tensors to our op
    layout. Per-expert to bound transient unpacked-weight memory."""
    mask = (1 << size_bits) - 1
    Kp, N = qw_e.shape
    K = Kp * pf
    dev = qw_e.device

    # unpack (K//pf, N) -> (K//pf, pf, N) -> (K, N): nibble j of group k8 is the
    # input channel k8*pf + j (natural order).
    uw = (qw_e.unsqueeze(1) >> shifts.view(1, pf, 1)) & mask   # (K//pf, pf, N)
    uw = uw.reshape(K, N)                                       # (K, N)
    w_kn = uw.t().contiguous().to(torch.int32)                 # (N, K)
    # repack along K (input), standard nibble order: nibble j = input k8*pf + j
    w_packed = torch.zeros((N, K // pf), dtype=torch.int32, device=dev)
    for j in range(pf):
        w_packed |= (w_kn[:, j::pf] & mask) << (j * size_bits)

    scales_op = sc_e.t().contiguous().to(torch.float16)        # (N, K//g)
    return w_packed, scales_op


def _ct_moe_to_op_layout(
    qweight: torch.Tensor,  # (E, K//pf, N) int32, GPTQ-packed (uint4b8, symmetric)
    scales: torch.Tensor,   # (E, K//g, N)
    size_bits: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched (looped over E) compressed-tensors -> our op layout. Returns
    w_packed (E, N, K//pf) int32, scales (E, N, K//g) fp16. Symmetric only
    (no zeros; the op uses the implicit uint4b8 zp=8)."""
    pf = 32 // size_bits
    E = qweight.shape[0]
    dev = qweight.device
    shifts = torch.arange(0, 32, size_bits, dtype=torch.int32, device=dev)
    wl, sl = [], []
    for e in range(E):
        w, s = _ct_to_op_layout_single(qweight[e], scales[e], pf, size_bits, shifts)
        wl.append(w)
        sl.append(s)
    return torch.stack(wl).contiguous(), torch.stack(sl).contiguous()


# --------------------------------------------------------------------------- #
# MoeWNA16 weight conversion -> our grouped-op layout (zero-copy)
# --------------------------------------------------------------------------- #
# MoeWNA16Method (vllm/.../quantization/moe_wna16.py) is the WNA16 fallback for
# AWQ/GPTQ MoE when AWQMoeMarlin is unsupported (the path Qwen3.6-35B-A3B takes
# on gfx1201). Its weight LOADER (`convert_awq_tensor`) stores weights in a
# de-AWQ'd STANDARD format, in uint8 (2 nibbles/byte):
#   w13_qweight (E, N, K//2) uint8, packed-2-along-K (LAST dim, natural order)
#   w13_qzeros  (E, N//2, K//g) uint8, packed-2-along-N (DIM 1)
#   w13_scales  (E, N, K//g)
# qweight is K-packed on the last dim -> a zero-copy `.view(int32)` yields our op
# layout (E, N, K//8). qzeros is N-packed on DIM 1 (not last) -> a flat view would
# wrongly collapse K, so combine 4 consecutive N-bytes into one int32 to get
# (E, N//8, K//g) (op convention: int32[n8] nibble j = zp of output 8*n8+j).
# AWQ zeros are raw zp (dequant (q-zp)*scale), matching the op + the validated
# dense AWQ path. has_zp=False (GPTQ-sym) -> z=None, implicit uint4b8 zp=8.
def _wna16_moe_to_op_layout(qweight_u8, scales, qzeros_u8):
    w = qweight_u8.contiguous().view(torch.int32)            # (E, N, K//8)
    s = scales if scales.dtype == torch.float16 else scales.to(torch.float16)
    z = None
    if qzeros_u8 is not None:
        E, Nh, Kg = qzeros_u8.shape                          # Nh = N//2
        zi = qzeros_u8.contiguous().to(torch.int32).reshape(E, Nh // 4, 4, Kg)
        z = (zi[:, :, 0, :] | (zi[:, :, 1, :] << 8)
             | (zi[:, :, 2, :] << 16) | (zi[:, :, 3, :] << 24)).contiguous()
    return w, s, z


def _supported_group_size(gs: int) -> bool:
    return gs is not None and gs >= 16 and gs <= 128 and gs % 16 == 0


# --------------------------------------------------------------------------- #
# shared grouped-MoE composition (used by BOTH the modular AWQ experts and the
# monolithic compressed-tensors hook)
# --------------------------------------------------------------------------- #
def _run_grouped_moe(x, w1, w2, w1_s, w2_s, w1_zp, w2_zp,
                     topk_weights, topk_ids, activation,
                     global_num_experts, expert_map,
                     apply_router_weight_on_input, kernel, out_dtype=None):
    """Compose the whole gated MoE from the one grouped FP8-WMMA op and return
    the final ``(M, K)`` output (already topk-weighted + reduced).

    moe_align -> grouped GEMM(w1, gather x by sorted//top_k) -> silu_and_mul ->
    grouped GEMM(w2, identity-gather top_k=1) -> fp32 topk scatter-reduce.

    ``activation`` is a ``MoEActivation`` (gated). ``w*_zp`` may be ``None`` for
    symmetric weights (the op then uses the implicit uint4b8 zp=8). Weights are
    in our op layout: w1 ``(E, 2*inter, K//8)``, w2 ``(E, K, inter//8)``,
    scales ``(E, N, K//g)``. This is the single validated implementation
    (``test_moe_experts.py`` mirrors it; 11/11 on gfx1201)."""
    import w4a8_fp8_wmma
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    mmq = w4a8_fp8_wmma.mmq_fp8_moe_gemm
    M = x.size(0)
    K = x.size(-1)
    top_k = topk_ids.size(1)
    E_local = w1.size(0)
    dev = x.device
    if out_dtype is None:
        out_dtype = x.dtype

    # Bounded-memory chunking: the padded-sorted scratch (out1/buf2) is O(M*top_k),
    # which can spike past the KV-cache budget at large M -- e.g. vLLM's startup
    # memory-profiling dummy run at max_num_batched_tokens. Process tokens in chunks
    # so peak scratch is O(chunk*top_k) regardless of M. No-op when M <= chunk (the
    # engaged decode/mid-M regime). Skipped for router-weight-on-input (top_k==1,
    # tiny scratch, and the weight is already folded into x).
    chunk = int(os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_APPLY_CHUNK", "4096"))
    if chunk and M > chunk and not apply_router_weight_on_input:
        out = torch.empty((M, K), dtype=out_dtype, device=dev)
        for lo in range(0, M, chunk):
            hi = min(lo + chunk, M)
            out[lo:hi] = _run_grouped_moe(
                x[lo:hi], w1, w2, w1_s, w2_s, w1_zp, w2_zp,
                topk_weights[lo:hi], topk_ids[lo:hi], activation,
                global_num_experts, expert_map, False, kernel, out_dtype)
        return out

    # The op requires fp16 scales; convert defensively (no-op if already fp16).
    if w1_s.dtype != torch.float16:
        w1_s = w1_s.to(torch.float16)
    if w2_s.dtype != torch.float16:
        w2_s = w2_s.to(torch.float16)
    if w1_zp is None:
        w1_zp = torch.empty(0, dtype=torch.int32, device=dev)
    if w2_zp is None:
        w2_zp = torch.empty(0, dtype=torch.int32, device=dev)

    # DECODE specialisation: at small M the WMMA grouped kernel ("wmma") wastes
    # most of its block_m-row tiles on routing padding (1-2 real rows/expert)
    # and pays a __syncthreads per K-group. "gemv" is a grouped GEMV that compacts
    # to the REAL routed rows and streams weights barrier-free -- op micro-bench
    # (Qwen3.6 expert shape, gemm1): 3-4.6x faster than wmma at T<=8, crossing back
    # near T~64. Use it (with the smallest block_m, less padding) for gemm1, and
    # its SCATTER variant for the fused gemm2+reduce (the atomic scatter is
    # contention-free at decode). Gated by M<=GEMV_MAX_M; "scalar" (golden) opts out.
    gemv_max = int(os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_GEMV_MAX_M", "96"))
    use_gemv = kernel != "scalar" and M <= gemv_max
    # GEMV uses block_m=8 (not a WMMA multiple -- it doesn't tile): caps real
    # rows/block at 8 so the acc[COLS][MMAX] register footprint stays small
    # (MMAX=8) even at batched decode M=16-32, where block_m=16 oversized MMAX
    # and crushed occupancy. Large per-expert counts just spill into more blocks.
    gemv_bm = int(os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_GEMV_BLOCK_M", "8"))
    block_m = gemv_bm if use_gemv else _choose_block_m(M, top_k, E_local)
    gver = "gemv" if use_gemv else kernel
    # pad_sorted_ids=True rounds the sorted_ids length up to a multiple of
    # block_m (the op binding needs P % block_m == 0 and exactly P/block_m
    # blocks). The trailing rows past num_tokens_post_padded (ntp) are left
    # uninitialised by the C op, so every consumer masks rows by (row < ntp).
    sorted_ids, expert_ids, ntp = moe_align_block_size(
        topk_ids, block_m, global_num_experts, expert_map,
        pad_sorted_ids=True, ignore_invalid_experts=True)
    P = sorted_ids.size(0)

    x_in = x
    if apply_router_weight_on_input:
        assert top_k == 1, "router-weight-on-input requires top_k==1"
        x_in = x_in * topk_weights.reshape(M, 1).to(x_in.dtype)
    x16 = x_in.to(torch.float16) if x_in.dtype != torch.float16 else x_in
    x16 = x16.contiguous()

    # gemm1 (w13) -> (gate * silu(up)) -> (P, inter). The "wmma" path fuses
    # silu_and_mul into gemm1's epilogue (mmq_fp8_moe_gemm1_silu), so gemm1 writes
    # the post-activation (P, inter) DIRECTLY -- dropping the (P, 2*inter) out1 +
    # (P, inter) buf2 HBM round-trip and the separate silu launch (ROADMAP Task 3,
    # the prefill/mid intermediate-traffic win; at DECODE the wall is gemm2's
    # weight-read BW, not these buffers -- so the use_gemv decode branch keeps the
    # unfused "gemv" path). Fused only for gated SILU on "wmma"; everything else
    # ("scalar" golden, GELU/other activations, the GEMV decode path) takes the
    # unfused gemm1 + apply_moe_activation. Bit-exact to the unfused path (see the
    # kernel's moe_silu_and_mul_h). Env-gated for A/B + bit-exact verification.
    # NOTE: the silu-fused kernel has NO tiled equivalent -- it keeps its own
    # v5/v6 A-residence kernels, selected in C++ by VLLM_W4A8_MOE_A_IN_LDS.
    fuse_silu = os.environ.get(
        "VLLM_ROCM_W4A8_FP8_WMMA_MOE_FUSE_SILU", "1") == "1"
    can_fuse = (fuse_silu and not use_gemv and gver == "wmma"
                and activation.is_gated and activation == MoEActivation.SILU)
    if can_fuse:
        buf2 = w4a8_fp8_wmma.mmq_fp8_moe_gemm1_silu(
            x16, w1, w1_s, sorted_ids, expert_ids, ntp,
            top_k, block_m, kernel=gver, w_zeros=w1_zp)     # (P, inter)
    else:
        # gemm1 (w13): padded-sorted (P, 2*inter)
        out1 = mmq(x16, w1, w1_s, sorted_ids, expert_ids, ntp,
                   top_k, block_m, kernel=gver, w_zeros=w1_zp)
        # gate * up -> (P, inter)
        act_dim = out1.size(1) // 2 if activation.is_gated else out1.size(1)
        buf2 = torch.empty((P, act_dim), dtype=torch.float16, device=dev)
        apply_moe_activation(activation, buf2, out1)

    if apply_router_weight_on_input:
        tw_flat = torch.ones(M * top_k, dtype=torch.float32, device=dev)
    else:
        tw_flat = topk_weights.reshape(-1).to(torch.float32).contiguous()

    if use_gemv:
        # gemm2 via the fused "gemv" SCATTER epilogue. gemv compacts to the real
        # routed slots (identity-gather src=row_pad, validity from sorted_ids), so
        # unlike the non-scatter+gather_reduce path it never touches the padding
        # rows. The atomic scatter is contention-free at decode (few tokens).
        # output must be pre-zeroed (captured in any HIP graph).
        output = torch.zeros((M, K), dtype=torch.float32, device=dev)
        w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter(
            buf2.contiguous(), w2, w2_s, sorted_ids, expert_ids, ntp, tw_flat,
            output, top_k, block_m, kernel=gver, w_zeros=w2_zp)
        return output.to(out_dtype)

    # gemm2 (w2) + topk-weight + reduce. SINGLE PATH (no adaptive branch): gemm2
    # NON-scatter -> (P,N), then the custom HIP gather-reduce kernel. The old fused
    # atomic-scatter epilogue's top_k-contended global atomicAdds dominated at
    # prefill (+3.7ms at T=2048); this kernel does a contention-free per-token
    # gather-reduce. GPU microbench (35B-A3B dims, bit-matches scatter rel~2e-4):
    # 3.71x faster than scatter at prefill, 1.83x at mid, ~par at decode (~50us
    # behind, <0.3% e2e). HIP-graph safe everywhere (no atomics, static shapes).
    ident = torch.arange(P, dtype=torch.int32, device=dev)
    out2 = mmq(buf2.contiguous(), w2, w2_s, ident, expert_ids, ntp, 1, block_m,
               kernel=kernel, w_zeros=w2_zp)                          # (P, K) fp16
    acc = w4a8_fp8_wmma.mmq_fp8_moe_gather_reduce(
        out2.contiguous(), sorted_ids, tw_flat, ntp, top_k)           # (M, K) fp32
    return acc.to(out_dtype)


# --------------------------------------------------------------------------- #
# torch.compile / cudagraph wrapper for the grouped MoE
# --------------------------------------------------------------------------- #
# Same fix as the dense path (vllm_adapter._w4a8_dense_apply): _run_grouped_moe has
# raw pybind ops (no fake, unpickleable) AND symbolic-M control flow (use_gemv / block_m
# / chunking), so tracing it directly under vLLM's full torch.compile + cudagraph aborts
# (fake-tensor / pickle / sympy-GreaterThan). Wrapping the whole compute in ONE opaque
# vLLM custom op makes it a fake-backed, pickle-clean, in-graph node: the M-branches run
# eager inside it. The op is FUNCTIONAL (returns the (M,K) output); the internal in-place
# scatter/copy are hidden, so no mutates_args. MoEActivation is a str-valued enum, passed
# by value and reconstructed inside.
def _w4a8_moe_apply(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
    kernel: str,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    act = MoEActivation(activation)
    return _run_grouped_moe(
        x, w1, w2, w1_s, w2_s, w1_zp, w2_zp, topk_weights, topk_ids, act,
        global_num_experts, expert_map, apply_router_weight_on_input, kernel,
        out_dtype=x.dtype)


def _w4a8_moe_apply_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_s: torch.Tensor,
    w2_s: torch.Tensor,
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str,
    global_num_experts: int,
    expert_map: Optional[torch.Tensor],
    apply_router_weight_on_input: bool,
    kernel: str,
) -> torch.Tensor:
    # MoE preserves the hidden dim: output is (M, K) == x.shape, dtype = x.dtype.
    return torch.empty_like(x)


if not hasattr(torch.ops.vllm, "w4a8_moe_apply"):
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:  # older vLLM layout
        from vllm.utils import direct_register_custom_op
    direct_register_custom_op(
        op_name="w4a8_moe_apply",
        op_func=_w4a8_moe_apply,
        mutates_args=[],
        fake_impl=_w4a8_moe_apply_fake,
    )


# --------------------------------------------------------------------------- #
# modular-kernel experts class
# --------------------------------------------------------------------------- #
def _build_experts_class():
    """Build the experts class lazily so importing this module never requires
    vLLM (register_moe imports it only inside the EngineCore)."""
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )
    from vllm.model_executor.layers.quantization.utils import quant_utils as _qu
    from vllm.platforms import current_platform

    # The AWQ MoE oracle passes kInt4Static; newer vLLMs also expose g32/asym
    # variants. Accept whichever THIS vLLM build defines — only kInt4Static is
    # guaranteed (e.g. vllm 0.21.1 has just that one). Our override path bypasses
    # the oracle, so this set only gates the (then-unused) _supports_quant_scheme.
    _SUPPORTED_W = {
        getattr(_qu, _n)
        for _n in ("kInt4Static", "kInt4Static32",
                   "kInt4StaticAsym", "kInt4Static32Asym")
        if hasattr(_qu, _n)
    }

    class W4A8Fp8WmmaExperts(mk.FusedMoEExpertsModular):
        """RDNA4 FP8-WMMA grouped GEMM experts for AWQ-4bit (uint4 + zero points)."""

        # -- oracle support gates -------------------------------------------- #
        @staticmethod
        def _supports_current_device() -> bool:
            return current_platform.is_rocm() and _on_gfx12x()

        @staticmethod
        def _supports_no_act_and_mul() -> bool:
            # We do gemm1(2*inter) -> silu_and_mul -> gemm2; gated only.
            return False

        @staticmethod
        def _supports_quant_scheme(weight_key, activation_key) -> bool:
            # int4 weight-only (bf16/fp16 activations). The oracle passes
            # kInt4Static for AWQ MoE; accept the asym variants too.
            if activation_key is not None:
                return False
            return weight_key in _SUPPORTED_W

        @staticmethod
        def _supports_activation(activation) -> bool:
            return activation in (MoEActivation.SILU, MoEActivation.GELU,
                                  MoEActivation.GELU_TANH)

        @staticmethod
        def _supports_parallel_config(moe_parallel_config) -> bool:
            return not (
                moe_parallel_config.use_fi_nvl_two_sided_kernels
                or moe_parallel_config.use_fi_nvl_one_sided_kernels
            )

        @staticmethod
        def activation_format() -> "mk.FusedMoEActivationFormat":
            return mk.FusedMoEActivationFormat.Standard

        def finalize_weight_and_reduce_impl(self) -> "mk.TopKWeightAndReduce":
            # apply() does the topk-weight multiply + reduce itself.
            return TopKWeightAndReduceNoOP()

        # -- problem size / workspaces --------------------------------------- #
        def moe_problem_size(self, a1, w1, w2, topk_ids):
            # w1 = w13 (E, 2*inter, K//8); w2 (E, hidden, inter//8). N here is the
            # intermediate size (inter), K is hidden, both un-packed.
            assert w1.dim() == 3 and w2.dim() == 3
            E = w1.size(0)
            K = a1.size(-1)
            N = w2.size(2) * 8           # inter (w2 packs inter//8 along last dim)
            assert a1.dim() == 2 and topk_ids.size(0) == a1.size(0)
            M = a1.size(0)
            topk = topk_ids.size(1)
            return E, M, N, K, topk

        def workspace_shapes(self, M, N, K, topk, global_num_experts,
                             local_num_experts, expert_tokens_meta, activation):
            # apply() allocates its own padded-sorted scratch; these only size the
            # framework's (aliased) output buffer, whose real shape is (M, K).
            workspace1 = (M * topk, max(N, K))
            workspace2 = (M * topk, N)
            output = (M, K)
            return (workspace1, workspace2, output)

        # -- the MoE itself -------------------------------------------------- #
        def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
                  activation, global_num_experts, expert_map, a1q_scale, a2_scale,
                  workspace13, workspace2, expert_tokens_meta,
                  apply_router_weight_on_input):
            # Whole gated MoE from the one grouped op (shared impl). w*_zp may be
            # None for symmetric. self.w1_scale/w1_zp come from the quant_config.
            # Through the opaque vllm:: op so the symbolic-M ladder runs eager and
            # the node stays in-graph under torch.compile + cudagraph (see
            # _w4a8_moe_apply). output.copy_ also casts to output.dtype, so the op
            # returns x.dtype. activation is a str-valued MoEActivation enum.
            act = activation.value if hasattr(activation, "value") else activation
            out = torch.ops.vllm.w4a8_moe_apply(
                hidden_states, w1, w2, self.w1_scale, self.w2_scale,
                self.w1_zp, self.w2_zp, topk_weights, topk_ids, act,
                global_num_experts, expert_map, apply_router_weight_on_input,
                _moe_kernel())
            output.copy_(out)

    return W4A8Fp8WmmaExperts


# Lazily-built singleton (so the class body's vLLM imports run only at register).
_EXPERTS_CLS = None


def get_experts_cls():
    global _EXPERTS_CLS
    if _EXPERTS_CLS is None:
        _EXPERTS_CLS = _build_experts_class()
    return _EXPERTS_CLS


# --------------------------------------------------------------------------- #
# AWQ-MoE hook (surgical: only AWQMarlinMoEMethod, never GPTQ / compressed-tensors)
# --------------------------------------------------------------------------- #
class _W4A8MoEBackend:
    """Sentinel standing in for a WNA16MoEBackend enum member (the enum can't be
    extended at runtime). Carries `.value`/`.name` for logging + identity; used as
    the dispatch key for our patched convert/make in the awq_marlin namespace."""
    value = "W4A8_FP8_WMMA"
    name = "W4A8_FP8_WMMA"

    def __repr__(self):
        return "WNA16MoEBackend.W4A8_FP8_WMMA"


W4A8_FP8_WMMA_BACKEND = _W4A8MoEBackend()

_MOE_REGISTERED = False
_OVERRIDE_LOGGED = False


def register_moe(verbose: bool = True) -> bool:
    """Route AWQ-4bit asymmetric MoE expert layers on gfx12x to our grouped
    FP8-WMMA kernel. Idempotent; no-op off gfx12x or when disabled.

    Surgical, NOT a global oracle patch: the WNA16 MoE oracle
    (`int_wna16._get_priority_backends`/`backend_to_kernel_cls`) is shared by
    GPTQ (auto_gptq), AWQ (awq_marlin) AND compressed-tensors MoE — prepending a
    backend there would hijack the other two (they also pass `kInt4Static`) and
    crash in their own un-patched `convert_to_wna16_moe_kernel_format`. Instead we
    only:
      - wrap `AWQMarlinMoEMethod.__init__` to override `wna16_moe_backend` /
        `experts_cls` to ours when the layer is a supported AWQ-asym-4bit MoE; and
      - patch `convert_to_wna16_moe_kernel_format` / `make_wna16_moe_kernel` in the
        `awq_marlin` namespace ONLY (dispatching on our sentinel / class).
    GPTQ and compressed-tensors MoE are left entirely on Marlin.
    """
    global _MOE_REGISTERED
    if _MOE_REGISTERED:
        return True
    _raise_if_removed_moe_env()  # loud at boot; can't be swallowed by autotune
    if not _moe_enabled():
        if verbose:
            print("[w4a8_fp8_wmma] MoE disabled via env")
        return False
    if not _on_gfx12x():
        if verbose:
            print("[w4a8_fp8_wmma] MoE: not gfx12x, leaving AWQ MoE on Marlin")
        return False

    import w4a8_fp8_wmma  # noqa: F401  (ensure the op is loaded)
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.quantization import awq_marlin
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinConfig,
        AWQMarlinMoEMethod,
    )

    our_experts_cls = get_experts_cls()

    _orig_init = AWQMarlinMoEMethod.__init__
    _orig_convert = awq_marlin.convert_to_wna16_moe_kernel_format
    _orig_make = awq_marlin.make_wna16_moe_kernel

    def _patched_init(self, quant_config, moe):
        _orig_init(self, quant_config, moe)
        try:
            ok = (
                isinstance(quant_config, AWQMarlinConfig)
                and getattr(quant_config, "zero_point", False)
                and quant_config.weight_bits == 4
                and _supported_group_size(quant_config.group_size)
            )
        except Exception:
            ok = False
        if ok:
            self.wna16_moe_backend = W4A8_FP8_WMMA_BACKEND
            self.experts_cls = our_experts_cls
            global _OVERRIDE_LOGGED
            if verbose and not _OVERRIDE_LOGGED:
                _OVERRIDE_LOGGED = True
                print("[w4a8_fp8_wmma] AWQ MoE -> W4A8Fp8WmmaExperts "
                      f"(g={quant_config.group_size}); grouped FP8-WMMA path")

    def _patched_convert(backend, layer, quant_config, input_dtype, w13, w2,
                         w13_scale, w2_scale, w13_g_idx=None, w2_g_idx=None,
                         w13_qzeros=None, w2_qzeros=None, w13_bias=None,
                         w2_bias=None):
        if backend is not W4A8_FP8_WMMA_BACKEND:
            return _orig_convert(
                backend, layer, quant_config, input_dtype, w13, w2, w13_scale,
                w2_scale, w13_g_idx, w2_g_idx, w13_qzeros, w2_qzeros, w13_bias,
                w2_bias)

        if w13_qzeros is None or w2_qzeros is None:
            raise ValueError(
                "W4A8 FP8 WMMA MoE requires AWQ zero points (asymmetric uint4).")
        gs = quant_config.group_size
        bits = quant_config.weight_bits
        if not _supported_group_size(gs):
            raise ValueError(
                f"W4A8 FP8 WMMA MoE: group_size={gs} unsupported "
                "(need multiple of 16 in [16,128]).")

        w13_p, w13_s, w13_z = _awq_moe_to_op_layout(w13, w13_scale, w13_qzeros,
                                                    gs, bits)
        w2_p, w2_s, w2_z = _awq_moe_to_op_layout(w2, w2_scale, w2_qzeros, gs, bits)
        E = w13.shape[0]
        empty_si = torch.empty((E, 0), dtype=torch.int32, device=w13.device)
        return (w13_p, w2_p, w13_s, w2_s, None, None, empty_si, empty_si,
                w13_z, w2_z, None, None, w13_bias, w2_bias)

    def _patched_make(moe_quant_config, moe_config, experts_cls,
                      is_k_full=False, w13_g_idx=None, w2_g_idx=None,
                      w13_g_idx_sort_indices=None, w2_g_idx_sort_indices=None,
                      routing_tables=None):
        # NOTE: the param MUST be named `experts_cls` — _setup_kernel calls
        # make_wna16_moe_kernel(..., experts_cls=...) by keyword. The closure's
        # own class is `our_experts_cls` to avoid shadowing.
        if experts_cls is not our_experts_cls:
            return _orig_make(
                moe_quant_config, moe_config, experts_cls, is_k_full,
                w13_g_idx, w2_g_idx, w13_g_idx_sort_indices,
                w2_g_idx_sort_indices, routing_tables)
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        prepare_finalize = maybe_make_prepare_finalize(
            moe=moe_config, quant_config=moe_quant_config,
            routing_tables=routing_tables, allow_new_interface=True,
            use_monolithic=False)
        assert prepare_finalize is not None
        experts = our_experts_cls(moe_config=moe_config,
                                  quant_config=moe_quant_config)
        return mk.FusedMoEKernel(prepare_finalize, experts)

    AWQMarlinMoEMethod.__init__ = _patched_init
    awq_marlin.convert_to_wna16_moe_kernel_format = _patched_convert
    awq_marlin.make_wna16_moe_kernel = _patched_make

    _MOE_REGISTERED = True
    if verbose:
        print("[w4a8_fp8_wmma] AWQ MoE hook installed (gfx12x): supported "
              "AWQ-4bit asym experts -> grouped FP8-WMMA; GPTQ/CT via own hooks")
    return True


# --------------------------------------------------------------------------- #
# GPTQ-MoE hook (auto_gptq; symmetric uint4b8). Mirrors the AWQ oracle override,
# but in the `auto_gptq` namespace and with the SYMMETRIC conversion (no zeros).
# --------------------------------------------------------------------------- #
# GPTQ-4bit MoE routes through `AutoGPTQMoEMethod`, which (like AWQMarlinMoEMethod)
# picks its experts via the shared `select_wna16_moe_backend` oracle -> stock
# `MarlinExperts` on gfx1201. register_moe only patches the `awq_marlin` namespace,
# so GPTQ was left on Marlin (it reached us only via the marlin-UNsupported fallback
# to MoeWNA16Method). This hook closes that gap for the common marlin-supported case.
_MOE_GPTQ_REGISTERED = False
_GPTQ_OVERRIDE_LOGGED = False


def register_moe_gptq(verbose: bool = True) -> bool:
    """Route GPTQ-4bit (symmetric uint4b8, no desc_act) MoE expert layers on gfx12x
    to our grouped FP8-WMMA kernel by overriding `AutoGPTQMoEMethod`'s oracle pick +
    patching convert/make in the `auto_gptq` namespace ONLY. Symmetric -> w_zeros=None
    (implicit uint4b8 zp=8); GPTQ weights are GPTQ-convention K-packed, same as the
    compressed-tensors path -> reuse `_ct_moe_to_op_layout`. Idempotent; AWQ/CT/Marlin
    untouched; no-op off gfx12x / disabled / unsupported config.

    NOTE (gfx1201): this hook is DORMANT on RDNA4 — `check_moe_marlin_supports_layer`
    is False there, so `AutoGPTQMoEMethod.get_quant_method` immediately falls back to
    `MoeWNA16Config -> MoeWNA16Method` BEFORE this override matters (verified
    2026-06-13: GPTQ MoE runs through `register_moe_wna16`, which engages our op under
    FORCE/crossover exactly like AWQ). This hook only fires on hardware where
    GPTQMoeMarlin IS supported (AutoGPTQMoEMethod actually used) — kept for that case,
    untested on it."""
    global _MOE_GPTQ_REGISTERED
    if _MOE_GPTQ_REGISTERED:
        return True
    _raise_if_removed_moe_env()  # loud at boot; can't be swallowed by autotune
    if not _moe_enabled():
        if verbose:
            print("[w4a8_fp8_wmma] GPTQ MoE disabled via env")
        return False
    if not _on_gfx12x():
        if verbose:
            print("[w4a8_fp8_wmma] GPTQ MoE: not gfx12x, leaving on Marlin")
        return False

    import w4a8_fp8_wmma  # noqa: F401
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.quantization import auto_gptq
    from vllm.model_executor.layers.quantization.auto_gptq import (
        AutoGPTQConfig,
        AutoGPTQMoEMethod,
    )

    our_experts_cls = get_experts_cls()
    _orig_init = AutoGPTQMoEMethod.__init__
    _orig_convert = auto_gptq.convert_to_wna16_moe_kernel_format
    _orig_make = auto_gptq.make_wna16_moe_kernel

    def _gptq_supported(quant_config) -> bool:
        # 4-bit GPTQ is uint4b8 (symmetric); our op needs no activation reorder
        # (no desc_act / non-trivial g_idx) and a supported group size.
        try:
            return (isinstance(quant_config, AutoGPTQConfig)
                    and quant_config.quant_type.size_bits == 4
                    and not getattr(quant_config, "desc_act", False)
                    and _supported_group_size(quant_config.group_size))
        except Exception:
            return False

    def _patched_init(self, quant_config, moe):
        _orig_init(self, quant_config, moe)
        if _gptq_supported(quant_config):
            self.wna16_moe_backend = W4A8_FP8_WMMA_BACKEND
            self.experts_cls = our_experts_cls
            global _GPTQ_OVERRIDE_LOGGED
            if verbose and not _GPTQ_OVERRIDE_LOGGED:
                _GPTQ_OVERRIDE_LOGGED = True
                print("[w4a8_fp8_wmma] GPTQ MoE -> W4A8Fp8WmmaExperts "
                      f"(symmetric int4 g={quant_config.group_size}); grouped FP8-WMMA")

    def _patched_convert(backend, layer, quant_config, input_dtype, w13, w2,
                         w13_scale, w2_scale, w13_g_idx=None, w2_g_idx=None,
                         w13_qzeros=None, w2_qzeros=None, w13_bias=None,
                         w2_bias=None, **kw):
        if backend is not W4A8_FP8_WMMA_BACKEND:
            return _orig_convert(
                backend, layer, quant_config, input_dtype, w13, w2, w13_scale,
                w2_scale, w13_g_idx=w13_g_idx, w2_g_idx=w2_g_idx,
                w13_qzeros=w13_qzeros, w2_qzeros=w2_qzeros, w13_bias=w13_bias,
                w2_bias=w2_bias, **kw)
        # GPTQ symmetric uint4b8: K-packed (E, K//8, N) -> our (E, N, K//8); zeros=None.
        w13_p, w13_s = _ct_moe_to_op_layout(w13, w13_scale)
        w2_p, w2_s = _ct_moe_to_op_layout(w2, w2_scale)
        E = w13.shape[0]
        empty_si = torch.empty((E, 0), dtype=torch.int32, device=w13.device)
        return (w13_p, w2_p, w13_s, w2_s, None, None, empty_si, empty_si,
                None, None, None, None, w13_bias, w2_bias)

    def _patched_make(moe_quant_config, moe_config, experts_cls,
                      is_k_full=False, w13_g_idx=None, w2_g_idx=None,
                      w13_g_idx_sort_indices=None, w2_g_idx_sort_indices=None,
                      routing_tables=None):
        if experts_cls is not our_experts_cls:
            return _orig_make(
                moe_quant_config, moe_config, experts_cls, is_k_full,
                w13_g_idx, w2_g_idx, w13_g_idx_sort_indices,
                w2_g_idx_sort_indices, routing_tables)
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        prepare_finalize = maybe_make_prepare_finalize(
            moe=moe_config, quant_config=moe_quant_config,
            routing_tables=routing_tables, allow_new_interface=True,
            use_monolithic=False)
        assert prepare_finalize is not None
        experts = our_experts_cls(moe_config=moe_config,
                                  quant_config=moe_quant_config)
        return mk.FusedMoEKernel(prepare_finalize, experts)

    AutoGPTQMoEMethod.__init__ = _patched_init
    auto_gptq.convert_to_wna16_moe_kernel_format = _patched_convert
    auto_gptq.make_wna16_moe_kernel = _patched_make

    _MOE_GPTQ_REGISTERED = True
    if verbose:
        print("[w4a8_fp8_wmma] GPTQ MoE hook installed (gfx12x): symmetric int4 "
              "experts -> grouped FP8-WMMA; AWQ/CT/Marlin untouched")
    return True


# --------------------------------------------------------------------------- #
# compressed-tensors MoE hook (the real target models, e.g. Qwen3.6-35B-A3B)
# --------------------------------------------------------------------------- #
# On gfx1201 an int4 compressed-tensors MoE does NOT use the modular oracle /
# Marlin path: `rocm_moe.is_supported` is gfx1100-only, so it lands on the
# monolithic `CompressedTensorsWNA16MoEMethod` (its `apply` calls the Triton
# `fused_experts`). That class is ALWAYS symmetric group int4/int8 and has no
# `experts_cls` to override -- so unlike the AWQ hook we patch its
# `process_weights_after_loading` (convert weights to our op layout) and `apply`
# (run our grouped-op composition) directly. GPTQ/AWQ/Marlin paths are untouched.
_MOE_CT_REGISTERED = False
_CT_OVERRIDE_LOGGED = False


def register_moe_ct(verbose: bool = True) -> bool:
    """Route symmetric int4 compressed-tensors MoE expert layers on gfx12x to our
    grouped FP8-WMMA kernel by patching `CompressedTensorsWNA16MoEMethod`
    (process_weights_after_loading + apply). Idempotent; no-op off gfx12x /
    disabled / unsupported config (then the stock Triton path runs)."""
    global _MOE_CT_REGISTERED
    if _MOE_CT_REGISTERED:
        return True
    _raise_if_removed_moe_env()  # loud at boot; can't be swallowed by autotune
    if not _moe_enabled():
        if verbose:
            print("[w4a8_fp8_wmma] CT MoE disabled via env")
        return False
    if not _on_gfx12x():
        if verbose:
            print("[w4a8_fp8_wmma] CT MoE: not gfx12x, leaving on Triton")
        return False

    import w4a8_fp8_wmma  # noqa: F401  (ensure the op is loaded)
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (  # noqa: E501
        CompressedTensorsWNA16MoEMethod,
    )

    _orig_process = CompressedTensorsWNA16MoEMethod.process_weights_after_loading
    _orig_apply = CompressedTensorsWNA16MoEMethod.apply

    def _engage(method) -> bool:
        # This class is always symmetric group quant; we additionally need
        # 4-bit + a supported group size. Anything else stays on Triton.
        try:
            return (method.num_bits == 4
                    and method.strategy == "group"
                    and _supported_group_size(method.group_size))
        except Exception:
            return False

    def _patched_process(self, layer):
        self._w4a8_ct = _engage(self)
        if not self._w4a8_ct:
            return _orig_process(self, layer)
        # Registered (pre-process) layout, GPTQ-convention packed along K:
        #   w13_weight_packed (E, hidden//8, 2*inter)  scale (E, hidden//g, 2*inter)
        #   w2_weight_packed  (E, inter//8, hidden)     scale (E, inter//g, hidden)
        # Convert each to our op layout (E, N, K//8) / (E, N, K//g) fp16.
        w13 = layer.w13_weight_packed.data
        w2 = layer.w2_weight_packed.data
        w13_s = layer.w13_weight_scale.data
        w2_s = layer.w2_weight_scale.data
        w13_p, w13_sp = _ct_moe_to_op_layout(w13, w13_s)
        w2_p, w2_sp = _ct_moe_to_op_layout(w2, w2_s)
        layer._w4a8_ct_w13 = w13_p
        layer._w4a8_ct_w2 = w2_p
        layer._w4a8_ct_w13_s = w13_sp
        layer._w4a8_ct_w2_s = w2_sp
        # Free the source packed weights/scales (we hold converted copies); keep
        # the Parameter objects registered so nothing iterating params crashes.
        for nm in ("w13_weight_packed", "w2_weight_packed",
                   "w13_weight_scale", "w2_weight_scale"):
            p = getattr(layer, nm, None)
            if p is not None and hasattr(p, "data"):
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        global _CT_OVERRIDE_LOGGED
        if verbose and not _CT_OVERRIDE_LOGGED:
            _CT_OVERRIDE_LOGGED = True
            print("[w4a8_fp8_wmma] CT MoE -> grouped FP8-WMMA "
                  f"(symmetric int4 g={self.group_size}); Triton bypassed")

    def _patched_apply(self, layer, x, topk_weights, topk_ids,
                       shared_experts, shared_experts_input):
        if not getattr(self, "_w4a8_ct", False):
            return _orig_apply(self, layer, x, topk_weights, topk_ids,
                               shared_experts, shared_experts_input)
        act = layer.activation
        if not isinstance(act, MoEActivation):
            act = MoEActivation.from_str(str(act))
        x2 = x.reshape(-1, x.shape[-1])
        # Symmetric -> w*_zp=None (op uses the implicit uint4b8 zp=8).
        # NOTE: shared_experts are NOT applied here -- the stock apply also does
        # not fold them into the expert GEMM (the layer handles them). TODO:
        # confirm e2e for shared-expert models.
        out = _run_grouped_moe(
            x2, layer._w4a8_ct_w13, layer._w4a8_ct_w2,
            layer._w4a8_ct_w13_s, layer._w4a8_ct_w2_s, None, None,
            topk_weights, topk_ids, act, layer.global_num_experts,
            layer.expert_map, layer.apply_router_weight_on_input,
            _moe_kernel(), out_dtype=x.dtype)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    CompressedTensorsWNA16MoEMethod.process_weights_after_loading = _patched_process
    CompressedTensorsWNA16MoEMethod.apply = _patched_apply

    _MOE_CT_REGISTERED = True
    if verbose:
        print("[w4a8_fp8_wmma] CT MoE hook installed (gfx12x): symmetric int4 "
              "experts -> grouped FP8-WMMA; AWQ/GPTQ/Marlin untouched")
    return True


# --------------------------------------------------------------------------- #
# MoeWNA16 MoE hook (the path Qwen3.6-35B-A3B + most AWQ/GPTQ MoE take on gfx1201)
# --------------------------------------------------------------------------- #
# When an AWQ/compressed-tensors int4 MoE is NOT supported by AWQMoeMarlin (the
# usual case on gfx1201), vLLM falls back to `MoeWNA16Method` whose monolithic
# `apply` runs the Triton `fused_experts`. We patch its process_weights (zero-copy
# view of the standard-format weights into our op layout) + apply (our grouped op).
_MOE_WNA16_REGISTERED = False
_WNA16_OVERRIDE_LOGGED = False


def register_moe_wna16(verbose: bool = True) -> bool:
    """Route int4 MoeWNA16 MoE expert layers on gfx12x (AWQ/GPTQ fallback path) to
    our grouped FP8-WMMA kernel by patching `MoeWNA16Method` (process + apply).
    Idempotent; no-op off gfx12x / disabled / unsupported (then stock Triton)."""
    global _MOE_WNA16_REGISTERED
    if _MOE_WNA16_REGISTERED:
        return True
    _raise_if_removed_moe_env()  # loud at boot; can't be swallowed by autotune
    if not _moe_enabled():
        if verbose:
            print("[w4a8_fp8_wmma] WNA16 MoE disabled via env")
        return False
    if not _on_gfx12x():
        if verbose:
            print("[w4a8_fp8_wmma] WNA16 MoE: not gfx12x, leaving on Triton")
        return False

    import w4a8_fp8_wmma  # noqa: F401
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method

    _orig_process = MoeWNA16Method.process_weights_after_loading
    _orig_apply = MoeWNA16Method.apply

    def _patched_process(self, layer):
        eng = False
        try:
            eng = (self.quant_config.weight_bits == 4
                   and _supported_group_size(layer.group_size))
        except Exception:
            eng = False
        self._w4a8_wna16 = eng
        # Always run the stock process first (it may finalize params); cheap.
        _orig_process(self, layer)
        if not eng:
            return
        has_zp = bool(getattr(self.quant_config, "has_zp", False))
        w13, w13_s, w13_z = _wna16_moe_to_op_layout(
            layer.w13_qweight.data, layer.w13_scales.data,
            layer.w13_qzeros.data if has_zp else None)
        w2, w2_s, w2_z = _wna16_moe_to_op_layout(
            layer.w2_qweight.data, layer.w2_scales.data,
            layer.w2_qzeros.data if has_zp else None)
        # views share storage with the (u8) params -> zero extra memory.
        layer._w4a8_w13, layer._w4a8_w2 = w13, w2
        layer._w4a8_w13s, layer._w4a8_w2s = w13_s, w2_s
        layer._w4a8_w13z, layer._w4a8_w2z = w13_z, w2_z
        global _WNA16_OVERRIDE_LOGGED
        if verbose and not _WNA16_OVERRIDE_LOGGED:
            _WNA16_OVERRIDE_LOGGED = True
            print("[w4a8_fp8_wmma] WNA16 MoE weights ready for grouped FP8-WMMA "
                  f"(int4 g={layer.group_size} has_zp={has_zp}); engages PER-BATCH "
                  "iff _moe_should_engage (crossover cache or "
                  "VLLM_ROCM_W4A8_FP8_WMMA_MOE_MIN_M) — else stock Triton fused_moe")

    def _patched_apply(self, layer, x, topk_weights, topk_ids,
                       shared_experts, shared_experts_input):
        if not getattr(self, "_w4a8_wna16", False):
            return _orig_apply(self, layer, x, topk_weights, topk_ids,
                               shared_experts, shared_experts_input)
        x2 = x.reshape(-1, x.shape[-1])
        # M-adaptive dispatch via the AOT crossover cache: engage our op only in
        # the proven-winning M-window for this exact (TP-sharded) expert shape;
        # otherwise fall back to stock (decode + un-tuned shapes stay >= stock).
        # Safe because this hook keeps the original uint8 weights (the int32
        # conversion is a zero-copy view), so _orig_apply still has what it needs.
        if not _moe_should_engage(x2.size(0), layer._w4a8_w13, layer._w4a8_w2,
                                  layer._w4a8_w13s, topk_ids, layer=layer):
            return _orig_apply(self, layer, x, topk_weights, topk_ids,
                               shared_experts, shared_experts_input)
        act = layer.activation
        if not isinstance(act, MoEActivation):
            act = MoEActivation.from_str(str(act))
        out = _run_grouped_moe(
            x2, layer._w4a8_w13, layer._w4a8_w2, layer._w4a8_w13s,
            layer._w4a8_w2s, layer._w4a8_w13z, layer._w4a8_w2z,
            topk_weights, topk_ids, act, layer.global_num_experts,
            layer.expert_map, layer.apply_router_weight_on_input,
            _moe_kernel(), out_dtype=x.dtype)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    MoeWNA16Method.process_weights_after_loading = _patched_process
    MoeWNA16Method.apply = _patched_apply

    _MOE_WNA16_REGISTERED = True
    if verbose:
        print("[w4a8_fp8_wmma] WNA16 MoE hook installed (gfx12x): int4 AWQ/GPTQ "
              "fallback experts -> grouped FP8-WMMA; Marlin untouched")
    return True
