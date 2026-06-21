# SPDX-License-Identifier: Apache-2.0
#
# RXF (Rotated eXtra Fast) runtime kernels: IQ4-NL 4-bit code (two per byte) +
# a plain fp16 scale per group of 32. Dequant: w = LUT[idx] * scale. 4.5 bpw.
#
# Group = 32 = 2 chained WMMA K-steps, so the activation can be quantized to int8
# and the matmul run as a WIDE blocked int8 dot — one scale per K-block, int32
# accumulate, one rescale per 32 — on the fast int8 cores (~2x fp16). Two
# linear/MoE paths ship here and are selected by a single config-level flag:
#   - W4A16 (default, act_dtype=fp16): dequant code*scale -> fp16, wide fp16 dot.
#     Correctness-safe; the served path until perplexity validates int8.
#   - W4A8  (opt-in,  act_dtype=int8): per-token int8 activation x int8 code,
#     K=32 blocked dot. The headline fast path.
#
# STANDALONE: this module has NO dependency on any other quant kernel and does
# NOT use the compiled ops.scaled_int8_quant — it bundles its own per-token int8
# activation quant. The only external deps are torch, triton, and the custom-op
# registration helper, so the kernel module can be lifted out on its own.
#
# A size-32 Hadamard rotation (FWHT-32) is applied offline to the weights and at
# runtime to the activation; H_hat*H_hat = I cancels in the matmul. It mixes
# outliers 2x harder than FWHT-16, paying back the coarser group.
#
# Mixed-precision protected experts (MoE only): a checkpoint may keep a small
# set of experts in full fp16/bf16. The MoE kernel takes a static per-expert
# format tag + slot table and a compact fp16 weight region; each workgroup
# resolves its expert's format from the tag (a uniform scalar branch) and takes
# the fp16-load path or the quantized path. Protected experts always run the
# fp16 activation (their weights aren't quantized), so the int8 path keeps the
# rotated fp16 activation around alongside the int8 one. The launch topology is
# one kernel with a worst-case grid, so CUDA graph capture/replay is unaffected.
import functools
import json
import math
import os
import re
from typing import Any

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

GROUP = 32

_RXF_LINEAR_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
}

_RXF_LINEAR_INT8_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,        # one group per K-block (forced; see launch)
    "GROUP_SIZE_M": 1,
}

_RXF_MOE_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8,
}

# Default IQ4-NL grid (integer-valued — the integers ARE the int8 weight
# operands for the W4A8 dot). Overridden by a config.json codebook via
# set_codebook() (e.g. a Lloyd-Max grid; must also be integer-valued for int8).
_NL_DEFAULT = [-127, -104, -83, -65, -49, -35, -22, -10,
                  1,   13,  25,  38,  53,  69,  89, 113]
_CODEBOOK_OVERRIDE: list[float] | None = None
_NL_TABLE_CACHE: dict[torch.device, torch.Tensor] = {}
_NL_TABLE_INT8_CACHE: dict[torch.device, torch.Tensor] = {}


def set_codebook(values: list[float] | None) -> None:
    """Install a custom 16-point dequant codebook (or None to keep IQ4-NL).

    The packed 4-bit indices are meaningless without the matching grid, so this
    MUST be the same codebook the checkpoint was quantized against.
    """
    global _CODEBOOK_OVERRIDE
    if values is not None and len(values) != 16:
        raise ValueError(
            f"codebook must have exactly 16 entries, got {len(values)}")
    new = list(values) if values is not None else None
    if new == _CODEBOOK_OVERRIDE:
        return                                    # unchanged, keep caches
    _CODEBOOK_OVERRIDE = new
    _NL_TABLE_CACHE.clear()
    _NL_TABLE_INT8_CACHE.clear()
    if new is not None:
        logger.info("RXF using custom codebook: %s", _CODEBOOK_OVERRIDE)


def _get_nl_table(device: torch.device) -> torch.Tensor:
    """fp16 codebook (for the W4A16 dequant path)."""
    if device not in _NL_TABLE_CACHE:
        _NL_TABLE_CACHE[device] = torch.tensor(
            _CODEBOOK_OVERRIDE or _NL_DEFAULT,
            dtype=torch.float16, device=device)
    return _NL_TABLE_CACHE[device]


def _get_nl_table_int8(device: torch.device) -> torch.Tensor:
    """int8 codebook (for the W4A8 int8 dot). Validates the grid is
    integer-valued and in int8 range — the int8 path is meaningless otherwise."""
    if device not in _NL_TABLE_INT8_CACHE:
        vals = _CODEBOOK_OVERRIDE or _NL_DEFAULT
        for v in vals:
            if float(v) != int(v) or not (-128 <= int(v) <= 127):
                raise ValueError(
                    "RXF int8 activation path requires an integer-valued "
                    f"codebook in int8 range; got {vals}. Use act_dtype=fp16 "
                    "or a (Lloyd-Max) integer grid.")
        _NL_TABLE_INT8_CACHE[device] = torch.tensor(
            [int(v) for v in vals], dtype=torch.int8, device=device)
    return _NL_TABLE_INT8_CACHE[device]


# Runtime activation dtype: "fp16" (W4A16, default) or "int8" (W4A8).
_ACT_DTYPE: str = "fp16"


def set_act_dtype(act_dtype: str | None) -> None:
    """Select the served activation path. 'int8' -> W4A8 wide blocked dot;
    anything else -> W4A16 fp16 dequant matmul (safe default)."""
    global _ACT_DTYPE
    new = "int8" if act_dtype == "int8" else "fp16"
    if new != _ACT_DTYPE:
        _ACT_DTYPE = new
        logger.info("RXF activation dtype: %s", new)


# =============================================================================
# Hadamard rotation (normalized Sylvester H_hat = (1/sqrt(32))*H32)
# =============================================================================
# The checkpoint stores weights pre-rotated per group of 32 along the input
# (contraction) dim. At runtime the SAME symmetric orthonormal H_hat is applied
# to the activation per consecutive 32 input channels BEFORE the matmul, so
# H_hat*H_hat = (1/32)*H32*H32 = I cancels exactly -> the matmul is unchanged.
#
# The rotation is a single standalone pass over the full [M,K] activation just
# before each GEMM, NOT inside the K-loop (in-GEMM re-runs the FWHT per K-block
# and over padded rows). Hoisted out, the GEMM is the plain dequant/int8 matmul.
#
# Protected fp16 experts: the activation reaching the MoE GEMM is rotated for
# ALL experts (one format-blind pass). fp16 expert weights are therefore stored
# pre-rotated along K too, so the rotations cancel identically on both paths.
#
# MANDATORY gate: must be on iff the checkpoint was quantized with rotation
# (config rotation=hadamard{S}), else the math is silently wrong. Set per-load
# from the single config-level flag, so linear and MoE can never diverge.
#
# _ROT_SPAN is the FWHT width S the checkpoint was rotated over (a power of two,
# a multiple of 32). The runtime applies the matching span so the offline
# weight-rotation and this activation-rotation cancel. 32 stays the default and
# reproduces the original fixed FWHT-32 bit-for-bit.
_APPLY_HADAMARD: bool = False
_ROT_SPAN: int = GROUP
# The model-wide learned Givens R [S,S] (rotation=givens{S}), or None. Set per
# load from config (set_givens_rotation). The linear path applies it explicitly
# (layer.rxf_givens_R); the MONOLITHIC MoE path applies it from this global
# inside invoke_rxf_moe_kernel (where the fixed-Hadamard FWHT used to run), since
# the experts call that kernel directly and have no per-layer hook.
_GIVENS_R: "torch.Tensor | None" = None

_ROT_RE = re.compile(r"^hadamard(\d+)$")
_GIVENS_RE = re.compile(r"^givens(\d+)$")


def set_givens_rotation(R: "torch.Tensor | None") -> None:
    """Install the model-wide learned Givens R [S,S] (or None) used by the MoE
    path. fp32, orthonormal. Idempotent per load."""
    global _GIVENS_R
    _GIVENS_R = R


def set_rotation(rotation: str | None) -> None:
    """Enable the per-span Hadamard activation rotation iff the checkpoint was
    quantized with it. rotation is the config flag:
      'hadamard{S}'  -> the fixed in-kernel FWHT (the existing path),
      'givens{S}'    -> a LEARNED per-module rotation; the in-kernel FWHT is
                        DISABLED here because the rotation is applied EXTERNALLY
                        in the quant method's apply() via invoke_rxf_givens_rotate
                        (per-layer R loaded from the checkpoint), and
      None/'off'     -> no rotation.
    S must be a power of two and a multiple of 32 (so every size-32 scale group
    lands inside one rotated block, and the FWHT span divides any power-of-two
    padded K). This setter only governs the IN-KERNEL Hadamard; the learned
    Givens path carries its rotation per-layer, not in this global."""
    global _APPLY_HADAMARD, _ROT_SPAN
    if rotation is None or rotation == "off":
        on, span = False, GROUP
    elif _GIVENS_RE.match(rotation):
        # Learned per-module rotation -> external; in-kernel FWHT stays OFF.
        span = int(_GIVENS_RE.match(rotation).group(1))
        if span < GROUP or (span & (span - 1)) != 0 or span % GROUP != 0:
            raise ValueError(
                f"RXF rotation span must be a power of two and a multiple of "
                f"{GROUP}, got {span}")
        on = False
    else:
        m = _ROT_RE.match(rotation)
        if m is None:
            raise ValueError(
                f"RXF unknown rotation {rotation!r}; expected 'hadamard<S>', "
                f"'givens<S>' (S a power of two, multiple of {GROUP}) or None")
        span = int(m.group(1))
        if span < GROUP or (span & (span - 1)) != 0 or span % GROUP != 0:
            raise ValueError(
                f"RXF rotation span must be a power of two and a multiple of "
                f"{GROUP}, got {span}")
        on = True
    if on != _APPLY_HADAMARD or span != _ROT_SPAN:
        _APPLY_HADAMARD = on
        _ROT_SPAN = span
        logger.info("RXF Hadamard activation rotation: %s (span=%d)",
                    "ON" if on else "off", span)


def invoke_rxf_givens_rotate(
    x: torch.Tensor, R: torch.Tensor,
) -> torch.Tensor:
    """Apply a LEARNED block-diagonal orthonormal rotation R [S,S] to a 2D
    [M, K] activation: rotate each consecutive S channels, v_block -> R @ v_block
    (== v_block @ R.T, row-vector form). K must be a multiple of S.

    The offline quantizer stored each weight row R-rotated the SAME way
    (quantize_rxf._apply_rotation_rows), so for the matmul x' W'ᵀ:
        (R x_block)·(R w_block) = x_blockᵀ RᵀR w_block = x_block·w_block
    cancels exactly (R orthonormal). Pure torch (a small S×S matmul per block) —
    CUDA-graph-safe and torch.compile-traceable, so it needs no custom op. fp32
    accumulate matches the offline fp32 rotate; cast back to x's dtype."""
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    M, K = x.shape
    S = R.shape[0]
    assert R.dim() == 2 and R.shape[1] == S, f"R must be square, got {tuple(R.shape)}"
    assert K % S == 0, f"K must be a multiple of rotation span {S}, got {K}"
    xr = x.reshape(M, K // S, S).to(torch.float32)
    xr = torch.matmul(xr, R.to(torch.float32).t())
    return xr.reshape(M, K).to(x.dtype)


@triton.jit
def _fwht_stage(x, ROWS: tl.constexpr, S: tl.constexpr,
                NG2H: tl.constexpr, H: tl.constexpr):
    # One FWHT butterfly stage over the size-S last axis: pair the first H with
    # the second H within each block of 2H. adds/subtracts only. The last axis is
    # the rotation span S (a power of two); NG2H = S // (2H).
    t = tl.reshape(x, (ROWS, NG2H, 2, H))
    t = tl.permute(t, (0, 1, 3, 2))          # (ROWS, NG2H, H, 2)
    lo, hi = tl.split(t)                     # each (ROWS, NG2H, H)
    t = tl.join(lo + hi, lo - hi)            # (ROWS, NG2H, H, 2)
    t = tl.permute(t, (0, 1, 3, 2))          # (ROWS, NG2H, 2, H)
    return tl.reshape(t, (ROWS, S))


@triton.jit
def _fwht(a, BM: tl.constexpr, BK: tl.constexpr, S: tl.constexpr):
    # Block-diagonal H_hat applied to a [BM, BK] activation tile: FWHT over each
    # consecutive S channels (BK is a multiple of S). log2(S) butterfly stages in
    # fp32 (H = 1,2,4,...,S/2; NG2H = S//(2H)), then *1/sqrt(S) to normalize.
    # span==32 uses the literal 0.1767766953 to reproduce the shipped FWHT-32
    # exactly. Ordering matches the offline _fwht_rows, so weight- and
    # activation-rotations cancel.
    NG: tl.constexpr = BK // S
    ROWS: tl.constexpr = BM * NG
    x = tl.reshape(a, (ROWS, S)).to(tl.float32)
    # log2(S) butterfly stages, written as explicit constexpr-GUARDED calls (the
    # only Triton-safe form: a `while` loop-carries H as a tensor, a Python `for`
    # over a tuple is rejected by the JIT parser, and a static_range can't rebind
    # a constexpr H). Each `if N < S` is constexpr (S constexpr) → dead stages are
    # pruned at compile time. Covers spans up to 2048 (H = 1..S/2).
    if 1 < S:    x = _fwht_stage(x, ROWS, S, S // 2,    1)
    if 2 < S:    x = _fwht_stage(x, ROWS, S, S // 4,    2)
    if 4 < S:    x = _fwht_stage(x, ROWS, S, S // 8,    4)
    if 8 < S:    x = _fwht_stage(x, ROWS, S, S // 16,   8)
    if 16 < S:   x = _fwht_stage(x, ROWS, S, S // 32,   16)
    if 32 < S:   x = _fwht_stage(x, ROWS, S, S // 64,   32)
    if 64 < S:   x = _fwht_stage(x, ROWS, S, S // 128,  64)
    if 128 < S:  x = _fwht_stage(x, ROWS, S, S // 256,  128)
    if 256 < S:  x = _fwht_stage(x, ROWS, S, S // 512,  256)
    if 512 < S:  x = _fwht_stage(x, ROWS, S, S // 1024, 512)
    if 1024 < S: x = _fwht_stage(x, ROWS, S, S // 2048, 1024)
    norm: tl.constexpr = 0.1767766953 if S == 32 else (1.0 / math.sqrt(S))
    x = x * norm
    return tl.reshape(x, (BM, BK)).to(a.dtype)


@triton.jit
def _hadamard_rotate_kernel(
    x_ptr, y_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_ym, stride_yk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    S: tl.constexpr,
):
    # Apply block-diagonal H_hat to an [M, K] activation: FWHT-S over each
    # consecutive S channels, once. BLOCK_K is a multiple of S and tile bounds
    # land on multiples of S (K is a multiple of S), so no S-group ever
    # straddles a tile boundary.
    tl.static_assert(BLOCK_K % S == 0, "BLOCK_K must be multiple of span S")
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    x = tl.load(
        x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
        mask=mask, other=0.0)
    x = _fwht(x, BLOCK_M, BLOCK_K, S)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_k[None, :] * stride_yk,
        x, mask=mask)


def invoke_hadamard32_rotate(x: torch.Tensor) -> torch.Tensor:
    """Rotate a 2D [M, K] activation by the block-diagonal SxS H_hat (one FWHT-S
    per consecutive S channels, S = the loaded rotation span), returning a new
    tensor of the same shape/dtype. K must be a multiple of S (guaranteed by the
    quant scheme)."""
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    M, K = x.shape
    S = _ROT_SPAN
    assert K % S == 0, f"K must be a multiple of span {S}, got {K}"
    y = torch.empty_like(x)
    BLOCK_M = 32
    # BLOCK_K must be a multiple of S (so a span never straddles a tile); 256
    # stays the default but widens to S when a wider span is in use.
    BLOCK_K = max(256, S)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    _hadamard_rotate_kernel[grid](
        x, y, M, K,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, S=S,
    )
    return y


# =============================================================================
# Fused FWHT-32 rotate + per-token int8 activation quant (one launch)
# =============================================================================
# float -> int8 truncates toward zero in Triton, so round explicitly (matches
# the offline weight quantizer). libdevice is platform-namespaced.
if current_platform.is_rocm():
    @triton.jit
    def _round_int8(x):
        return tl.extra.hip.libdevice.round(x).to(tl.int8)
elif current_platform.is_xpu():
    @triton.jit
    def _round_int8(x):
        return tl.extra.intel.libdevice.round(x).to(tl.int8)
else:
    @triton.jit
    def _round_int8(x):
        return tl.extra.cuda.libdevice.round(x).to(tl.int8)


@triton.jit
def _rxf_rotate_quant_int8_kernel(
    x_ptr, xq_ptr, scale_ptr,
    stride_xm, stride_qm,
    K,
    APPLY_ROT: tl.constexpr,
    BLOCK: tl.constexpr,        # next_pow2(K); a multiple of the span S
    S: tl.constexpr,            # FWHT rotation span (power of two, multiple of 32)
):
    # One program per token (row). Load the whole row, apply the block-diagonal
    # FWHT-S in registers (same stage ordering / 1/sqrt(S) norm as _fwht and the
    # offline _fwht_rows, so it cancels in the matmul), then a per-token
    # symmetric int8 quant. Fuses rotation + abs-max + quantize into a single [K]
    # read and an int8 [K] write — no fp32 materialization, no throwaway rotated
    # tensor. Padding lanes (BLOCK > K) load as 0 and form whole zero S-groups
    # (K % S == 0, BLOCK a power of two >= S), so they rotate to 0 and never
    # perturb a real group or the row max.
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row_id * stride_xm + cols,
                mask=mask, other=0.0).to(tl.float32)
    if APPLY_ROT:
        NG: tl.constexpr = BLOCK // S
        xr = tl.reshape(x, (NG, S))
        # log2(S) butterfly stages as explicit constexpr-guarded calls (see _fwht).
        if 1 < S:    xr = _fwht_stage(xr, NG, S, S // 2,    1)
        if 2 < S:    xr = _fwht_stage(xr, NG, S, S // 4,    2)
        if 4 < S:    xr = _fwht_stage(xr, NG, S, S // 8,    4)
        if 8 < S:    xr = _fwht_stage(xr, NG, S, S // 16,   8)
        if 16 < S:   xr = _fwht_stage(xr, NG, S, S // 32,   16)
        if 32 < S:   xr = _fwht_stage(xr, NG, S, S // 64,   32)
        if 64 < S:   xr = _fwht_stage(xr, NG, S, S // 128,  64)
        if 128 < S:  xr = _fwht_stage(xr, NG, S, S // 256,  128)
        if 256 < S:  xr = _fwht_stage(xr, NG, S, S // 512,  256)
        if 512 < S:  xr = _fwht_stage(xr, NG, S, S // 1024, 512)
        if 1024 < S: xr = _fwht_stage(xr, NG, S, S // 2048, 1024)
        norm: tl.constexpr = 0.1767766953 if S == 32 else (1.0 / math.sqrt(S))
        x = tl.reshape(xr * norm, (BLOCK,))
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-12)
    scale = absmax / 127.0
    q = _round_int8(x * (127.0 / absmax))
    tl.store(xq_ptr + row_id * stride_qm + cols, q, mask=mask)
    tl.store(scale_ptr + row_id, scale)


def invoke_rxf_rotate_quant_int8(
    x: torch.Tensor, apply_rot: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused FWHT-S rotation (when apply_rot, S = the loaded rotation span) +
    per-token int8 quant of a 2D [M, K] activation in one launch. Returns
    (q int8 [M, K], scale fp32 [M]). Replaces the separate rotate pass +
    eager-torch per-token quant: one [M,K] read, one int8 write, no fp32 copy, no
    throwaway rotated tensor. K must be a multiple of S when rotating (guaranteed
    by the quant scheme). BLOCK = next_pow2(K) >= K >= S, so S divides BLOCK and
    the in-register reshape to (BLOCK//S, S) is exact."""
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    M, K = x.shape
    S = _ROT_SPAN
    if apply_rot:
        assert K % S == 0, f"K must be a multiple of span {S} to rotate, got {K}"
    x = x.contiguous()
    q = torch.empty((M, K), device=x.device, dtype=torch.int8)
    scale = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(K)
    num_warps = min(max(BLOCK // 256, 1), 8)
    _rxf_rotate_quant_int8_kernel[(M,)](
        x, q, scale,
        x.stride(0), q.stride(0),
        K,
        APPLY_ROT=apply_rot,
        BLOCK=BLOCK,
        S=S,
        num_warps=num_warps, num_stages=1,
    )
    return q, scale


@functools.lru_cache
def get_rxf_configs(
    N: int, K: int, dtype_tag: str = "rxf"
) -> dict[int, Any] | None:
    # The rotation is applied outside the GEMM (invoke_hadamard32_rotate), so the
    # GEMM is the plain matmul whether or not the checkpoint is rotated. Rotated
    # and non-rotated models therefore share the same configs.
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},"
        f"dtype={dtype_tag},group_size=32.json"
    )
    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name
    )
    if os.path.exists(config_file_path):
        with open(config_file_path, encoding="utf-8") as f:
            logger.info("Using tuned config from %s for RXF kernel.",
                        config_file_path)
            return {int(key): val for key, val in json.load(f).items()}
    logger.warning(
        "Using default RXF kernel config. Performance might be "
        "sub-optimal! Config file not found at %s",
        config_file_path,
    )
    return None


def _pick_config(
    M: int, tuned: dict[int, Any] | None, default: dict
) -> dict:
    if tuned is None:
        return default
    if M in tuned:
        return tuned[M]
    closest = min(tuned.keys(), key=lambda k: abs(k - M))
    return tuned[closest]


def _constrain_default_block_m(M: int, config: dict) -> dict:
    """Shrink BLOCK_SIZE_M for small M when falling back to the DEFAULT config
    (no tuned file for this N,K). The default is sized for large batches; at
    decode M=1 that pads one real row across a 128-row tile. The new tile is a
    power of two floored at 16 (valid WMMA M) and never raised above default."""
    bm = config["BLOCK_SIZE_M"]
    if M >= bm:
        return config
    new_bm = max(16, triton.next_power_of_2(M))
    if new_bm >= bm:
        return config
    out = dict(config)
    out["BLOCK_SIZE_M"] = new_bm
    return out


# =============================================================================
# Linear — W4A16 fallback (dequant code*fp16-scale -> wide fp16 dot)
# =============================================================================
@triton.jit
def rxf_linear_kernel(
    a_ptr,
    b_packed_ptr,
    c_ptr,
    b_scale_ptr,
    nl_table_ptr,
    bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sn, stride_sk,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K % 32 == 0,
                     "BLOCK_SIZE_K must be multiple of 32")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_k_scale = tl.arange(0, BLOCK_SIZE_K // 32)

    a_ptrs = (a_ptr
              + offs_am[:, None] * stride_am
              + offs_k[None, :] * stride_ak)
    b_packed_ptrs = (b_packed_ptr
                     + offs_bn[:, None] * stride_bn
                     + offs_k_packed[None, :] * stride_bk)
    b_scale_ptrs = (b_scale_ptr
                    + offs_bn[:, None] * stride_sn
                    + offs_k_scale[None, :] * stride_sk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_full = K // BLOCK_SIZE_K
    for _k in range(0, num_k_full):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M), other=0.0)
        b_packed = tl.load(b_packed_ptrs)
        b_scale = tl.load(b_scale_ptrs).to(tl.float32)       # fp16 -> f32

        lo = (b_packed & 0x0F).to(tl.uint16)
        hi = (b_packed >> 4).to(tl.uint16)
        idx = tl.interleave(lo, hi)
        nl_val = tl.load(nl_table_ptr + idx.to(tl.int32))

        b_scale_3d = b_scale[:, :, None]
        b_scale_3d = tl.broadcast_to(
            b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K // 32, 32))
        b_scale_full = tl.reshape(b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K))

        b = tl.trans((nl_val * b_scale_full).to(compute_type))
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_packed_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        b_scale_ptrs += (BLOCK_SIZE_K // 32) * stride_sk

    tail_remaining = K - num_k_full * BLOCK_SIZE_K
    if tail_remaining > 0:
        k_mask = offs_k < tail_remaining
        a = tl.load(
            a_ptrs, mask=(offs_am[:, None] < M) & k_mask[None, :], other=0.0)

        k_packed_mask = offs_k_packed < ((tail_remaining + 1) // 2)
        b_packed = tl.load(b_packed_ptrs, mask=k_packed_mask[None, :], other=0)

        k_scale_mask = offs_k_scale < tl.cdiv(tail_remaining, 32)
        b_scale = tl.load(b_scale_ptrs, mask=k_scale_mask[None, :],
                          other=0.0).to(tl.float32)

        lo = (b_packed & 0x0F).to(tl.uint16)
        hi = (b_packed >> 4).to(tl.uint16)
        idx = tl.interleave(lo, hi)
        nl_val = tl.load(nl_table_ptr + idx.to(tl.int32))

        b_scale_3d = b_scale[:, :, None]
        b_scale_3d = tl.broadcast_to(
            b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K // 32, 32))
        b_scale_full = tl.reshape(b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K))

        b = tl.trans((nl_val * b_scale_full).to(compute_type))
        b = tl.where(k_mask[:, None], b, 0.0)
        accumulator += tl.dot(a, b)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator += bias[None, :]

    result = accumulator.to(compute_type)
    c_ptrs = (c_ptr
              + offs_am[:, None] * stride_cm
              + offs_cn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


# =============================================================================
# Linear — W4A8 fast path (per-token int8 activation x int8 code, K=32 dot)
# =============================================================================
@triton.jit
def rxf_linear_int8_kernel(
    a_ptr,                 # int8 [M, K]
    b_packed_ptr,          # uint8 [N, K//2]
    c_ptr,
    b_scale_ptr,           # fp16 [N, K//32]  per-group weight scale
    a_scale_ptr,           # fp32 [M]         per-token activation scale
    nl_int8_ptr,           # int8 [16]        integer codebook
    bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_sn, stride_sk,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,    # forced to 32: one group per K-block
    GROUP_SIZE_M: tl.constexpr,
    compute_type: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K == 32,
                     "int8 kernel uses one group (32) per K-block")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 2)

    a_ptrs = (a_ptr
              + offs_am[:, None] * stride_am
              + offs_k[None, :] * stride_ak)
    b_packed_ptrs = (b_packed_ptr
                     + offs_bn[:, None] * stride_bn
                     + offs_k_packed[None, :] * stride_bk)
    # one fp16 scale per (n, group); the group index advances once per K-block.
    b_scale_ptrs = b_scale_ptr + offs_bn * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_full = K // BLOCK_SIZE_K    # = K // 32 groups
    for _k in range(0, num_k_full):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M), other=0)   # int8
        b_packed = tl.load(b_packed_ptrs)

        lo = (b_packed & 0x0F).to(tl.uint16)
        hi = (b_packed >> 4).to(tl.uint16)
        idx = tl.interleave(lo, hi)
        code = tl.load(nl_int8_ptr + idx.to(tl.int32))      # int8 [BN, 32]

        ws = tl.load(b_scale_ptrs).to(tl.float32)           # [BN] this group

        # WIDE int8 dot: int8[BM,32] x int8[32,BN] -> int32[BM,BN], then rescale
        # by this group's fp16 weight scale into the fp32 accumulator.
        p = tl.dot(a, tl.trans(code), out_dtype=tl.int32)
        accumulator += p.to(tl.float32) * ws[None, :]

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_packed_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        b_scale_ptrs += stride_sk

    # Per-token activation scale, constant over K -> applied once at the end.
    a_scale = tl.load(a_scale_ptr + offs_am, mask=offs_am < M, other=0.0)
    accumulator = accumulator * a_scale[:, None]

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0)
        accumulator += bias[None, :]

    result = accumulator.to(compute_type)
    c_ptrs = (c_ptr
              + offs_am[:, None] * stride_cm
              + offs_cn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


def _rxf_linear_launch_fp16(
    A, B_packed, B_scale, bias, has_bias, config,
) -> torch.Tensor:
    M, K = A.shape
    N = B_packed.shape[0]
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    nl_table = _get_nl_table(A.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    block_k = config.get("BLOCK_SIZE_K", 32)
    if block_k % 32 != 0:
        block_k = ((block_k + 31) // 32) * 32

    extra_kwargs = {}
    for k in ("num_warps", "num_stages", "waves_per_eu",
              "matrix_instr_nonkdim", "kpack"):
        if k in config:
            extra_kwargs[k] = config[k]

    rxf_linear_kernel[grid](
        A, B_packed, C, B_scale, nl_table,
        bias if has_bias else C,
        M, N, K,
        A.stride(0), A.stride(1),
        B_packed.stride(0), B_packed.stride(1),
        C.stride(0), C.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 1),
        compute_type=compute_type,
        **extra_kwargs,
    )
    return C


def _rxf_linear_launch_int8(
    A, B_packed, B_scale, bias, has_bias, config, apply_rot,
) -> torch.Tensor:
    """A is the RAW fp16/bf16 activation; fused-rotated (when apply_rot) and
    per-token int8-quantized here in a single launch."""
    M, K = A.shape
    N = B_packed.shape[0]
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16
    q, a_scale = invoke_rxf_rotate_quant_int8(A, apply_rot)
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    nl_int8 = _get_nl_table_int8(A.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    extra_kwargs = {}
    for k in ("num_warps", "num_stages", "waves_per_eu",
              "matrix_instr_nonkdim", "kpack"):
        if k in config:
            extra_kwargs[k] = config[k]

    rxf_linear_int8_kernel[grid](
        q, B_packed, C, B_scale, a_scale, nl_int8,
        bias if has_bias else C,
        M, N, K,
        q.stride(0), q.stride(1),
        B_packed.stride(0), B_packed.stride(1),
        C.stride(0), C.stride(1),
        B_scale.stride(0), B_scale.stride(1),
        HAS_BIAS=has_bias,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 1),
        compute_type=compute_type,
        **extra_kwargs,
    )
    return C


def _rxf_linear_func(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_scale: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
) -> torch.Tensor:
    M = A.shape[0]
    N = B_packed.shape[0]
    K = B_packed.shape[1] * 2
    # Weights were stored pre-rotated; the activation rotation cancels them in the
    # plain matmul below. int8 fuses rotate+quant into one launch (raw A in);
    # fp16 takes the standalone rotate pre-pass. No-op when not rotated.
    if _ACT_DTYPE == "int8":
        tuned = get_rxf_configs(N, K, "rxf_int8")
        config = _pick_config(M, tuned, _RXF_LINEAR_INT8_DEFAULT_CONFIG)
        if tuned is None:
            config = _constrain_default_block_m(M, config)
        return _rxf_linear_launch_int8(A, B_packed, B_scale, bias, has_bias,
                                       config, _APPLY_HADAMARD)
    if _APPLY_HADAMARD:
        A = invoke_hadamard32_rotate(A)
    tuned = get_rxf_configs(N, K, "rxf")
    config = _pick_config(M, tuned, _RXF_LINEAR_DEFAULT_CONFIG)
    if tuned is None:
        config = _constrain_default_block_m(M, config)
    return _rxf_linear_launch_fp16(A, B_packed, B_scale, bias, has_bias, config)


def _rxf_linear_fake(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_scale: torch.Tensor,
    bias: torch.Tensor,
    has_bias: bool,
) -> torch.Tensor:
    return torch.empty(
        (A.size(0), B_packed.size(0)), dtype=A.dtype, device=A.device)


direct_register_custom_op(
    "rxf_linear_func",
    _rxf_linear_func,
    fake_impl=_rxf_linear_fake,
)


def invoke_rxf_linear_kernel(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    B_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    dummy = A
    return torch.ops.vllm.rxf_linear_func(
        A, B_packed, B_scale,
        bias if bias is not None else dummy,
        bias is not None,
    )


# =============================================================================
# Fused MoE
# =============================================================================
@triton.jit
def _moe_fp16_dequant_acc(
    a_ptrs, b_packed_ptrs, b_scale_ptrs, nl_table_ptr,
    token_mask, K,
    stride_ak, stride_bpk, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    # W4A16 dequant K-loop: unpack two NL indices per byte, scale = fp16 per
    # group of 32, wide fp16 dot. No exponent / no int8.
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 2)
    offs_k_scale = tl.arange(0, BLOCK_SIZE_K // 32)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_full = K // BLOCK_SIZE_K
    for _k in range(0, num_k_full):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b_packed = tl.load(b_packed_ptrs)
        b_scale = tl.load(b_scale_ptrs).to(tl.float32)

        lo = (b_packed & 0x0F).to(tl.uint16)
        hi = (b_packed >> 4).to(tl.uint16)
        idx = tl.interleave(lo, hi)
        nl_val = tl.load(nl_table_ptr + idx.to(tl.int32))

        b_scale_3d = b_scale[:, :, None]
        b_scale_3d = tl.broadcast_to(
            b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K // 32, 32))
        b_scale_full = tl.reshape(b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K))

        b = tl.trans((nl_val * b_scale_full).to(compute_type))
        accumulator += tl.dot(a, b)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_packed_ptrs += (BLOCK_SIZE_K // 2) * stride_bpk
        b_scale_ptrs += (BLOCK_SIZE_K // 32) * stride_bsk

    tail_remaining = K - num_k_full * BLOCK_SIZE_K
    if tail_remaining > 0:
        k_mask = offs_k < tail_remaining
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0)
        k_packed_mask = offs_k_packed < ((tail_remaining + 1) // 2)
        b_packed = tl.load(b_packed_ptrs, mask=k_packed_mask[None, :], other=0)
        k_scale_mask = offs_k_scale < tl.cdiv(tail_remaining, 32)
        b_scale = tl.load(b_scale_ptrs, mask=k_scale_mask[None, :],
                          other=0.0).to(tl.float32)

        lo = (b_packed & 0x0F).to(tl.uint16)
        hi = (b_packed >> 4).to(tl.uint16)
        idx = tl.interleave(lo, hi)
        nl_val = tl.load(nl_table_ptr + idx.to(tl.int32))

        b_scale_3d = b_scale[:, :, None]
        b_scale_3d = tl.broadcast_to(
            b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K // 32, 32))
        b_scale_full = tl.reshape(b_scale_3d, (BLOCK_SIZE_N, BLOCK_SIZE_K))

        b = tl.trans((nl_val * b_scale_full).to(compute_type))
        b = tl.where(k_mask[:, None], b, 0.0)
        accumulator += tl.dot(a, b)

    return accumulator


@triton.jit
def _moe_int8_acc(
    a_ptrs, b_packed_ptrs, b_scale_ptrs, nl_int8_ptr,
    token_mask, K,
    stride_ak, stride_bpk, stride_bsk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,    # forced 32
    compute_type: tl.constexpr,
):
    # W4A8 K-loop: one group (32) per step, int8[BM,32] x int8[32,BN] -> int32,
    # rescaled by this group's fp16 weight scale into the fp32 accumulator. The
    # per-token activation scale is applied by the caller after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k_full = K // BLOCK_SIZE_K
    for _k in range(0, num_k_full):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0)       # int8
        b_packed = tl.load(b_packed_ptrs)

        lo = (b_packed & 0x0F).to(tl.uint16)
        hi = (b_packed >> 4).to(tl.uint16)
        idx = tl.interleave(lo, hi)
        code = tl.load(nl_int8_ptr + idx.to(tl.int32))               # int8

        ws = tl.load(b_scale_ptrs).to(tl.float32)                    # [BN, 1]
        p = tl.dot(a, tl.trans(code), out_dtype=tl.int32)
        accumulator += p.to(tl.float32) * tl.trans(ws)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_packed_ptrs += (BLOCK_SIZE_K // 2) * stride_bpk
        b_scale_ptrs += stride_bsk
    return accumulator


@triton.jit
def _moe_fp16_acc(
    a_ptrs, w_ptrs, token_mask, K,
    stride_ak, stride_fk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    compute_type: tl.constexpr,
):
    # Protected-expert K-loop: plain dense load of the expert's fp16/bf16 tile,
    # no dequant. Weights are pre-rotated when rotation is on. Always uses the
    # fp16 activation (protected weights are not quantized).
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    num_k_full = K // BLOCK_SIZE_K
    for _k in range(0, num_k_full):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        w = tl.load(w_ptrs)
        accumulator += tl.dot(a, tl.trans(w.to(compute_type)))
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += BLOCK_SIZE_K * stride_fk

    tail_remaining = K - num_k_full * BLOCK_SIZE_K
    if tail_remaining > 0:
        k_mask = offs_k < tail_remaining
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :],
                    other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
        accumulator += tl.dot(a, tl.trans(w.to(compute_type)))

    return accumulator


@triton.jit
def rxf_moe_kernel(
    a_ptr,                 # fp16/bf16 [M, K] rotated activation (fp16 path)
    a_int8_ptr,            # int8 [M, K] per-token-quantized (int8 path)
    a_scale_ptr,           # fp32 [M] per-token activation scale (int8 path)
    b_packed_ptr,
    c_ptr,
    b_scale_ptr,           # fp16 [E, N, K//32]
    nl_table_ptr,          # fp16 codebook (W4A16)
    nl_int8_ptr,           # int8 codebook (W4A8)
    format_tag_ptr,
    fp16_slot_ptr,
    b_fp16_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_a8m, stride_a8k,
    stride_bpe, stride_bpn, stride_bpk,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsk,
    stride_fe, stride_fn, stride_fk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    HAS_FP16: tl.constexpr,
    ACT_INT8: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_K % 32 == 0,
                     "BLOCK_SIZE_K must be multiple of 32")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    if off_experts == -1:
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (c_ptr + offs_token[:, None] * stride_cm
                  + offs_cn[None, :] * stride_cn)
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs,
                 tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type),
                 mask=c_mask)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N
               + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_am_k = tl.arange(0, BLOCK_SIZE_K)
    a_row = offs_token[:, None] // top_k

    # Per-expert format dispatch: the tag is one scalar for the whole expert
    # tile (uniform branch, no divergence). HAS_FP16=False prunes the fp16 path.
    is_fp16 = False
    if HAS_FP16:
        is_fp16 = tl.load(format_tag_ptr + off_experts) != 0

    if is_fp16:
        # Protected expert: dense fp16 weights x rotated fp16 activation.
        a_ptrs = a_ptr + (a_row * stride_am + offs_am_k[None, :] * stride_ak)
        slot = tl.load(fp16_slot_ptr + off_experts).to(tl.int64)
        w_ptrs = (b_fp16_ptr + slot * stride_fe
                  + offs_bn[:, None] * stride_fn
                  + offs_am_k[None, :] * stride_fk)
        accumulator = _moe_fp16_acc(
            a_ptrs, w_ptrs, token_mask, K,
            stride_ak, stride_fk,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type)
    else:
        offs_bk_packed = tl.arange(0, BLOCK_SIZE_K // 2)
        b_packed_ptrs = (b_packed_ptr + off_experts * stride_bpe
                         + offs_bn[:, None] * stride_bpn
                         + offs_bk_packed[None, :] * stride_bpk)
        if ACT_INT8:
            offs_bk_scale = tl.arange(0, BLOCK_SIZE_K // 32)
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse
                            + offs_bn[:, None] * stride_bsn
                            + offs_bk_scale[None, :] * stride_bsk)
            # Distinct name from the bf16 a_ptrs in the is_fp16 / fp16-dequant
            # branches: Triton unifies a variable's type across all branches, so
            # reusing 'a_ptrs' here (int8) collided with the bf16 pointer
            # (Mismatched type then/else) whenever HAS_FP16 and ACT_INT8 are both
            # on — i.e. W4A8 with protected fp16 experts.
            a_ptrs_i8 = (a_int8_ptr
                         + a_row * stride_a8m + offs_am_k[None, :] * stride_a8k)
            accumulator = _moe_int8_acc(
                a_ptrs_i8, b_packed_ptrs, b_scale_ptrs, nl_int8_ptr,
                token_mask, K,
                stride_a8k, stride_bpk, stride_bsk,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type)
            a_scale = tl.load(a_scale_ptr + offs_token // top_k,
                              mask=token_mask, other=0.0)
            accumulator = accumulator * a_scale[:, None]
        else:
            offs_bk_scale = tl.arange(0, BLOCK_SIZE_K // 32)
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse
                            + offs_bn[:, None] * stride_bsn
                            + offs_bk_scale[None, :] * stride_bsk)
            a_ptrs = (a_ptr + a_row * stride_am + offs_am_k[None, :] * stride_ak)
            accumulator = _moe_fp16_dequant_acc(
                a_ptrs, b_packed_ptrs, b_scale_ptrs, nl_table_ptr,
                token_mask, K,
                stride_ak, stride_bpk, stride_bsk,
                BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, compute_type)

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = (c_ptr + stride_cm * offs_token[:, None]
              + stride_cn * offs_cn[None, :])
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_rxf_moe_kernel(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    format_tag: torch.Tensor | None,
    fp16_slot: torch.Tensor | None,
    B_fp16: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict,
    compute_type: "tl.dtype",
) -> None:
    assert B_packed.dtype == torch.uint8
    assert B_scale.dtype == torch.float16     # plain fp16 per-group scale
    assert topk_weights is not None or not mul_routed_weight

    has_fp16 = B_fp16 is not None
    if has_fp16:
        assert format_tag is not None and fp16_slot is not None
        assert B_fp16.dtype == A.dtype
        assert format_tag.dtype == torch.uint8
        assert fp16_slot.dtype == torch.int32
        ft, sl, bf = format_tag, fp16_slot, B_fp16
        stride_fe, stride_fn, stride_fk = (
            B_fp16.stride(0), B_fp16.stride(1), B_fp16.stride(2))
    else:
        ft, sl, bf = expert_ids, expert_ids, B_packed
        stride_fe = stride_fn = stride_fk = 0

    # Weights were stored pre-rotated; the activation rotation (covering ALL
    # experts, including protected ones) cancels them in the matmul. Two rotation
    # families: the fixed in-kernel FWHT (_APPLY_HADAMARD), or the learned
    # model-wide Givens R (_GIVENS_R) applied here as an external block-diagonal
    # matmul — they are mutually exclusive (set_rotation turns the FWHT off for
    # givens). The Givens form can't fuse into the FWHT+quant kernel, so it
    # materializes the rotated activation first, then quantizes with apply_rot=False.
    use_givens = _GIVENS_R is not None
    if use_givens:
        A = invoke_rxf_givens_rotate(A, _GIVENS_R)
    act_int8 = _ACT_DTYPE == "int8"
    if act_int8:
        if has_fp16:
            # Protected fp16 experts read the rotated fp16 activation (a_ptr), so
            # it must be materialized; quant then runs from it (no re-rotate).
            if _APPLY_HADAMARD:
                A = invoke_hadamard32_rotate(A)
            a_int8, a_scale = invoke_rxf_rotate_quant_int8(A, apply_rot=False)
        else:
            # No fp16 experts: a_ptr is never dereferenced (HAS_FP16 prunes it),
            # so skip materializing the rotated fp16 — fuse rotate+quant in one
            # launch and pass the raw A as the (unused) fp16 dummy. (Givens already
            # rotated A above, so apply_rot is False in that case.)
            a_int8, a_scale = invoke_rxf_rotate_quant_int8(
                A, _APPLY_HADAMARD and not use_givens)
    else:
        if _APPLY_HADAMARD:
            A = invoke_hadamard32_rotate(A)
        # Dummies, never dereferenced (ACT_INT8=False prunes the int8 path).
        a_int8, a_scale = A, A

    nl_table = _get_nl_table(A.device)
    nl_int8 = (_get_nl_table_int8(A.device) if act_int8
               else nl_table)             # dummy when fp16

    M = A.size(0)
    _, N, packed_K = B_packed.shape
    K = packed_K * 2

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        EM = min(sorted_token_ids.size(0),
                 A.size(0) * top_k * config["BLOCK_SIZE_M"])

    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    block_k = config.get("BLOCK_SIZE_K", 32)
    if block_k % 32 != 0:
        block_k = ((block_k + 31) // 32) * 32
    if act_int8:
        block_k = 32                      # int8 path: one group per K-block

    extra_kwargs = {}
    for k in ("num_warps", "num_stages", "waves_per_eu",
              "matrix_instr_nonkdim", "kpack"):
        if k in config:
            extra_kwargs[k] = config[k]

    rxf_moe_kernel[grid](
        A, a_int8, a_scale,
        B_packed, C, B_scale,
        nl_table, nl_int8,
        ft, sl, bf,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, M * top_k,
        A.stride(0), A.stride(1),
        a_int8.stride(0), a_int8.stride(1),
        B_packed.stride(0), B_packed.stride(1), B_packed.stride(2),
        C.stride(-2), C.stride(-1),
        B_scale.stride(0), B_scale.stride(1), B_scale.stride(2),
        stride_fe, stride_fn, stride_fk,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=config.get("GROUP_SIZE_M", 8),
        HAS_FP16=has_fp16,
        ACT_INT8=act_int8,
    )
