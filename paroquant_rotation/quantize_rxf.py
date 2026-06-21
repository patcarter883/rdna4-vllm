"""
RXF (Rotated eXtra Fast) quantizer — IQ4-NL grid + plain fp16 group-32 scale.

  python quantize_rxf.py --input <src> --output <dst>
                         [--ignore RE]... [--layers a,b]
                         [--fp16-experts SPEC]
                         [--act-dtype int8|fp16]
                         [--dry-run] [--gpu]

Per quantized module (weight [N, K]):
  - weight_packed   uint8   [N, K//2]   2 x 4-bit NL indices per byte
  - weight_scale    float16 [N, K//32]  one fp16 scale per group of 32
  Dequant: NL[idx] * weight_scale  (no mantissa/exponent — plain fp16 scalar).
  - 4.5 bits per weight (4-bit code + fp16 scale / 32)
  - Zero forced-zero weights (NL grid has no zero point)
  - Scale search is BOTH-POLARITY (mxfp4_16-style): the fp16 scale sign carries
    the group's polarity (the asymmetric NL grid or its mirror).
  - Group = 32 = 2 chained WMMA K-steps, so the runtime can run a wide int8
    blocked dot (one scale per K-block) on the fast cores. The coarser group is
    paid back by a size-32 Hadamard rotation (FWHT-32) that mixes outliers 2x
    harder than FWHT-16. See test_roundtrip.py.

Protected fp16 experts (--fp16-experts "12:5,12:17" or a JSON file
{"12": [5, 17]}): the listed MoE experts are NOT quantized. Each protected
module emits one tensor <module>.weight_fp16 in the source dtype, pre-rotated
along K when rotation is on (the runtime rotates activations format-blind
before every MoE GEMM). config.json quantization_config gains "fp16_experts".

Output: <input_name>-RXF
Format: rxf-pack-quantized

NOTE — activation-aware calibration (ParoQuant stage c) is ENABLED.
  `--act-aware <calib>` with `--rotation-kind givens` fits the learned rotation R
  to minimize the REAL per-token int8 ACTIVATION-quant error on the calibration
  activations (activation conditioning), the objective stage (b)'s weight-MSE fit
  could not reach (stage b was a PPL null — Hadamard already near-optimal for
  weight incoherence). Gated behind ACT_AWARE_ENABLED; collect_activations() runs
  the calibration forward pass and also pools activation blocks for the R fit.
"""
import argparse
import gc
import json
import math
import queue
import re
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from model_registry import get_spec, KIND_EXPERT, KIND_LINEAR, KIND_ATTN

GROUP = 32
SHARD_BYTES = 5 * 1024**3

# Nonzero groups whose exact scale rounds to 0 or a subnormal in float16 are
# "underflowed": the plain fp16 per-group scale can't represent them, so they
# dequant to ~0. RXF has no mantissa/exponent fallback, so this is the quality
# warning (and the signal that an expert may deserve --fp16-experts protection).
FP16_SCALE_UNDERFLOW = 6.103515625e-5   # smallest normal float16

NL_TABLE = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113],
    dtype=torch.float16)
NL_MAX = torch.tensor(127.0, dtype=torch.float16)

# Lloyd-Max codebook fit to Step-3.7-Flash MoE experts (sub-Gaussian, near
# symmetric). Pooled over layers 3..44 x {gate,up,down}; cuts per-group MSE
# ~22% vs IQ4-NL. Near-uniform, denser mid-range, sparser tails.
LLOYD_TABLE = torch.tensor(
    [-125, -102, -83, -66, -51, -36, -23, -9,
        4,   18,  32,  46,  62,  80, 100, 124],
    dtype=torch.float16)

CODEBOOKS = {"nl": NL_TABLE, "lloyd": LLOYD_TABLE}

# Active codebook — set by main() from --codebook, defaults to IQ4-NL.
ACTIVE_TABLE = NL_TABLE
ACTIVE_NAME = "iq4_nl"

# Activation-aware calibration master switch. When True, --act-aware runs the
# calibration forward pass (collect_activations). Its PRIMARY consumer is the
# LEARNED-Givens stage (c): the rotation R is fit to minimize the REAL per-token
# int8 activation-quant error on calibration activations (activation conditioning,
# the DFlash insight), NOT the weight-quant MSE that stage (b) minimized — stage
# (b) was a measured PPL null precisely because Hadamard is already near-optimal
# for weight incoherence. With it False, --act-aware prints a notice and runs the
# naive path. See fit_givens_rotation(score="activation") + main().
ACT_AWARE_ENABLED = True


def set_codebook(name, table):
    global ACTIVE_TABLE, ACTIVE_NAME
    ACTIVE_TABLE = torch.as_tensor(table, dtype=torch.float16)
    ACTIVE_NAME = name
    if ACTIVE_TABLE.numel() != 16:
        raise ValueError(f"codebook must have exactly 16 entries, "
                         f"got {ACTIVE_TABLE.numel()}")


def resolve_codebook(arg):
    """arg: 'nl' | 'lloyd' | path to JSON list of 16 numbers -> (name, table)."""
    if arg in CODEBOOKS:
        return arg if arg == "lloyd" else "iq4_nl", CODEBOOKS[arg]
    path = Path(arg)
    if not path.exists():
        raise ValueError(f"--codebook: unknown preset and not a file: {arg}")
    vals = json.load(open(path))
    if isinstance(vals, dict):
        vals = vals.get("codebook", vals)
    return f"custom:{path.stem}", torch.tensor(vals, dtype=torch.float16)


# ---------------------------------------------------------------------------
# Hadamard rotation (normalized Sylvester H_hat = (1/sqrt(S))*H_S)
# ---------------------------------------------------------------------------
# A per-SPAN orthonormal rotation applied to each consecutive block of S input
# channels BEFORE scale search + codebook assignment. Gaussianizes the block,
# tightening the scale and the IQ4-NL fit. The runtime kernel applies the same
# H_hat to the activation, so H_hat*H_hat = I cancels it out of the matmul ->
# free. A WIDER span mixes outliers harder and — unlike span-32 — can move
# outlier energy ACROSS the size-32 scale-group boundaries (QuaRot/SpinQuant
# rotate the full hidden dim for exactly this reason). The K=32 int8 GEMM is
# UNTOUCHED: the span is only the pre-pass rotation width; 32 stays the scale-
# group granularity of the (already-rotated) weight. See PAROQUANT_RXF_INTEGRATION.md.
#
# ParoQuant staging (all pre-pass only, GEMM untouched):
#   (a) WIDER FIXED span S in {32,64,128,...} — implemented here (offline side).
#   (b) LEARNED rotation matrix — stored in the checkpoint, loaded at runtime.   [TODO]
#   (c) IMPORTANCE-AWARE learned rotation, learned jointly with calibration —    [TODO]
#       re-enables ACT_AWARE_ENABLED (importance transformed INTO the rotated basis).
ROTATION_NAME = "hadamard32"          # config-json tag; widened to f"hadamard{S}"
APPLY_ROTATION = False
ROTATION_SPAN = GROUP                 # FWHT width S (power of two, multiple of GROUP)
# Rotation KIND: "hadamard" = the fixed data-blind Sylvester FWHT (stages a/—,
# in-kernel at runtime), or "givens" = a LEARNED per-module orthonormal R (stage
# b/c). Givens R is a span x span matrix fit to minimize the (importance-weighted)
# post-quant weight MSE; it is stored per-module as <module>.weight_rotation and
# applied at runtime as an EXTERNAL block-diagonal matmul (not the in-kernel FWHT).
# The cancellation is identical: weight stored R-rotated, activation R-rotated,
# R orthonormal => (R x)·(R w) = x·w. See PAROQUANT_RXF_INTEGRATION.md §2(b/c).
ROTATION_KIND = "hadamard"
# The fitted shared Givens R [span, span] (orthonormal), set ONCE before the
# per-module quant loop and reused for every module — a single model-wide rotation
# (QuaRot/SpinQuant style). Shared so merged linears (q/k/v, gate/up) and TP shards
# all apply the SAME activation rotation the weights were rotated by; block-diagonal
# over span so it serves any K (dense + MoE) uniformly. Written into config.json.
GIVENS_R = None


def set_givens_rotation(R):
    """Install the model-wide learned Givens rotation R [span, span] (or None)."""
    global GIVENS_R
    GIVENS_R = R


def set_rotation(on, span=GROUP, kind="hadamard"):
    """Enable rotation and fix its span S + kind. S must be a power of two and a
    multiple of GROUP (=32) so each size-32 scale group lands inside one rotated
    block — the cancellation and the K=32 GEMM are then both span-agnostic.
    kind="hadamard" tags config rotation=hadamard{S} (the fixed FWHT); kind="givens"
    tags rotation=givens{S} (a learned per-module R emitted into the checkpoint)."""
    global APPLY_ROTATION, ROTATION_SPAN, ROTATION_NAME, ROTATION_KIND
    APPLY_ROTATION = bool(on)
    if on:
        span = int(span)
        if span < GROUP or span & (span - 1) != 0 or span % GROUP != 0:
            raise ValueError(
                f"rotation span must be a power of two and a multiple of "
                f"{GROUP}, got {span}")
        if kind not in ("hadamard", "givens"):
            raise ValueError(f"rotation kind must be 'hadamard' or 'givens', "
                             f"got {kind!r}")
        ROTATION_SPAN = span
        ROTATION_KIND = kind
        if kind == "hadamard":
            ROTATION_NAME = "hadamard32" if span == 32 else f"hadamard{span}"
        else:
            ROTATION_NAME = f"givens{span}"


def _fwht_rows(x, span=None):
    """In-place-style FWHT over the last axis of x [..., span], FP32 accumulate.
    log2(span) butterfly stages (natural/Hadamard order, symmetric matrix), then
    *1/sqrt(span) to normalize. Caller casts back to the working dtype. Matches
    the runtime kernel's _fwht* ordering exactly so the rotation cancels in the
    matmul: (1/sqrt(S) H_S)(1/sqrt(S) H_S) = (1/S) S I = I.

    Span 32 reuses the literal 0.1767766953 (= 1/sqrt(32)) so the default path is
    bit-for-bit identical to the shipped FWHT-32 (proven in sanity_wider_rotation.py)."""
    span = ROTATION_SPAN if span is None else span
    assert span & (span - 1) == 0, f"span must be a power of two, got {span}"
    orig = x.shape
    x = x.reshape(-1, span).float()
    h = 1
    while h < span:
        x = x.reshape(-1, span // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        x = torch.stack([a + b, a - b], dim=2)
        x = x.reshape(-1, span)
        h *= 2
    norm = 0.1767766953 if span == 32 else (1.0 / (span ** 0.5))
    return (x * norm).reshape(orig)


# Back-compat alias: existing callers (rxf_quantize, rotate_fp16_weight) keep
# their name but now rotate over ROTATION_SPAN (32 by default).
def _fwht32_rows(x):
    return _fwht_rows(x, ROTATION_SPAN)


# ---------------------------------------------------------------------------
# Learned Givens rotation (ParoQuant stage b/c) — the LEARNED, importance-aware
# alternative to the fixed Hadamard. A span x span orthonormal R, built as a
# product of KROT rounds of disjoint Givens (pairwise) rotations, fit by greedy
# coordinate descent to minimize the (optionally importance-weighted) post-quant
# weight reconstruction MSE. Block-diagonal over each consecutive `span` input
# channels — same structure as the FWHT, so cancellation, the K=32 GEMM, the pack
# format and the per-group scale are ALL untouched (pre-pass-only, §0/§2 of
# PAROQUANT_RXF_INTEGRATION.md). Unlike the fixed Hadamard, R is fit to the data
# so importance lives natively in the rotated basis — resolving the quantizer's
# "Hadamard ⊥ importance" hard-block (stage c).
# ---------------------------------------------------------------------------

def _apply_rotation_rows(x, R):
    """Rotate the last axis of x [..., S] by the S x S matrix R: v -> R @ v,
    i.e. x @ R.T (row-vector convention). FP32 accumulate. The runtime applies
    the SAME x_block @ R.T to the activation, so (R x)·(R w) = x·w cancels."""
    orig = x.shape
    S = R.shape[0]
    xf = x.reshape(-1, S).float()
    return torch.matmul(xf, R.float().t()).reshape(orig)


def _givens_quant_mse(blocks, nl, imp=None):
    """Fast importance-weighted NL-quant MSE PROXY for ranking rotations.
    blocks: [G, GROUP] fp32 scale-groups; nl: [16] grid; imp: optional [G, GROUP].
    Uses the symmetric absmax scale (scale = absmax / max|nl|) + nearest-codepoint
    snap — a cheap stand-in for the exact signed interval search, good enough to
    ORDER candidate angles. Returns a scalar mean (importance-weighted) MSE."""
    maxabs = nl.abs().max()
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / maxabs
    idx = (blocks.unsqueeze(-1) / scale.unsqueeze(-1) - nl).abs().argmin(-1)
    deq = nl[idx] * scale
    e2 = (blocks - deq) ** 2
    if imp is not None:
        e2 = e2 * imp
    return e2.mean()


def _act_int8_quant_mse(blocks, imp=None):
    """Per-block symmetric int8 activation-quant MSE — the stage-(c) objective.
    blocks: [G, span] fp32 ACTIVATION blocks (rows of real calibration activation,
    reshaped to the rotation span). Mirrors the runtime fused rotate+quant
    (`rxf_kernels._rxf_rotate_quant_int8_kernel`): scale = absmax/127, round to the
    nearest integer in [-127, 127], dequant, squared error.

    Faithfulness caveat: the runtime int8 scale is PER-TOKEN (absmax over the full
    K row = max over that row's span-blocks); here it is PER span-BLOCK. A single
    shared span x span R acts identically on every block, so minimizing the mean
    per-block absmax-driven error tracks the per-row max it sets — a tight, poolable
    surrogate. The served PPL A/B is the ground-truth arbiter (the stage-(b) lesson:
    a moved offline proxy need not move PPL). imp: optional [G, span] rotated-basis
    importance weighting (default uniform = pure outlier flattening)."""
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = absmax / 127.0
    q = torch.round(blocks / scale).clamp_(-127, 127)
    deq = q * scale
    e2 = (blocks - deq) ** 2
    if imp is not None:
        e2 = e2 * imp
    return e2.mean()


def _rotated_importance(imp_block, R):
    """Transform per-channel importance into the ROTATED basis (stage c). The
    output error injected by a rotated-weight quant error eps is (R x)·eps, so the
    importance of rotated channel k is E[((R x)_k)^2] = sum_j R[k,j]^2 E[x_j^2].
    With imp_block ~ E[x_j^2] per block position [S], rotated imp = (R^2) @ imp.
    This is exactly the 'importance transformed into the rotated basis' the fixed
    Hadamard path could not do (it has no learned R to transform through)."""
    return torch.matmul(R.float() ** 2, imp_block.float())


def fit_givens_rotation(blocks_all, nl, imp_block=None, span=GROUP, krot=6,
                        n_angles=33, max_groups=8192, seed=0, init="hadamard",
                        score="weight"):
    """Fit a `span` x `span` orthonormal R from a pool of rotation blocks by
    greedy Givens coordinate descent, INITIALIZED at the fixed Hadamard and
    refined: KROT rounds, each a random perfect matching of the span channels
    into span/2 disjoint pairs; per pair line-search the angle that most reduces
    a quant-error objective on the rotated blocks. Only strictly-improving angles
    are committed, so the result is GUARANTEED no worse than the Hadamard it starts
    from (Hadamard is near-optimal for incoherence; a greedy fit from identity
    cannot rediscover its full mixing, so we refine FROM it — the QuaRot/SpinQuant
    move).

    score: which objective the line-search minimizes —
      'weight'     (stage b) — blocks_all are WEIGHT rows; minimize the post-quant
                   IQ4-NL weight-reconstruction MSE grouped into size-GROUP scale
                   groups (importance via _rotated_importance). Measured a PPL NULL.
      'activation' (stage c) — blocks_all are real calibration ACTIVATION rows;
                   minimize the per-block int8 activation-quant MSE
                   (_act_int8_quant_mse) — activation conditioning, the genuine
                   edge over data-blind Hadamard (it sees the REAL activation
                   outlier structure). `nl` is unused on this path.

    blocks_all: [G, span] fp32 — weight rows (weight score) or activation rows
    (activation score), pooled from a SAMPLE of the model (modules with
    K % span == 0). nl: [16] codebook (weight score only). imp_block: optional
    [span] per-block-position importance (un-rotated basis; the fit transforms it
    through R). init: 'hadamard' (start at the FWHT) or 'identity'. Returns R
    [span, span] fp32 with R @ R.T = I."""
    dev = blocks_all.device
    blocks_all = blocks_all.float()
    G = blocks_all.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    if G > max_groups:
        sel = torch.randperm(G, generator=g)[:max_groups].to(dev)
        blocks = blocks_all.index_select(0, sel).clone()
    else:
        blocks = blocks_all.clone()
    nlf = nl.float().to(dev)

    if imp_block is not None:
        imp_block = imp_block.float().to(dev)                        # [span]

    GR = span // GROUP                               # scale groups per block
    # Initialize R (and pre-rotate the blocks) at the Hadamard, so refinement can
    # only improve on it. R = _fwht_rows(I) is the exact normalized FWHT the
    # offline/runtime use, so a no-improvement fit reproduces Hadamard bit-for-bit.
    if init == "hadamard":
        R = _fwht_rows(torch.eye(span, dtype=torch.float32, device=dev), span)
        blocks = _apply_rotation_rows(blocks, R)
    else:
        R = torch.eye(span, dtype=torch.float32, device=dev)

    def _score(b):
        if score == "activation":
            # stage c: per-block int8 activation-quant MSE on the rotated rows.
            imp = None
            if imp_block is not None:
                ir = _rotated_importance(imp_block, R)               # [span]
                imp = ir.unsqueeze(0).expand_as(b)
            return _act_int8_quant_mse(b, imp)
        # stage b (weight): group the [B, span] blocks into size-GROUP scale
        # groups and score the IQ4-NL weight-reconstruction MSE.
        bg = b.reshape(-1, GROUP)
        imp = None
        if imp_block is not None:
            # rotated-basis importance for THIS R, tiled to the scale groups
            ir = _rotated_importance(imp_block, R)                    # [span]
            imp = ir.reshape(GR, GROUP).repeat(b.shape[0], 1)
        return _givens_quant_mse(bg, nlf, imp)

    angles = torch.linspace(-math.pi / 4, math.pi / 4, n_angles, device=dev)
    cur = _score(blocks)
    for r in range(krot):
        perm = torch.randperm(span, generator=g).tolist()
        pairs = [(perm[2 * t], perm[2 * t + 1]) for t in range(span // 2)]
        for (i, j) in pairs:
            xi = blocks[:, i].clone()
            xj = blocks[:, j].clone()
            best_th, best_sc = 0.0, cur
            for th in angles.tolist():
                c, s = math.cos(th), math.sin(th)
                blocks[:, i] = xi * c + xj * s
                blocks[:, j] = -xi * s + xj * c
                # importance transform depends on R, which only changes when we
                # COMMIT; within the search R is fixed, so _score is consistent.
                sc = _score(blocks)
                if sc < best_sc:
                    best_sc, best_th = sc, th
            c, s = math.cos(best_th), math.sin(best_th)
            blocks[:, i] = xi * c + xj * s
            blocks[:, j] = -xi * s + xj * c
            if best_th != 0.0:
                # Compose the committed Givens into R: rows i,j of R rotate.
                ri = R[i].clone()
                rj = R[j].clone()
                R[i] = ri * c + rj * s
                R[j] = -ri * s + rj * c
            cur = best_sc
    return R


def _lookup_importance(act_scales, mod_name):
    """Per-channel importance [K] for a module from collected activations, or
    None. Mirrors _get_importance's name variants (the quant loop's lookup)."""
    if not act_scales:
        return None
    for cand in (mod_name, mod_name + ".weight",
                 mod_name.replace(".weight_packed", ""),
                 mod_name.replace(".weight", "")):
        if cand in act_scales:
            return act_scales[cand]
    return None


def pool_rotation_blocks(plan, src, idx, spec, span, act_scales,
                         target_blocks=16384, device="cpu"):
    """Pool size-`span` rotation blocks (+ per-block-position importance) for ONE
    model-wide Givens R. STRATIFIED: take an equal strided slice from EVERY
    quantizable module (one big module — a single down_proj is ~1e6 blocks — must
    not dominate the fit, or R over-fits that module's basis and concentrates the
    OTHER modules' tiny weights into degenerate near-zero scale groups). Returns
    (blocks [G, span] fp32, imp_block [span] fp32 | None)."""
    mods_all = []
    for name, shard, kind in plan:
        if kind not in ("linear", "expert_fused"):
            continue
        with safe_open(src / shard, framework="pt") as f:
            w = f.get_tensor(name)
        if kind == "expert_fused":
            mods_all.extend(spec.split_experts(name, w))
        else:
            mod = name[:-7] if name.endswith(".weight") else name
            mods_all.append((mod, w))
    mods_all = [(n, w) for n, w in mods_all if w.shape[-1] % span == 0]
    if not mods_all:
        raise ValueError("no rotatable modules to fit the Givens R on")
    per_mod = max(1, target_blocks // len(mods_all))
    pool = []
    imp_accum = torch.zeros(span, dtype=torch.float64)
    imp_count = 0
    for mod_name, w2d in mods_all:
        b = w2d.reshape(-1, span).float()
        if b.shape[0] > per_mod:                     # even strided slice
            sel = torch.linspace(0, b.shape[0] - 1, per_mod).long()
            b = b.index_select(0, sel)
        pool.append(b)
        ik = _lookup_importance(act_scales, mod_name)
        if ik is not None and ik.numel() % span == 0:
            imp_accum += ik.double().reshape(-1, span).mean(0)
            imp_count += 1
    blocks = torch.cat(pool, 0).to(device)
    imp_block = (imp_accum / imp_count).float().to(device) if imp_count else None
    return blocks, imp_block


def rotate_fp16_weight(w):
    """Pre-rotate a dense 2-D weight [N, K] along K for a protected fp16
    expert: FWHT over each consecutive ROTATION_SPAN input channels, cast back
    to the source dtype. MUST be applied when rotation is on — the runtime
    rotates the activation format-blind before every MoE GEMM, so an unrotated
    fp16 weight would be silently wrong. Orthonormal -> only fp16/bf16 roundoff."""
    if not APPLY_ROTATION:
        return w.contiguous()
    N, K = w.shape
    if K % ROTATION_SPAN != 0:
        raise ValueError(
            f"protected fp16 expert needs K % {ROTATION_SPAN} == 0 under "
            f"rotation span {ROTATION_SPAN}, got K={K}")
    r = _fwht_rows(w.reshape(N, K // ROTATION_SPAN, ROTATION_SPAN)).reshape(N, K)
    return r.to(w.dtype).contiguous()


# ---------------------------------------------------------------------------
# Quantization core
# ---------------------------------------------------------------------------

def _quantize_group(gw, nl, scale, importance=None):
    """Quantize groups against NL grid.
    gw: [N, nG, 16] fp16, scale: [N, nG, 1] (fp32 here: the snapped fp16
    scale cast to fp32, mirroring the kernel's fp32 dequant).
    importance: optional [N, nG, 16] activation-aware weights.
    Returns (idx [N,nG,16], mse [N,nG]).
    A snapped scale of 0 (all-zero channel) yields inf/nan distances; argmin
    still returns a valid index and deq = nl*0 = 0, which is exact for the
    all-zero weights that produce it."""
    scaled = gw / scale
    dists = (scaled.unsqueeze(-1) - nl).abs()
    idx = dists.argmin(dim=-1)
    deq = nl[idx] * scale
    err_sq = (gw - deq) ** 2
    if importance is not None:
        err_sq = err_sq * importance
    mse = err_sq.mean(dim=-1)
    return idx, mse


def _pos_exact(w, nl_sorted, mids, gi, fp16_min):
    """Exact min-MSE positive scale per group via interval enumeration.
    w: [g,16] fp32. Returns (s>0 [g], mse [g]). MSE(s) is piecewise-quadratic in
    s with breakpoints at s = w_i/mid_j; within each interval the assignment is
    fixed and the optimum is the closed-form least-squares s* = Σwc/Σc²."""
    g = w.shape[0]
    absmids = mids.abs().clamp_min(fp16_min)
    cand = (w.abs()[:, :, None] / absmids[None, None, :]).reshape(g, -1)
    cand = cand.clamp_min(fp16_min)
    cand, _ = torch.sort(cand, dim=1)
    cand = torch.cat([cand[:, :1] * 0.5, cand, cand[:, -1:] * 2.0], dim=1)
    reps = torch.sqrt(cand[:, :-1] * cand[:, 1:])            # interval reps
    V = w[:, None, :] / reps[:, :, None]                     # [g, M, 16]
    idx = torch.bucketize(V, mids)                           # nearest codepoint
    C = nl_sorted[idx]
    if gi is not None:
        imp = gi[:, None, :]
        num = (imp * w[:, None, :] * C).sum(-1)
        den = (imp * C * C).sum(-1)
    else:
        num = (w[:, None, :] * C).sum(-1)
        den = (C * C).sum(-1)
    s = num / den.clamp_min(fp16_min)
    s = torch.clamp(s, cand[:, :-1], cand[:, 1:])            # stay in interval
    deq = C * s[:, :, None]
    e2 = (w[:, None, :] - deq) ** 2
    if gi is not None:
        e2 = e2 * gi[:, None, :]
    mse = e2.mean(-1)
    bm, bi = mse.min(1)
    bs = torch.gather(s, 1, bi[:, None])[:, 0]
    return bs, bm


def _exact_group_scale(gw, nl, gi, fp16_min, chunk=8192):
    """Exact SIGNED per-group scale for every group in gw [G,16]. Negative
    polarity handled by running the positive search on -w (the asymmetric NL
    grid or its mirror — worth ~15% group MSE, which is why the fp16 scale
    carries the sign). All-zero groups get scale 0 so the snap_scale_fp16
    underflow guard sees them. Chunked to bound memory."""
    dev = gw.device
    nl_sorted, _ = torch.sort(nl.float())
    mids = ((nl_sorted[:-1] + nl_sorted[1:]) / 2).contiguous()
    G = gw.shape[0]
    best = torch.empty(G, device=dev, dtype=torch.float32)
    gwf = gw.float()
    gif = gi.float() if gi is not None else None
    for c0 in range(0, G, chunk):
        sl = slice(c0, min(c0 + chunk, G))
        w = gwf[sl]
        gic = gif[sl] if gif is not None else None
        sp, mp = _pos_exact(w, nl_sorted, mids, gic, fp16_min)
        tn, mn = _pos_exact(-w, nl_sorted, mids, gic, fp16_min)
        best[sl] = torch.where(mn < mp, -tn, sp)
    # Robust fallback for NON-FINITE scales. The exact interval search occasionally
    # returns NaN/Inf for a group — and on ROCm it does so for WHOLE chunks of
    # certain (e.g. Givens-rotated) weights that the SAME search handles cleanly on
    # CPU (a GPU fp flake, not bad data). Zeroing those groups silently drops ~29%
    # of the weight -> broken model. Instead fall back to the plain SYMMETRIC scale
    # (absmax / max|code|): a valid, if slightly sub-optimal, quantization for those
    # groups — never NaN, never a dropped row. Genuine all-zero groups still map to 0.
    gamax = gw.abs().amax(dim=-1).float()
    # Defensive: a degenerate group whose interval search yields a non-finite scale
    # must not poison the fp16 scale tensor with NaN/Inf (-> NaN at serve time).
    # Fall back to the plain symmetric scale (absmax / max|code|) — a valid, if
    # sub-optimal, quantization rather than a dropped group. (The Givens-rotation
    # GPU-matmul flake that this once papered over is fixed at the source — the
    # rotation now runs on CPU; see rxf_quantize / §6.)
    bad = ~torch.isfinite(best)
    if bad.any():
        best = torch.where(bad, gamax / nl.float().abs().max().clamp_min(1e-12),
                           best)
    best[gamax == 0] = 0.0                       # all-zero group -> scale 0
    return best


def snap_scale_fp16(d):
    """Snap SIGNED per-group scales d [N, nG] (fp32) to float16 — the scale the
    RXF kernel reads directly (no mantissa/exponent). The fp16 sign carries the
    group's scale polarity.

    Returns (snapped float16 [N, nG], underflow int): count of nonzero groups
    whose fp16 scale collapsed to 0 or a subnormal (|snapped| <
    FP16_SCALE_UNDERFLOW), i.e. the fp16 scalar could not represent them."""
    snapped = d.to(torch.float16)
    nonzero = d.abs() > 0
    underflow = int(((snapped.float().abs() < FP16_SCALE_UNDERFLOW)
                     & nonzero).sum().item())
    return snapped, underflow


def rxf_quantize(w, importance=None):
    """Quantize a 2D weight [N, K] to RXF.

    IQ4-NL grid (16-point, signed, asymmetric). Plain fp16 group scale per 32
    weights: exact closed-form signed scale search (both polarities), then snap
    to float16. Codebook indices are re-assigned against the SNAPPED fp16 scale
    — the scale the kernel will actually use.

    Args:
        w: [N, K] weight tensor
        importance: optional [N, K] per-weight importance (gated OFF, see
                    ACT_AWARE_ENABLED).

    Returns:
        tensors  — weight_packed (uint8 [N,K//2]), weight_scale (float16
                   [N,K//32]) on the input device. The learned Givens R is
                   model-wide (set_givens_rotation) and written into config.json,
                   NOT per-module, so merged linears / TP shards all share it.
        mse_flat — [N * nG] per-group MSE vs the snapped scale (CPU-movable)
        underflow — int, nonzero groups whose fp16 scale underflowed to ~0
    """
    N, K = w.shape
    assert K % 2 == 0, f"K must be even, got {K}"
    rotate_hadamard = APPLY_ROTATION and ROTATION_KIND == "hadamard"
    rotate_givens = APPLY_ROTATION and ROTATION_KIND == "givens"
    if rotate_hadamard and importance is not None:
        # The FIXED Hadamard has no learned R to transform importance through, so
        # rotation and importance stay mutually exclusive on this path. The LEARNED
        # Givens path (below) DOES transform importance into the rotated basis —
        # that is the stage-(c) edge — so it is NOT blocked here.
        raise ValueError(
            "Hadamard rotation and activation-aware importance are mutually "
            "exclusive: importance must be transformed into the rotated basis "
            "first, which the fixed Hadamard cannot do (use rotation=givens).")
    if APPLY_ROTATION and K % ROTATION_SPAN != 0:
        # Block-diagonal SxS rotation requires its blocks to land exactly on
        # multiples of ROTATION_SPAN. K%SPAN!=0 would fold real energy into pad
        # lanes that are then truncated, breaking cancellation for the last block
        # only. The runtime also rejects K%SPAN!=0, so guard loudly here. (SPAN is
        # a multiple of GROUP, so K%SPAN==0 implies K%GROUP==0.)
        raise ValueError(
            f"{ROTATION_KIND} rotation requires K to be a multiple of the span "
            f"{ROTATION_SPAN}, got K={K}")
    wf = w.to(torch.float16)
    dev = wf.device
    nl = ACTIVE_TABLE.to(dev)
    FP16_MIN = torch.tensor(6.104e-5, dtype=torch.float16, device=dev)

    pad = (GROUP - K % GROUP) % GROUP
    if pad:
        wf = torch.nn.functional.pad(wf, (0, pad))
    Kp = wf.shape[1]
    nG = Kp // GROUP

    # ── Rotation: rotate each consecutive ROTATION_SPAN channels FIRST (a wider
    # span reaches across the size-32 scale-group boundaries), THEN reshape to
    # size-32 groups so the scale search + codebook run unchanged on the rotated
    # weights. The rotation is over the FULL ROW window, decoupled from the K=32
    # GEMM — only the weight is changed offline; the GEMM still reads size-32
    # scale groups. The runtime applies the matching activation rotation so it
    # cancels. ──
    R_givens = None
    gi = None
    if rotate_hadamard:
        nS = Kp // ROTATION_SPAN
        wf = _fwht_rows(wf.reshape(N, nS, ROTATION_SPAN)).reshape(N, Kp)
        wf = wf.to(torch.float16)
    elif rotate_givens:
        span = ROTATION_SPAN
        nB = Kp // span
        # The model-wide learned R is fit ONCE before the loop (main()).
        if GIVENS_R is None:
            raise ValueError(
                "rotation=givens but no learned R installed; call "
                "set_givens_rotation(R) before quantizing (main() fits it).")
        R_givens = GIVENS_R
        assert R_givens.shape == (span, span), (
            f"learned R is {tuple(R_givens.shape)}, expected ({span},{span})")
        # ROCm matmul flake: the large-M [N*nB, span] @ [span, span] rotation
        # silently drops ~29% of output rows to ZERO on GPU (norm loss 43.9->37.2),
        # though the IDENTICAL matmul is exact on CPU (hadamard avoids this — its
        # FWHT is butterfly add/sub, no matmul). The rotation is cheap (one tiny
        # SxS matmul/row), so run it on CPU then move back; the expensive scale
        # search stays on GPU. See PAROQUANT_RXF_INTEGRATION.md §6.
        wf = _apply_rotation_rows(
            wf.reshape(N, nB, span).cpu(), R_givens.cpu()
        ).reshape(N, Kp).to(torch.float16).to(dev)
        # Importance INTO the rotated basis: per block-position [span], tiled to
        # the full row + broadcast across rows. This is the transform the fixed
        # Hadamard could not do; the quant core's gi path then uses it natively.
        if importance is not None:
            imp_k = importance.float().to(dev).mean(0)            # [K]
            if pad:
                imp_k = torch.nn.functional.pad(imp_k, (0, pad))  # [Kp]
            imp_block = imp_k.reshape(nB, span).mean(0)           # [span]
            rib = _rotated_importance(imp_block, R_givens)        # [span]
            gi = (rib.to(torch.float16).repeat(nB)
                  .unsqueeze(0).expand(N, -1).reshape(N, nG, GROUP))

    gw = wf.reshape(N, nG, GROUP)                                # [N, nG, 16]

    # Un-rotated (or no-rotation) importance: reshape to match groups if provided.
    if gi is None and importance is not None:
        imp = importance.to(torch.float16).to(dev)
        if pad:
            imp = torch.nn.functional.pad(imp, (0, pad))
        gi = imp.reshape(N, nG, GROUP)                           # [N, nG, 16]

    # ── Exact closed-form per-group scale (interval enumeration, signed) ──
    gw_flat = gw.reshape(-1, GROUP)                              # [N*nG, 16]
    gi_flat = gi.reshape(-1, GROUP) if gi is not None else None
    best = _exact_group_scale(gw_flat, nl, gi_flat, FP16_MIN).reshape(N, nG)

    # ── fp16 snap: one plain float16 scale per group (no mantissa/exponent) ──
    snapped, underflow = snap_scale_fp16(best)

    # ── Final quantization against the SNAPPED fp16 scale (the kernel's scale).
    # Divide in fp32 (snapped.float()) to mirror the kernel's fp32 dequant. ──
    idx, mse = _quantize_group(gw, nl, snapped.float().unsqueeze(-1))

    # ── Pack: 4-bit NL index, two per byte ──
    idx_flat = idx.reshape(N, Kp).to(torch.uint8)
    packed = (idx_flat[:, 0::2] | (idx_flat[:, 1::2] << 4)).to(torch.uint8)

    nG_orig = (K + GROUP - 1) // GROUP
    tensors = {
        "weight_packed": packed[:, :K // 2].contiguous(),
        "weight_scale":  snapped[:, :nG_orig].contiguous(),   # float16 [N,K//32]
    }
    return tensors, mse.reshape(-1), underflow


# ---------------------------------------------------------------------------
# Block-wise FP8 (e4m3) — optional attention path (--fp8-attn)
# ---------------------------------------------------------------------------
FP8_E4M3_MAX = 448.0


def fp8_block_quantize(w, block=(128, 128)):
    """Block-wise symmetric FP8 (e4m3) quantization, DeepSeek/vLLM convention.

    w: [N, K] float weight. Returns (w_fp8 [N,K] float8_e4m3fn,
    weight_scale_inv [ceil(N/bn), ceil(K/bk)] float32) such that
        W ~= w_fp8.float() * weight_scale_inv   (per bn x bk block).
    """
    N, K = w.shape
    bn, bk = block
    nN, nK = (N + bn - 1) // bn, (K + bk - 1) // bk
    wf = w.float()
    wp = torch.nn.functional.pad(wf, (0, nK * bk - K, 0, nN * bn - N))
    wp = wp.reshape(nN, bn, nK, bk).permute(0, 2, 1, 3)            # [nN,nK,bn,bk]
    amax = wp.abs().amax(dim=(-1, -2)).clamp_min(1e-12)           # [nN,nK]
    scale = amax / FP8_E4M3_MAX
    wq = (wp / scale[:, :, None, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    wq = wq.to(torch.float8_e4m3fn)
    wq = wq.permute(0, 2, 1, 3).reshape(nN * bn, nK * bk)[:N, :K].contiguous()
    return wq, scale.to(torch.float32).contiguous()


# ---------------------------------------------------------------------------
# Activation-aware calibration
# ---------------------------------------------------------------------------

def collect_activations(model_path, datasets, proportions, n_samples=512,
                        seq_len=2048, device="cpu", span=GROUP,
                        collect_blocks=False, max_blocks_per_mod=2048,
                        target_blocks=16384):
    """Run calibration data through the model and collect per-layer
    input activation magnitudes for activation-aware quantization.

    Args:
        model_path: path to BF16/FP16 model
        datasets: list of dataset names or paths (HF datasets or local jsonl)
        proportions: list of floats summing to 1.0, proportion from each dataset
        n_samples: total calibration samples
        seq_len: sequence length per sample
        device: device to run calibration on
        span: rotation span; SIGNED activation rows are reshaped to [*, span]
              blocks for the stage-(c) Givens fit.
        collect_blocks: also pool a stratified sample of SIGNED activation blocks
              (for fit_givens_rotation(score="activation")).
        max_blocks_per_mod: cap of pooled blocks PER Linear module (bounds memory).
        target_blocks: total pooled-block budget across all modules.

    Returns:
        (act_scales, act_blocks):
          act_scales: dict weight name -> [K] per-channel importance (mean |x|,
                      normalized to mean 1.0).
          act_blocks: pooled SIGNED activation blocks [G, span] fp32 (CPU), or
                      None when collect_blocks is False / nothing eligible.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import datasets as hf_datasets

    assert len(datasets) == len(proportions)
    assert abs(sum(proportions) - 1.0) < 1e-6

    print(f"  Loading model for calibration from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # device == "auto" -> shard across all visible GPUs (needed for models that
    # don't fit on one card, e.g. 35B fp16 ~70GB). Inputs go to the model's
    # first device.
    device_map = "auto" if device == "auto" else device
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map=device_map,
        trust_remote_code=True)
    model.eval()
    input_device = model.device if device == "auto" else device

    # ── Collect calibration tokens from mixed datasets ──
    all_tokens = []
    for ds_name, prop in zip(datasets, proportions):
        n_from_ds = max(1, int(n_samples * prop))
        print(f"    Loading {n_from_ds} samples from {ds_name}...")

        if ds_name.endswith(".jsonl") or ds_name.endswith(".json"):
            import json as json_mod
            texts = []
            with open(ds_name) as f:
                for line in f:
                    obj = json_mod.loads(line)
                    text = obj.get("text", obj.get("content", str(obj)))
                    texts.append(text)
                    if len(texts) >= n_from_ds:
                        break
        else:
            ds = hf_datasets.load_dataset(ds_name, split="train",
                                           streaming=True)
            texts = []
            for sample in ds:
                text = sample.get("text", sample.get("content", ""))
                if len(text) > 100:
                    texts.append(text)
                if len(texts) >= n_from_ds:
                    break

        for text in texts:
            toks = tokenizer.encode(text, add_special_tokens=False,
                                     max_length=seq_len, truncation=True)
            if len(toks) >= 64:
                all_tokens.append(torch.tensor(toks[:seq_len]))

    print(f"    Collected {len(all_tokens)} calibration sequences")

    # ── Hook linear layers to capture input activations ──
    act_sums = {}
    act_counts = {}
    act_blocks_buf = {}          # name -> list of [g, span] cpu fp32 (signed)
    act_blocks_count = {}        # name -> blocks pooled so far (cap guard)
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            xin = input[0].detach().float()
            x = xin.abs()
            # x: [batch, seq, hidden] — average over batch and seq
            act_mean = x.mean(dim=(0, 1))  # [hidden] = [K] for this layer
            if name not in act_sums:
                act_sums[name] = act_mean
                act_counts[name] = 1
            else:
                act_sums[name] += act_mean
                act_counts[name] += 1
            # Pool a STRIDED sample of SIGNED activation blocks for the stage-(c)
            # Givens fit (per-module cap → no single big layer dominates the pool).
            if collect_blocks and xin.shape[-1] % span == 0:
                have = act_blocks_count.get(name, 0)
                if have < max_blocks_per_mod:
                    blk = xin.reshape(-1, span)           # [tokens*nB, span]
                    take = min(max_blocks_per_mod - have, blk.shape[0])
                    if take < blk.shape[0]:
                        sel = torch.linspace(0, blk.shape[0] - 1, take).long()
                        blk = blk.index_select(0, sel.to(blk.device))
                    act_blocks_buf.setdefault(name, []).append(blk.cpu())
                    act_blocks_count[name] = have + blk.shape[0]
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)

    # ── Run calibration ──
    print(f"    Running {len(all_tokens)} sequences through model...")
    with torch.no_grad():
        for i, toks in enumerate(all_tokens):
            input_ids = toks.unsqueeze(0).to(input_device)
            model(input_ids)
            if (i + 1) % 50 == 0:
                print(f"      {i+1}/{len(all_tokens)} sequences", flush=True)

    for h in hooks:
        h.remove()

    # ── Build per-weight importance: |activation| broadcasted to [N, K] ──
    act_scales = {}
    for name, act_sum in act_sums.items():
        avg_act = act_sum / act_counts[name]  # [K]
        # Normalize so mean importance = 1.0 (doesn't change relative weighting)
        avg_act = avg_act / (avg_act.mean() + 1e-8)
        act_scales[name] = avg_act.to(torch.float16).cpu()

    # ── Pool the SIGNED activation blocks (stratified, equal per module) ──
    act_blocks = None
    if collect_blocks and act_blocks_buf:
        mods = sorted(act_blocks_buf)
        per_mod = max(1, target_blocks // len(mods))
        pool = []
        for name in mods:
            b = torch.cat(act_blocks_buf[name], 0)        # [g, span]
            if b.shape[0] > per_mod:                      # even strided slice
                sel = torch.linspace(0, b.shape[0] - 1, per_mod).long()
                b = b.index_select(0, sel)
            pool.append(b)
        act_blocks = torch.cat(pool, 0).float()           # [G, span] cpu
        print(f"    Pooled {act_blocks.shape[0]} activation blocks "
              f"(span={span}) from {len(mods)} modules for the Givens fit")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    Collected activations for {len(act_scales)} layers")
    return act_scales, act_blocks


# ---------------------------------------------------------------------------
# Model traversal helpers
# ---------------------------------------------------------------------------

def is_ignored(name, patterns):
    for p in patterns:
        if p.startswith("re:"):
            if re.search(p[3:], name):
                return True
        elif p in name:
            return True
    return False


# Map registry kinds -> the quantizer's internal plan tags.
_KIND_TO_PLAN = {KIND_EXPERT: "expert_fused", KIND_LINEAR: "linear"}


def plan_kind(spec, name, shape):
    """Registry kind -> 'expert_fused' | 'linear' | 'copy'."""
    return _KIND_TO_PLAN.get(spec.classify(name, shape), "copy")


# ---------------------------------------------------------------------------
# Protected fp16 experts
# ---------------------------------------------------------------------------
# The expert index is in the post-split output module name (registry
# expert_out_tmpl produces '...experts.{e}.{proj}'); per-expert-tensor archs
# (qwen3_moe / minimax_m2) carry it in the source name the same way.
_EXPERT_ID_RE = re.compile(r"\.experts\.(\d+)\.")


def parse_fp16_experts(spec_arg):
    """--fp16-experts SPEC -> {layer_idx: sorted [expert ids]}.
    SPEC is inline 'L:E,L:E,...' pairs or a path to a JSON {"L": [E, ...]}."""
    if not spec_arg:
        return {}
    out = defaultdict(set)
    path = Path(spec_arg)
    if path.exists():
        data = json.load(open(path))
        if not isinstance(data, dict):
            raise ValueError(f"--fp16-experts JSON must be an object, "
                             f"got {type(data).__name__}")
        for k, v in data.items():
            out[int(k)].update(int(e) for e in v)
    else:
        for pair in spec_arg.split(","):
            pair = pair.strip()
            if not pair:
                continue
            try:
                l, e = pair.split(":")
                out[int(l)].add(int(e))
            except ValueError:
                raise ValueError(
                    f"--fp16-experts: bad pair {pair!r} (want layer:expert "
                    f"pairs like '12:5,30:88' or a JSON file path)") from None
    return {l: sorted(es) for l, es in out.items()}


def protected_expert_id(fp16_experts, layer_idx, mod_name):
    """Expert id when this post-split module is a protected expert, else None."""
    ids = fp16_experts.get(layer_idx)
    if not ids:
        return None
    m = _EXPERT_ID_RE.search(mod_name + ".")
    if m and int(m.group(1)) in ids:
        return int(m.group(1))
    return None


def validate_fp16_experts(fp16_experts, spec, src, idx, cfg, only):
    """Pre-flight for --fp16-experts: every listed layer must have MoE expert
    modules and every listed expert id must be in range. Errors out before any
    quantization. Prints the protected-expert cost table."""
    if not fp16_experts:
        return
    if only is not None:
        missing = sorted(set(fp16_experts) - only)
        if missing:
            raise ValueError(
                f"--fp16-experts lists layer(s) {missing} excluded by "
                f"--layers; their experts would be neither quantized nor "
                f"protected.")

    # Derive per-layer expert count + per-expert byte cost from the checkpoint
    # itself (shapes are ground truth; config num-expert keys vary by family).
    per_layer = {l: {"count": 0, "elems": 0} for l in fp16_experts}
    for name, shard in idx.items():
        li = spec.layer_index(name)
        if li not in per_layer:
            continue
        with safe_open(src / shard, framework="pt") as f:
            shape = f.get_slice(name).get_shape()
        kind = spec.classify(name, shape)
        if kind == KIND_EXPERT:
            E = shape[0]
            per_layer[li]["count"] = max(per_layer[li]["count"], E)
            per_layer[li]["elems"] += math.prod(shape) // E
        elif kind == KIND_LINEAR:
            m = _EXPERT_ID_RE.search(name)
            if m:
                e = int(m.group(1))
                per_layer[li]["count"] = max(per_layer[li]["count"], e + 1)
                if e == 0:
                    per_layer[li]["elems"] += math.prod(shape)

    tc = cfg.get("text_config", cfg)
    cfg_E = (tc.get("num_experts") or tc.get("n_routed_experts")
             or tc.get("moe_num_experts"))

    print("protected fp16 experts (--fp16-experts):")
    print(f"  {'layer':>5}  {'expert':>6}  {'fp16 MB':>9}  {'rxf MB':>9}  "
          f"{'cost':>6}")
    total_fp16 = total_q = 0
    for l in sorted(fp16_experts):
        info = per_layer[l]
        if info["count"] == 0:
            raise ValueError(
                f"--fp16-experts: layer {l} has no MoE expert modules in "
                f"{src}")
        if cfg_E and info["count"] != cfg_E:
            print(f"  NOTE: layer {l}: checkpoint has {info['count']} experts, "
                  f"config says {cfg_E}; using checkpoint count.")
        bad = [e for e in fp16_experts[l] if not 0 <= e < info["count"]]
        if bad:
            raise ValueError(
                f"--fp16-experts: expert id(s) {bad} out of range for layer "
                f"{l} ({info['count']} experts)")
        fp16_b = info["elems"] * 2                 # source dtype, 2 bytes
        q_b = info["elems"] * 4.5 / 8             # 4.5 bits per weight
        for e in fp16_experts[l]:
            print(f"  {l:>5}  {e:>6}  {fp16_b / 1e6:>9.1f}  {q_b / 1e6:>9.1f}  "
                  f"{fp16_b / q_b:>5.2f}x")
            total_fp16 += fp16_b
            total_q += q_b
    n = sum(len(v) for v in fp16_experts.values())
    print(f"  total: {n} expert(s), {total_fp16 / 1e6:.1f} MB fp16 vs "
          f"{total_q / 1e6:.1f} MB quantized "
          f"(+{(total_fp16 - total_q) / 1e6:.1f} MB)")


def build_quant_config(spec, config, extra_ignore, fp8_attn=False,
                       fp16_experts=None, ignore_override=None,
                       act_dtype="fp16", rotation_matrix=None):
    weights = {
        "num_bits": 4,
        "type": "float",
        "group_size": GROUP,
        "grid": ACTIVE_NAME,
        "codebook": [int(v) if float(v).is_integer() else float(v)
                     for v in ACTIVE_TABLE.tolist()],
        "scale_dtype": "torch.float16",
        "scale_encoding": "fp16_per_group",
        "symmetric": False,
        # Rotation tag. MANDATORY gate: the kernel must apply the matching
        # activation rotation, else the matmul is silently wrong. None = no
        # rotation (backward compatible). 'hadamard{S}' = fixed in-kernel FWHT;
        # 'givens{S}' = the learned model-wide R below (applied externally).
        "rotation": ROTATION_NAME if APPLY_ROTATION else None,
        # Runtime activation dtype: "fp16" (safe default, fp16 cores) or
        # "int8" (W4A8 wide blocked dot, fast cores). Read by RXFConfig;
        # env RXF_ACT_DTYPE overrides at serve time.
        "act_dtype": act_dtype,
    }
    if rotation_matrix is not None:
        # The model-wide learned orthonormal R [S,S] (rotation=givens{S}). The
        # runtime builds a tensor from this and rotates each activation's
        # consecutive-S channels by it, exactly cancelling the offline weight
        # rotation. Stored inline (tiny: S² floats) so no extra checkpoint tensor.
        weights["rotation_matrix"] = rotation_matrix
    groups = {"group_0": {
        "format": "rxf-pack-quantized",
        "input_activations": None,
        "output_activations": None,
        "targets": ["Linear"],
        "weights": weights,
    }}
    # ignore_override (from --overrideLayers) replaces the spec's default ignore
    # patterns with a list already rewritten in main() to drop the broad patterns
    # that would have skipped the now-quantized override modules.
    base_ignore = (ignore_override if ignore_override is not None
                   else spec.runtime_ignore(config))
    ignore = list(base_ignore) + list(extra_ignore)
    if fp8_attn and spec.fp8_attn_res:
        bn, bk = spec.fp8_attn_block
        groups["group_1"] = {
            "format": "float-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": list(spec.fp8_attn_targets),
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "block",
                "block_structure": [int(bn), int(bk)],
                "symmetric": True,
                "dynamic": False,
                "scale_dtype": "torch.float32",
                "scale_name": "weight_scale_inv",
                "fp8_max": FP8_E4M3_MAX,
                "dequant": "W = weight.float() * weight_scale_inv  (per block)",
            },
        }
        # The FP8'd attn projections must reach the runtime fp8 loader, so they
        # must NOT be in ignore (which is checked first). Drop the broad
        # self_attn ignore and keep only the non-fp8 attn Linears (e.g. g_proj).
        ignore = [p for p in ignore if "self_attn" not in p]
        ignore += list(spec.fp8_attn_keep_ignore)
    qcfg = {
        "config_groups": groups,
        "format": "rxf-pack-quantized",
        "quant_method": "rxf",
        "quantization_status": "compressed",
        "version": "rxf-2.0",
        "ignore": ignore,
    }
    if fp16_experts:
        # String layer keys, int id lists — exactly what RXFConfig
        # .from_config reads.
        qcfg["fp16_experts"] = {str(l): sorted(es)
                                for l, es in fp16_experts.items()}
    return qcfg


# ---------------------------------------------------------------------------
# CPU quantization is an inline RAM-budgeted streaming loop in main() (no
# process pool): one rxf_quantize call already saturates all cores via
# ATen intra-op threads, and running experts concurrently would multiply the
# ~GB interval-enumeration scratch. See the CPU branch below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Trigger flags forwarded by the image entrypoint; consumed here so argparse
    # does not choke on them (mirrors tune.py's --tune handling).
    ap.add_argument("--quantize", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--rxf", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", default=None, type=Path,
                    help="output dir (default: <input>-RXF)")
    ap.add_argument("--act-dtype", choices=["fp16", "int8"], default="fp16",
                    help="runtime activation dtype written into config.json. "
                         "'fp16' (default, correctness-safe, fp16 cores) or "
                         "'int8' (W4A8 wide blocked dot, 2x fast cores). The "
                         "checkpoint is identical either way — this only sets "
                         "the served path; env RXF_ACT_DTYPE overrides it.")
    ap.add_argument("--ignore", action="append", default=[])
    ap.add_argument("--layers", default=None,
                    help="restrict to layer indices, e.g. 3 or 3,4,7")
    ap.add_argument("--overrideLayers", nargs="+", default=[], metavar="PATTERN",
                    help="force normally-COPIED modules INTO the RXF grid. "
                         "The arch spec copies attention, vision, mtp, lm_head, "
                         "embeddings, norms by default; list ANY name fragment "
                         "or 're:'-prefixed regex to override that and quantize "
                         "the matches instead. Examples: "
                         "--overrideLayers self_attn linear_attn  (quantize all "
                         "attention), --overrideLayers visual lm_head, "
                         "--overrideLayers 're:\\.self_attn\\.(q|k|v|o)_proj'. "
                         "Only 2-D .weight tensors can be quantized; matches on "
                         "1-D/non-weight tensors are left copied with a warning. "
                         "The runtime 'ignore' list is rewritten so quantized "
                         "overrides load and their copied siblings stay skipped.")
    ap.add_argument("--fp16-experts", default=None, metavar="SPEC",
                    help="protect MoE experts in the source dtype instead of "
                         "quantizing: inline 'layer:expert' pairs "
                         "('12:5,12:17,30:88') or a path to a JSON file "
                         "{\"12\": [5, 17], \"30\": [88]}. Each protected "
                         "module emits <module>.weight_fp16 (pre-rotated when "
                         "rotation is on) and no packed/scale tensors; "
                         "config.json gains quantization_config.fp16_experts.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quant-only", action="store_true",
                    help="skip copy-through (not a runnable output)")
    ap.add_argument("--gpu", action="store_true",
                    help="run quantization on GPU (4 layers at a time, "
                         "1 per GPU). Falls back to CPU if no GPU.")
    ap.add_argument("--gpu-all", action="store_true",
                    help="load all layers to GPUs at once (round-robin), "
                         "quantize in parallel, single write pass. "
                         "Needs enough VRAM for all weights.")
    ap.add_argument("--gpu-stream", action="store_true",
                    help="stream layers: a background thread prefetches the "
                         "next chunk of layers from disk into RAM while the "
                         "GPUs quantize the current chunk (1 layer/GPU), "
                         "freeing each chunk after write. Bounds host RAM to "
                         "~2 chunks instead of the whole model. For large "
                         "models that don't fit fully in RAM.")
    ap.add_argument("--load-chunks", type=int, default=4,
                    help="number of layer chunks for --gpu-stream (default 4). "
                         "Peak host RAM ~= 2/N of the model weights; N=4 keeps "
                         "~half resident. Higher N = less RAM, finer overlap.")
    ap.add_argument("--act-aware", nargs="+", metavar="DATASET",
                    help="activation-aware calibration datasets. "
                         "HF dataset names or local .jsonl paths. "
                         "1 to 10 datasets.")
    ap.add_argument("--act-proportions", type=str, default=None,
                    help="comma-separated proportions for each dataset, "
                         "e.g. '0.4,0.3,0.2,0.1'. Must sum to 1.0. "
                         "If omitted, datasets are weighted equally.")
    ap.add_argument("--act-samples", type=int, default=512,
                    help="total calibration samples (default 512)")
    ap.add_argument("--act-seq-len", type=int, default=2048,
                    help="sequence length per calibration sample (default 2048)")
    ap.add_argument("--act-device", type=str, default="cuda:0",
                    help="device for calibration forward pass (default cuda:0)")
    ap.add_argument("--savegroupmse", action="store_true",
                    help="write per-group MSE CSV to output dir (large file)")
    ap.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True,
                    help="after quantizing, write static eval metrics "
                         "(integrity/coverage/bpw/fidelity) to "
                         "<output>/eval_metrics.txt. Use --no-eval to skip.")
    ap.add_argument("--codebook", default="nl",
                    help="16-point codebook: 'nl' (default IQ4-NL), 'lloyd' "
                         "(Step-3.7 sub-Gaussian optimal), or path to a JSON "
                         "list of 16 numbers. Written into config.json so vLLM "
                         "decodes with the matching grid.")
    ap.add_argument("--fp8-attn", dest="fp8_attn", action="store_true",
                    help="also block-FP8 (e4m3, 128x128) quantize attention "
                         "q/k/v/o_proj instead of copying them. Adds a "
                         "float-quantized config group; requires the arch spec "
                         "to declare fp8_attn_res (e.g. step3p7). No-op on archs "
                         "without it.")
    ap.add_argument("--no-rotate", dest="rotate", action="store_false",
                    default=True,
                    help="disable the per-group 32x32 Hadamard rotation, which "
                         "is ON by default. Rotation Gaussianizes each group "
                         "before scale+codebook (tightens the IQ4-NL fit and the "
                         "fp16 group scale) and writes rotation=hadamard32 into "
                         "config.json. It is auto-disabled if any quantize-target "
                         "has K %% 32 != 0 or with activation-aware calibration; "
                         "pass this flag to force it off.")
    ap.add_argument("--rotation-span", type=int, default=GROUP,
                    metavar="S",
                    help="FWHT rotation WIDTH S (power of two, multiple of 32). "
                         "Default 32 = the shipped per-group Hadamard "
                         "(rotation=hadamard32). A WIDER span (e.g. 128, 512) "
                         "mixes outliers across the size-32 scale-group "
                         "boundaries — pre-pass only, the K=32 int8 GEMM is "
                         "untouched — and writes rotation=hadamard<S> into "
                         "config.json (the runtime kernel must apply the matching "
                         "span; see PAROQUANT_RXF_INTEGRATION.md). ParoQuant "
                         "stage (a). Requires every quantize-target to have "
                         "K %% S == 0, else rotation is disabled model-wide.")
    ap.add_argument("--rotation-kind", choices=["hadamard", "givens"],
                    default="hadamard",
                    help="rotation FAMILY. 'hadamard' (default) = the fixed "
                         "data-blind Sylvester FWHT (ParoQuant stage a). 'givens' "
                         "= a LEARNED model-wide orthonormal R (ParoQuant stage "
                         "b/c), fit by Hadamard-init Givens coordinate descent: "
                         "data-blind it minimizes the post-quant weight MSE (stage "
                         "b); with --act-aware <calib> it minimizes the REAL "
                         "per-token int8 ACTIVATION-quant error on calibration "
                         "activations (stage c, activation conditioning). Emits "
                         "rotation=givens<S> + an inline rotation_matrix. "
                         "MEASURED A NULL/REGRESSION on Qwen3.5-4B (see "
                         "PAROQUANT_RXF_INTEGRATION.md §7): the learned R loses to "
                         "the fixed Hadamard on both objectives; kept for the "
                         "record + future models, not a recommended default.")
    args = ap.parse_args()
    if args.gpu_all or args.gpu_stream:
        args.gpu = True

    cb_name, cb_table = resolve_codebook(args.codebook)
    set_codebook(cb_name, cb_table)
    print(f"codebook: {cb_name}  {[int(v) for v in cb_table.tolist()]}")

    # Rotation is ON by default; it falls back to off when it can't be used.
    # Finalized below after the model dimensions are known (K %% span pre-flight).
    rotate = args.rotate
    rotation_span = int(args.rotation_span)
    rotation_kind = args.rotation_kind
    if rotate and (rotation_span < GROUP
                   or rotation_span & (rotation_span - 1) != 0
                   or rotation_span % GROUP != 0):
        raise ValueError(
            f"--rotation-span must be a power of two and a multiple of {GROUP}, "
            f"got {rotation_span}")
    if rotate and args.act_aware and rotation_kind == "hadamard":
        # The FIXED Hadamard cannot fold importance (no learned R to transform it
        # through), so it yields to an explicit --act-aware. The LEARNED Givens
        # rotation DOES co-fit importance in the rotated basis, so it stays ON.
        print("  NOTE: --act-aware requested with the fixed Hadamard; disabling "
              "rotation (use --rotation-kind givens to learn an "
              "importance-aware rotation instead).")
        rotate = False

    only = None
    if args.layers:
        only = {int(x) for x in args.layers.split(",")}

    fp16_experts = parse_fp16_experts(args.fp16_experts)

    # ── Activation-aware calibration (ParoQuant stage c — see module docstring) ──
    act_scales = {}
    act_blocks = None
    if args.act_aware and not ACT_AWARE_ENABLED:
        print("  NOTE: activation-aware calibration is disabled "
              "(ACT_AWARE_ENABLED=False); using naive path.")
    if args.act_aware and ACT_AWARE_ENABLED:
        datasets = args.act_aware
        if len(datasets) > 10:
            raise ValueError("Max 10 calibration datasets")
        if args.act_proportions:
            proportions = [float(x) for x in args.act_proportions.split(",")]
            if len(proportions) != len(datasets):
                raise ValueError(
                    f"Got {len(datasets)} datasets but {len(proportions)} "
                    f"proportions")
            if abs(sum(proportions) - 1.0) > 0.01:
                raise ValueError(
                    f"Proportions must sum to 1.0, got {sum(proportions)}")
        else:
            proportions = [1.0 / len(datasets)] * len(datasets)
        print("Activation-aware calibration:")
        for ds, p in zip(datasets, proportions):
            print(f"  {p*100:.0f}%  {ds}")
        # Stage (c): for the learned-Givens path, pool SIGNED activation blocks so
        # R can be fit to the real int8 activation-quant error (activation
        # conditioning). Other paths only need the per-channel importance.
        want_blocks = (rotation_kind == "givens")
        act_scales_collected, act_blocks = collect_activations(
            args.input, datasets, proportions,
            n_samples=args.act_samples,
            seq_len=args.act_seq_len,
            device=args.act_device,
            span=rotation_span,
            collect_blocks=want_blocks,
        )
        # ISOLATE THE VARIABLE: when fitting R on the activation objective, leave
        # weight-side importance OFF so the ONLY change vs stage (b) is R's
        # objective (a clean A/B). Otherwise keep the legacy per-channel weight
        # importance path.
        if want_blocks and act_blocks is not None:
            act_scales = {}
        else:
            act_scales = act_scales_collected

    src = args.input
    dst = args.output or src.parent / f"{src.name}-RXF"

    idx = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(src / "config.json"))
    ignore = list(args.ignore)

    spec = get_spec(cfg)
    print(f"detected arch: {spec.family}")

    # ── Protected-expert pre-flight: validate + cost table, before any work ──
    validate_fp16_experts(fp16_experts, spec, src, idx, cfg, only)

    def _protected(mod_name):
        return protected_expert_id(
            fp16_experts, spec.layer_index(mod_name), mod_name) is not None

    # ── Classify tensors via the model registry ──
    do_fp8_attn = bool(args.fp8_attn and spec.fp8_attn_res)
    if args.fp8_attn and not spec.fp8_attn_res:
        print(f"  NOTE: --fp8-attn requested but arch '{spec.family}' declares "
              f"no fp8_attn_res; attention copied as-is.")
    counts = {"linear": 0, "expert_fused": 0, "fp8_attn": 0,
              "copy": 0, "ignored": 0}
    plan = []
    rot_bad = []   # quantize targets with K %% 32 != 0 (rotation pre-flight)
    overridden = []        # modules forced copy->linear via --overrideLayers
    override_skipped = []  # --overrideLayers matches that aren't 2-D .weight
    linear_like = {}       # name -> is a 2-D .weight (Linear-shaped) tensor
    for name, shard in idx.items():
        if only is not None and spec.layer_index(name) not in only:
            plan.append((name, shard, "copy"))
            counts["copy"] += 1
            continue
        if is_ignored(name, ignore):          # user --ignore extras only
            plan.append((name, shard, "copy"))
            counts["ignored"] += 1
            continue
        with safe_open(src / shard, framework="pt") as f:
            shape = f.get_slice(name).get_shape()
        reg_kind = spec.classify(name, shape)
        kind = _KIND_TO_PLAN.get(reg_kind, "copy")
        linear_like[name] = (name.endswith(".weight") and len(shape) == 2)
        # Optional block-FP8 attention: re-tag matching attn projections, but
        # ONLY genuine attention (KIND_ATTN). MTP-layer attn classifies as
        # KIND_MTP and is copied/ignored at runtime — fp8'ing it would break the
        # loader, so leave it copied.
        if (do_fp8_attn and reg_kind == KIND_ATTN
                and any(r.search(name) for r in spec.fp8_attn_res)):
            kind = "fp8_attn"
        # --overrideLayers: force a normally-copied module into the RXF grid.
        # Only 2-D .weight tensors are quantizable; anything else (1-D norm/bias,
        # non-.weight) that matches is left copied and flagged.
        if kind == "copy" and is_ignored(name, args.overrideLayers):
            if linear_like[name]:
                kind = "linear"
                overridden.append(name)
            else:
                override_skipped.append(name)
        plan.append((name, shard, kind))
        counts[kind] = counts.get(kind, 0) + 1
        # K (contraction dim) is the last dim for both a 2-D linear [out, in] and
        # a stacked 3-D expert [E, *, in]; the rotation needs K %% span == 0.
        if kind in ("linear", "expert_fused") and shape[-1] % rotation_span != 0:
            rot_bad.append((name, shape[-1]))

    # ── Rotation pre-flight: it's a single config-level flag (all-or-nothing for
    # the checkpoint), so if ANY quantize target fails K %% span, fall back to
    # no-rotation for the whole model rather than producing an unloadable group. ──
    if rotate and rot_bad:
        ex = ", ".join(f"{n} (K={k})" for n, k in rot_bad[:3])
        more = f" +{len(rot_bad) - 3} more" if len(rot_bad) > 3 else ""
        print(f"  ROTATION DISABLED: {len(rot_bad)} quantize target(s) have "
              f"K %% {rotation_span} != 0 — e.g. {ex}{more}. Quantizing without "
              f"rotation.")
        rotate = False
    if rotate and rotation_kind == "givens" and fp16_experts:
        # The learned-Givens runtime path is wired for dense linear modules; the
        # protected-fp16-expert rotation (rotate_fp16_weight) still applies the
        # Hadamard FWHT, which would not match a Givens R. Block the combination
        # loudly rather than emit a silently-wrong checkpoint.
        raise ValueError(
            "--rotation-kind givens with --fp16-experts is not yet supported "
            "(protected experts would be Hadamard-rotated, not Givens-rotated). "
            "Use --rotation-kind hadamard, or drop --fp16-experts.")
    set_rotation(rotate, rotation_span, rotation_kind)
    _rot_desc = (f"{ROTATION_NAME} "
                 + ("(learned Givens)" if rotation_kind == "givens"
                    else f"(per-{ROTATION_SPAN} Hadamard)"))
    print(f"rotation: {_rot_desc if rotate else 'off'}")

    print(f"source: {src}")
    print(f"output: {dst}")
    print(f"tensors: {len(idx)}  | {counts}")

    # ── --overrideLayers: rewrite the runtime ignore list ──
    # Forcing copy->linear means the broad spec ignore patterns (e.g.
    # 're:.*\.self_attn\..*', 're:.*\.visual\..*', 're:^lm_head') would still tell
    # the runtime to SKIP the modules we just quantized. Drop every spec pattern
    # that matches an overridden module, then re-add explicit anchored ignores for
    # the Linear-shaped siblings that pattern legitimately covered but that stayed
    # copied — so a partial override (e.g. only q_proj) keeps k/v/o ignored.
    final_ignore = None
    if overridden:
        ov = set(overridden)
        kept, reinstate = [], set()
        for p in spec.runtime_ignore(cfg):
            if any(is_ignored(n, [p]) for n in ov):
                for nm, _, k in plan:
                    # Reinstate every copied .weight the dropped pattern covered
                    # (Linear or not — incl. conv1d / norms), so dropping a broad
                    # pattern only un-ignores the overridden modules and nothing
                    # else. Harmless to list non-Linear modules; the runtime quant
                    # loader only consults ignore for Linear layers.
                    if (k == "copy" and nm.endswith(".weight") and nm not in ov
                            and is_ignored(nm, [p])):
                        reinstate.add(nm[:-len(".weight")])
            else:
                kept.append(p)
        final_ignore = kept + [f"re:^{re.escape(m)}$" for m in sorted(reinstate)]
        print(f"overrideLayers: forced {len(overridden)} module(s) copy->RXF "
              f"(quantized); rewrote ignore "
              f"({len(spec.runtime_ignore(cfg))} spec -> {len(final_ignore)} "
              f"patterns, {len(reinstate)} sibling(s) reinstated)")
    if override_skipped:
        print(f"  WARNING: --overrideLayers matched {len(override_skipped)} "
              f"non-quantizable tensor(s) (not 2-D .weight); left copied — "
              f"e.g. {override_skipped[0]}")

    if args.dry_run:
        for name, _, kind in plan:
            if kind in ("linear", "expert_fused", "fp8_attn"):
                print(f"  {kind:13s}  {name}")
        return

    # ── Device setup ──
    use_gpu = args.gpu
    n_gpus = torch.cuda.device_count() if use_gpu else 0
    if use_gpu and n_gpus == 0:
        print("  WARNING: --gpu requested but no GPUs found, falling back to CPU")
        use_gpu = False
    elif use_gpu:
        print(f"  GPU mode: {n_gpus} GPU(s) available")

    # ── Learned Givens rotation: fit ONE model-wide R before quantizing, so every
    # module rotates its weight by the same R and the runtime applies the matching
    # activation rotation (set into config.json as rotation_matrix). ──
    rotation_matrix = None
    if rotate and rotation_kind == "givens":
        if act_blocks is not None and act_blocks.numel():
            # ── Stage (c): fit R to the REAL per-token int8 ACTIVATION-quant error
            # on calibration activations (activation conditioning — the DFlash
            # insight; the genuine edge over data-blind Hadamard). The fit runs on
            # CPU: the descent is cheap and the GPU large-M small-K matmul flake
            # (PAROQUANT_RXF_INTEGRATION.md §6) does not bite there. ──
            print(f"  fitting model-wide learned Givens R (span={ROTATION_SPAN}, "
                  f"objective=ACTIVATION int8-quant error, stage c) on cpu ...",
                  flush=True)
            ab = act_blocks.float().cpu()
            # Hadamard baseline (the init) vs fitted R, on the SAME blocks.
            H = _fwht_rows(torch.eye(ROTATION_SPAN, dtype=torch.float32),
                           ROTATION_SPAN)
            mse_had = _act_int8_quant_mse(_apply_rotation_rows(ab, H)).item()
            R = fit_givens_rotation(ab, ACTIVE_TABLE, span=ROTATION_SPAN,
                                    score="activation").cpu()
            mse_fit = _act_int8_quant_mse(_apply_rotation_rows(ab, R)).item()
            set_givens_rotation(R)
            rotation_matrix = R.tolist()
            eye = torch.eye(ROTATION_SPAN)
            ortho = (R @ R.t() - eye).abs().max().item()
            gain = mse_had / mse_fit if mse_fit > 0 else float("inf")
            print(f"  learned R (activation obj): {ab.shape[0]} blocks, "
                  f"int8 act-MSE Hadamard={mse_had:.4e} -> fit={mse_fit:.4e} "
                  f"({gain:.4f}x), R·Rᵀ−I max={ortho:.2e}", flush=True)
        else:
            # ── Stage (b): data-blind weight-quant MSE objective (measured null). ──
            fit_dev = "cuda:0" if (use_gpu and n_gpus > 0) else "cpu"
            print(f"  fitting model-wide learned Givens R (span={ROTATION_SPAN}, "
                  f"objective=weight-quant MSE, stage b; "
                  f"importance={'on' if act_scales else 'off (uniform)'}) "
                  f"on {fit_dev} ...", flush=True)
            nl_fit = ACTIVE_TABLE.to(fit_dev)
            blocks, imp_block = pool_rotation_blocks(
                plan, src, idx, spec, ROTATION_SPAN, act_scales, device=fit_dev)
            R = fit_givens_rotation(blocks, nl_fit, imp_block=imp_block,
                                    span=ROTATION_SPAN).cpu()
            set_givens_rotation(R)
            rotation_matrix = R.tolist()
            eye = torch.eye(ROTATION_SPAN)
            ortho = (R @ R.t() - eye).abs().max().item()
            print(f"  learned R: pooled {blocks.shape[0]} blocks, "
                  f"R·Rᵀ−I max={ortho:.2e}", flush=True)

    # ==================================================================
    # Quantization pass
    # ==================================================================
    print(f"\n=== Quantizing to RXF ({ACTIVE_NAME} grid + fp16 group-{GROUP} "
          f"scale, act_dtype={args.act_dtype}) ===")
    dst.mkdir(parents=True, exist_ok=True)

    cfg["quantization_config"] = build_quant_config(
        spec, cfg, ignore, fp8_attn=do_fp8_attn, fp16_experts=fp16_experts,
        ignore_override=final_ignore, act_dtype=args.act_dtype,
        rotation_matrix=rotation_matrix)
    json.dump(cfg, open(dst / "config.json", "w"), indent=2)
    for fn in ("tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja",
               "processor_config.json", "configuration.json",
               "preprocessor_config.json", "video_preprocessor_config.json"):
        if (src / fn).exists():
            shutil.copy2(src / fn, dst / fn)

    out_index   = {}
    out_tensors = {}
    shard_id    = 0
    cur_bytes   = 0
    total_bytes = 0

    def flush(force=False):
        nonlocal out_tensors, shard_id, cur_bytes
        if not out_tensors or (not force and cur_bytes < SHARD_BYTES):
            return
        fn = f"model-{shard_id:05d}.safetensors"
        save_file(out_tensors, dst / fn, metadata={"format": "pt"})
        for k in out_tensors:
            out_index[k] = fn
        out_tensors = {}
        shard_id += 1
        cur_bytes = 0
        gc.collect()

    def add(n, t):
        nonlocal cur_bytes, total_bytes
        t = t.contiguous()
        out_tensors[n] = t
        b = t.numel() * t.element_size()
        cur_bytes += b
        total_bytes += b

    nq = 0
    n_fp16 = 0
    total_mse = 0.0
    total_groups = 0
    total_starved = 0
    fp16_bytes = 0

    mse_csv_file = None
    mse_csv_rows_written = 0
    if args.savegroupmse:
        csv_path = dst / "group_mse.csv"
        mse_csv_file = open(csv_path, "w")
        mse_csv_file.write("module,group_id,mse\n")

    def record_mse(mod_name, mse_flat):
        nonlocal mse_csv_rows_written
        if mse_csv_file is None:
            return
        for gid, val in enumerate(mse_flat.float().tolist()):
            mse_csv_file.write(f"{mod_name},{gid},{val:.8e}\n")
            mse_csv_rows_written += 1

    def _get_importance(mod_name, N, K):
        """Look up activation importance for a module. Returns [N,K] or None."""
        if not act_scales:
            return None
        # act_scales keys are transformer module names, mod_name may have
        # weight suffixes stripped. Try exact match and common variants.
        for candidate in [mod_name, mod_name + ".weight",
                          mod_name.replace(".weight_packed", ""),
                          mod_name.replace(".weight", "")]:
            if candidate in act_scales:
                imp_k = act_scales[candidate]  # [K]
                return imp_k.unsqueeze(0).expand(N, -1)[:, :K]
        return None

    def _process_module(mod_name, w):
        """Quantize one 2-D module OR pass it through fp16-protected.
        Returns (tensors_dict, mse_flat | None, starved). Protected experts
        emit only weight_fp16 (pre-rotated under rotation) — no packed/scale
        tensors (the runtime never reads them)."""
        if _protected(mod_name):
            return {"weight_fp16": rotate_fp16_weight(w)}, None, 0
        imp = _get_importance(mod_name, w.shape[0], w.shape[1])
        return rxf_quantize(w, importance=imp)

    def collect(mod_name, tensors, mse_flat, starved):
        """Fold one module's result into the output stream + stats."""
        nonlocal nq, n_fp16, total_mse, total_groups, total_starved, fp16_bytes
        for suf, val in tensors.items():
            add(f"{mod_name}.{suf}", val.cpu())
        if mse_flat is None:
            n_fp16 += 1
            fp16_bytes += sum(v.numel() * v.element_size()
                              for v in tensors.values())
            print(f"    fp16-protected: {mod_name}", flush=True)
            return
        total_mse += mse_flat.float().cpu().sum().item()
        record_mse(mod_name, mse_flat.cpu())
        total_groups += mse_flat.numel()
        if starved:
            total_starved += starved
            print(f"    fp16-scale-underflow: {starved} group(s) in {mod_name} "
                  f"(|scale| < {FP16_SCALE_UNDERFLOW:.2e})", flush=True)
        nq += 1

    # ── GPU: quantize one layer's worth of tensors on a single device ──
    def _gpu_quantize_layer(layer_work, gpu_id):
        """Quantize all (mod_name, weight) pairs on one GPU.
        Returns list of (mod_name, tensors_dict, mse_flat, starved)."""
        device = f"cuda:{gpu_id}"
        results = []
        for mod_name, w in layer_work:
            if _protected(mod_name):
                # fp16 pass-through: rotation runs on CPU, no GPU round-trip.
                results.append(
                    (mod_name, {"weight_fp16": rotate_fp16_weight(w)},
                     None, 0))
                continue
            w_gpu = w.to(device)
            imp = _get_importance(mod_name, w.shape[0], w.shape[1])
            tensors, mse_flat, starved = rxf_quantize(w_gpu, importance=imp)
            del w_gpu
            results.append((mod_name, tensors, mse_flat, starved))
        torch.cuda.empty_cache()
        return results

    # ── Group quantizable tensors by layer (plan only; weights loaded lazily) ──
    layer_plan = defaultdict(list)   # layer_idx -> [(name, shard, kind)]
    copy_entries = []
    fp8_attn_entries = []
    for name, shard, kind in plan:
        if kind == "copy":
            copy_entries.append((name, shard))
        elif kind == "fp8_attn":
            fp8_attn_entries.append((name, shard))
        else:
            layer_plan[spec.layer_index(name)].append((name, shard, kind))

    def _load_layers(layer_indices):
        """Read the weights for the given layers from disk into CPU RAM.
        Returns {layer_idx: [(mod_name, w2d)]}. Fused experts are split here."""
        work = {}
        for li in layer_indices:
            items = []
            for name, shard, kind in layer_plan[li]:
                with safe_open(src / shard, framework="pt") as f:
                    w = f.get_tensor(name)
                if kind == "linear":
                    mod = name[:-7] if name.endswith(".weight") else name
                    items.append((mod, w))
                else:   # expert_fused
                    for mn, w2d in spec.split_experts(name, w):
                        items.append((mn, w2d))
                    del w
            work[li] = items
        return work

    # ── Copy-through (streamed one tensor at a time) ──
    for name, shard in copy_entries:
        if not args.quant_only:
            with safe_open(src / shard, framework="pt") as f:
                add(name, f.get_tensor(name))
            flush()

    # ── Block-FP8 attention (streamed one tensor at a time) ──
    if fp8_attn_entries:
        fp8_dev = "cuda:0" if (use_gpu and n_gpus > 0) else "cpu"
        fblock = tuple(spec.fp8_attn_block)
        print(f"  block-FP8 attention: {len(fp8_attn_entries)} tensors "
              f"(e4m3, block={list(fblock)}) on {fp8_dev}", flush=True)
        for name, shard in fp8_attn_entries:
            with safe_open(src / shard, framework="pt") as f:
                w = f.get_tensor(name)
            wq, scale = fp8_block_quantize(w.to(fp8_dev), fblock)
            add(name, wq.cpu())                       # <name>.weight (fp8)
            add(name + "_scale_inv", scale.cpu())     # <name>.weight_scale_inv
            del w, wq, scale
            flush()
        if fp8_dev != "cpu":
            torch.cuda.empty_cache()

    sorted_layers = sorted(layer_plan.keys())
    print(f"  {len(sorted_layers)} layers to quantize, "
          f"{sum(len(v) for v in layer_plan.values())} tensor groups")

    if use_gpu and args.gpu_stream:
        # ── GPU-stream: double-buffer disk->RAM load against GPU quantize ──
        n_chunks = max(1, args.load_chunks)
        k = math.ceil(len(sorted_layers) / n_chunks)
        chunks = [sorted_layers[i:i + k]
                  for i in range(0, len(sorted_layers), k)]
        print(f"  [GPU-STREAM] {len(chunks)} chunk(s) of <= {k} layers, "
              f"prefetched (~2 chunks resident in host RAM), "
              f"{n_gpus} GPU(s)", flush=True)

        prefetch = queue.Queue(maxsize=1)   # one chunk waits while current runs

        def _loader():
            for ci, chunk in enumerate(chunks):
                work = _load_layers(chunk)        # disk -> RAM, overlaps GPU work
                prefetch.put((ci, chunk, work))   # blocks if a chunk already waits
            prefetch.put(None)

        loader_thread = threading.Thread(target=_loader, daemon=True)
        loader_thread.start()

        while True:
            item = prefetch.get()
            if item is None:
                break
            ci, chunk, work = item
            for batch_start in range(0, len(chunk), n_gpus):
                batch_layers = chunk[batch_start:batch_start + n_gpus]
                with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                    futures = {}
                    for gpu_id, layer_idx in enumerate(batch_layers):
                        fut = pool.submit(
                            _gpu_quantize_layer, work[layer_idx], gpu_id)
                        futures[fut] = layer_idx
                    for fut in as_completed(futures):
                        for mod_name, tensors, mse_flat, starved in fut.result():
                            collect(mod_name, tensors, mse_flat, starved)
                        flush()
                for li in batch_layers:          # free RAM as each layer finishes
                    work.pop(li, None)
                gc.collect()
            del work
            gc.collect()
            print(f"  [GPU-STREAM] chunk {ci} done, {nq} modules total",
                  flush=True)
        loader_thread.join()

    elif use_gpu and args.gpu_all:
        layer_work = _load_layers(sorted_layers)
        # ── GPU-all: load all layers to GPUs round-robin, quantize in parallel ──
        print(f"  [GPU-ALL] assigning {len(sorted_layers)} layers across "
              f"{n_gpus} GPUs", flush=True)

        # Assign layers to GPUs round-robin and pre-load weights
        gpu_assignments = defaultdict(list)  # gpu_id -> [(layer_idx, work)]
        for i, layer_idx in enumerate(sorted_layers):
            gpu_id = i % n_gpus
            gpu_assignments[gpu_id].append((layer_idx, layer_work[layer_idx]))

        def _gpu_quantize_all(gpu_work, gpu_id):
            """Quantize all assigned layers on one GPU.
            Results stay in GPU memory until the main thread fetches them."""
            device = f"cuda:{gpu_id}"
            total_mods = sum(len(w) for _, w in gpu_work)
            done = 0
            all_results = []
            for li_done, (layer_idx, work) in enumerate(gpu_work, 1):
                for mod_name, w in work:
                    if _protected(mod_name):
                        all_results.append(
                            (layer_idx, mod_name,
                             {"weight_fp16": rotate_fp16_weight(w)}, None, 0))
                        done += 1
                        continue
                    w_gpu = w.to(device)
                    imp = _get_importance(mod_name, w.shape[0], w.shape[1])
                    tensors, mse_flat, starved = rxf_quantize(
                        w_gpu, importance=imp)
                    del w_gpu
                    all_results.append(
                        (layer_idx, mod_name, tensors, mse_flat, starved))
                    done += 1
                vram = torch.cuda.memory_allocated(device) / 1e9
                print(f"  [GPU-ALL] GPU {gpu_id}: layer {layer_idx} done "
                      f"({li_done}/{len(gpu_work)} layers, {done}/{total_mods} "
                      f"modules, {vram:.1f} GB VRAM)", flush=True)
            return all_results

        with ThreadPoolExecutor(max_workers=n_gpus) as pool:
            futures = {}
            for gpu_id, gpu_work in gpu_assignments.items():
                n_mods = sum(len(w) for _, w in gpu_work)
                n_layers = len(gpu_work)
                print(f"    GPU {gpu_id}: {n_layers} layers, "
                      f"{n_mods} modules", flush=True)
                fut = pool.submit(_gpu_quantize_all, gpu_work, gpu_id)
                futures[fut] = gpu_id

            # Collect one GPU at a time — pull to CPU, write, free
            for fut in as_completed(futures):
                gpu_id = futures[fut]
                results = fut.result()
                for layer_idx, mod_name, tensors, mse_flat, starved in results:
                    collect(mod_name, tensors, mse_flat, starved)
                del results
                torch.cuda.empty_cache()
                flush()
                print(f"  [GPU-ALL] GPU {gpu_id} done, "
                      f"{nq} modules total", flush=True)

        del gpu_assignments
        layer_work.clear()
        gc.collect()

    elif use_gpu:
        layer_work = _load_layers(sorted_layers)
        # ── GPU path: 1 layer per GPU, n_gpus layers in parallel ──
        for batch_start in range(0, len(sorted_layers), n_gpus):
            batch_layers = sorted_layers[batch_start:batch_start + n_gpus]

            with ThreadPoolExecutor(max_workers=n_gpus) as pool:
                futures = {}
                for gpu_id, layer_idx in enumerate(batch_layers):
                    fut = pool.submit(
                        _gpu_quantize_layer, layer_work[layer_idx], gpu_id)
                    futures[fut] = layer_idx

                for fut in as_completed(futures):
                    for mod_name, tensors, mse_flat, starved in fut.result():
                        collect(mod_name, tensors, mse_flat, starved)
                    flush()

            for li in batch_layers:
                del layer_work[li]
            gc.collect()
            print(f"  [GPU] layers {batch_layers[0]}-{batch_layers[-1]}: "
                  f"{nq} modules done", flush=True)
    else:
        # ── CPU path: per-layer RAM-budgeted streaming quantize ───────────────
        # A prefetch thread reads one layer's tensors disk->RAM up to a budget
        # (90% of *available* RAM, minus the 5GB pack buffer and one quant's
        # interval-enumeration scratch). Compute pulls a tensor and quantizes
        # each of its targets one at a time with all cores (ATen intra-op
        # threads), freeing the FP16 the moment its packed+scale result exists,
        # so RAM drains as work completes. Hard-capped at one layer: the loader
        # never reads ahead into the next layer.
        import psutil
        SCRATCH_RESERVE = 2 * 1024**3   # ~one active interval-enum scratch + slack

        def _ram_budget():
            avail = psutil.virtual_memory().available
            b = int(avail * 0.9) - SHARD_BYTES - SCRATCH_RESERVE
            return max(b, 256 * 1024**2)   # floor; an oversized single tensor
                                           # still loads via the held==0 rule

        for layer_idx in sorted_layers:
            entries = layer_plan[layer_idx]          # [(name, shard, kind)]
            budget = _ram_budget()
            avail_gb = psutil.virtual_memory().available / 1e9
            print(f"  layer {layer_idx}: {len(entries)} tensors, "
                  f"{avail_gb:.1f} GB avail, prefetch budget "
                  f"{budget/1e9:.1f} GB", flush=True)

            q = queue.Queue()
            cond = threading.Condition()
            held = {"bytes": 0}

            def _loader(entries=entries, budget=budget):
                # Read this layer's tensors in order; block before loading one
                # that would push resident FP16 over budget (unless nothing is
                # held yet, so an oversized single tensor still makes progress).
                for name, shard, kind in entries:
                    with safe_open(src / shard, framework="pt") as f:
                        w = f.get_tensor(name)
                    nbytes = w.numel() * w.element_size()
                    with cond:
                        while held["bytes"] > 0 and held["bytes"] + nbytes > budget:
                            cond.wait()
                        held["bytes"] += nbytes
                    q.put((name, kind, w, nbytes))
                q.put(None)

            loader = threading.Thread(target=_loader, daemon=True)
            loader.start()

            while True:
                item = q.get()
                if item is None:
                    break
                name, kind, w, nbytes = item
                if kind == "linear":
                    mod = name[:-7] if name.endswith(".weight") else name
                    targets = [(mod, w)]
                else:                                 # expert_fused: split lazily
                    targets = spec.split_experts(name, w)
                for mod_name, w2d in targets:
                    tensors, mse_flat, starved = _process_module(mod_name, w2d)
                    collect(mod_name, tensors, mse_flat, starved)
                    del w2d
                    flush()                           # final output: 5GB shards
                del w, targets, item
                with cond:                            # free this tensor; wake loader
                    held["bytes"] -= nbytes
                    cond.notify()
                gc.collect()

            loader.join()
            print(f"  layer {layer_idx}: {nq} modules done", flush=True)
            gc.collect()

    flush(force=True)

    # ── Close per-group MSE CSV ──
    if mse_csv_file is not None:
        mse_csv_file.close()
        print(f"wrote per-group MSE to {csv_path} ({mse_csv_rows_written:,} rows)")

    # ── Write final index ──
    json.dump({"metadata": {"total_size": total_bytes},
               "weight_map": out_index},
              open(dst / "model.safetensors.index.json", "w"))

    # ── Summary ──
    mean_mse = total_mse / total_groups if total_groups > 0 else 0.0
    print(f"\nquantized {nq} modules -> {dst}")
    print(f"mean per-group MSE (vs snapped scale): {mean_mse:.4e}")
    print(f"total groups: {total_groups:,}")
    print(f"fp16-scale-underflow groups (|scale| < {FP16_SCALE_UNDERFLOW:.2e}): "
          f"{total_starved:,}")
    if fp16_experts:
        print(f"fp16-protected expert modules: {n_fp16} "
              f"({fp16_bytes / 1e6:.1f} MB, ~3.76x their quantized size)")
    print(f"format: rxf-pack-quantized ({ACTIVE_NAME} grid + fp16 "
          f"group-{GROUP} scale)")
    print("bits per weight: 4.5 (4-bit code + fp16 scale / 32)")
    model_bytes = total_bytes
    if nq > 0:
        print(f"output size: {model_bytes / 1e9:.2f} GB")

    # ── Static eval metrics -> output dir (default on; never voids the run) ──
    if args.eval and nq > 0:
        try:
            import metrics_step
            out_file = dst / "eval_metrics.txt"
            print(f"\n=== eval metrics -> {out_file} ===")
            metrics_step.run_metrics(src, dst, out_file=out_file)
        except Exception as e:
            print(f"  WARNING: eval metrics failed ({e!r}); "
                  f"quantized output is unaffected.")


if __name__ == "__main__":
    main()
