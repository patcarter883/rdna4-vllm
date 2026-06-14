"""Heterogeneous tensor-parallel sharding for vLLM on asymmetric multi-GPU rigs.

Standard vLLM TP splits every parallel dimension evenly across ``tp_size`` ranks
(``vllm.distributed.utils.divide`` hard-asserts divisibility). On a heterogeneous
rig -- e.g. an RX 9070 XT (64 CU) paired with an RX 9070 (56 CU) -- an even split
makes the bigger card finish its shard first and spin-wait at the all-reduce
barrier (the het-TP "sync bubble", observed in DIARY.md as the rank1 COMM=61%
artifact). This module computes a *proportional* split (e.g. 64:56) for the FFN /
MoE intermediate dimensions so both cards reach the barrier together.

Scope -- deliberately narrow (see patches/HET_TP_PATCH.md):
  * Only the *reduction* dimensions are split proportionally: the FFN intermediate
    (gate_up output / down input) and the MoE intermediate (w13 output / w2 input).
    Their row-parallel layer all-reduces a FULL hidden-dim tensor on every rank, so
    collectives stay equal-sized -- proportional splitting is provably safe there.
  * Attention Q/KV heads and the lm_head vocab stay EVEN. Heads are coarse integers
    (can't hit 64:56 with few KV heads); the lm_head logit all-*gather* needs equal
    shards. Attention is ~1.8% of decode, so leaving it even costs ~nothing.

Correctness invariant: gate_up's output split and down's input split index the SAME
intermediate channels, so they MUST use the identical partition. Always call
``partition_sizes(intermediate, weights, align=group_size)`` for both -- the result
is deterministic, so the pair stays consistent. ``align`` = the quant ``group_size``
(128 for AWQ-INT4), which is itself a multiple of the int4 pack factor (8) and the
WMMA/Triton N-tile, so every shard remains a valid W4A8 kernel input.

This file is pure Python (no torch/vllm import at module load) so the apportionment
math is unit-testable on CPU:  ``python patches/het_tp.py``
"""
from __future__ import annotations

import functools
import math
import os

# Env override, e.g. VLLM_TP_CU_WEIGHTS="64,56" (one weight per TP rank, in rank order).
# Unset / malformed -> get_cu_weights returns None -> callers fall back to even divide.
_ENV = "VLLM_TP_CU_WEIGHTS"


@functools.lru_cache(maxsize=1)
def get_cu_weights(tp_size: int | None = None) -> tuple[int, ...] | None:
    """Per-rank compute weights (CU counts) for proportional TP.

    Source: ``$VLLM_TP_CU_WEIGHTS`` ("64,56"). Returns a tuple of length tp_size, or
    None when unset/invalid (=> caller uses the stock even split). Cached because the
    rig topology is fixed for the process lifetime.
    """
    raw = os.environ.get(_ENV)
    if not raw:
        return None
    try:
        weights = tuple(int(x) for x in raw.split(",") if x.strip())
    except ValueError:
        return None
    if not weights or any(w <= 0 for w in weights):
        return None
    if tp_size is not None and len(weights) != tp_size:
        # Misconfiguration (weights don't match the world size): fail safe to even.
        return None
    return weights


def partition_sizes(total: int, weights, align: int = 1) -> list[int]:
    """Split ``total`` into ``len(weights)`` parts proportional to ``weights``, each a
    multiple of ``align``, summing *exactly* to ``total``.

    Largest-remainder (Hamilton) apportionment over ``total // align`` indivisible
    tiles. Leftover tiles go to the ranks with the largest fractional remainder, ties
    broken toward the larger weight (favour the bigger card). Deterministic: same
    inputs -> same output, which is what keeps the gate_up/down split consistent.
    """
    if align <= 0:
        raise ValueError(f"align must be positive, got {align}")
    if total % align != 0:
        raise ValueError(f"total={total} is not a multiple of align={align}")
    n = len(weights)
    units = total // align
    if units < n:
        raise ValueError(
            f"total//align={units} < num_ranks={n}: dimension too small to split "
            f"at this alignment; keep this layer on the even path."
        )
    wsum = sum(weights)
    ideal = [units * w / wsum for w in weights]
    floor = [int(x) for x in ideal]
    leftover = units - sum(floor)
    order = sorted(
        range(n),
        key=lambda i: (ideal[i] - floor[i], weights[i]),
        reverse=True,
    )
    for i in range(leftover):
        floor[order[i]] += 1
    return [f * align for f in floor]


def partition_offsets(total: int, weights, align: int = 1) -> list[int]:
    """Cumulative start offsets matching :func:`partition_sizes`.

    Length ``len(weights) + 1``; ``offsets[rank]`` is this rank's start index and
    ``offsets[rank+1]`` its end (exclusive). ``offsets[-1] == total``.
    """
    sizes = partition_sizes(total, weights, align)
    offs = [0]
    for s in sizes:
        offs.append(offs[-1] + s)
    return offs


def het_size(total: int, rank: int, tp_size: int, align: int = 1, weights=None) -> int:
    """This rank's shard size of ``total`` -- proportional if weights are configured,
    else the stock even ``divide(total, tp_size)``. Use in layer ``__init__`` to size
    ``*_per_partition`` (and hence the parameter allocation)."""
    weights = weights or get_cu_weights(tp_size)
    if weights is None:
        return total // tp_size  # mirrors vllm.distributed.utils.divide (caller pre-asserts)
    return partition_sizes(total, weights, align)[rank]


def het_offset(total: int, rank: int, tp_size: int, align: int = 1, weights=None) -> int:
    """Start index of this rank's shard of ``total`` -- proportional if configured,
    else ``rank * (total // tp_size)``. Use in weight loaders in place of
    ``self.tp_rank * shard_size``."""
    weights = weights or get_cu_weights(tp_size)
    if weights is None:
        return rank * (total // tp_size)
    return partition_offsets(total, weights, align)[rank]


# --- Runtime integration helpers (used by the in-tree vLLM edits) -------------

# Logical alignment floor: a multiple of Marlin min_thread_k(128) — which also covers
# min_thread_n(64), the WMMA 16/64 tile, and the int4 pack factor(8). The actual align
# is lcm(this, group_size) so shards are also whole quant groups.
HET_ALIGN_FLOOR = 128


def het_align(group_size: int | None) -> int:
    """Logical shard alignment: ``lcm(group_size, 128)``. For AWQ g32 -> 128; g128 ->
    128; g256 -> 256. Guarantees every shard is a whole number of quant groups AND
    satisfies the Marlin/WMMA tile floor (so awq_marlin stays valid too — §11)."""
    g = group_size if (group_size and group_size > 0) else HET_ALIGN_FLOOR
    return math.lcm(g, HET_ALIGN_FLOOR)


def het_eligible(prefix: str) -> bool:
    """True only for FFN/MoE reduction-dim layers. Attention heads + vocab/lm_head
    stay on the even path (coarse integers / equal-shard gather). See §2."""
    p = prefix.lower()
    if any(k in p for k in ("attn", "lm_head", "embed", "vocab")):
        return False
    return any(k in p for k in ("mlp", "experts", "feed_forward", "ffn",
                                "gate_up", "gate_proj", "up_proj", "down_proj"))


def het_active(tp_size: int) -> bool:
    """Master gate: het only engages when CU weights are configured. When unset every
    caller falls back to the stock even split (byte-identical to upstream)."""
    return get_cu_weights(tp_size) is not None


def het_axis_offset(logical_total: int, align: int, axis_total: int,
                    rank: int, tp_size: int, weights=None) -> int:
    """Start index for this rank's shard **in the param axis's own units**.

    The layer-wide split is computed once in logical element units, then this rank's
    logical start is converted to the axis's units by dividing by units-per-element
    ``u = logical_total // axis_total`` (= pack_factor for a packed axis, group_size for
    a grouped/scale axis, 1 for a plain axis). This single formula handles qweight,
    qzeros and scales without per-param packing knowledge. Falls back to the even
    offset when weights are unset.
    """
    weights = weights or get_cu_weights(tp_size)
    if weights is None:
        return rank * (axis_total // tp_size)
    if logical_total % axis_total != 0:
        raise ValueError(f"logical_total={logical_total} not divisible by axis_total={axis_total}")
    u = logical_total // axis_total
    log_start = partition_offsets(logical_total, weights, align)[rank]
    if log_start % u != 0:
        raise ValueError(f"logical offset {log_start} not divisible by units/elt {u}; "
                         f"align ({align}) must be a multiple of u")
    return log_start // u


def het_axis_shard(logical_total: int, align: int, axis_total: int,
                   rank: int, tp_size: int, weights=None) -> tuple[int, int]:
    """``(start, size)`` for this rank's shard, both in the param axis's own units.

    Like :func:`het_axis_offset` but also returns the per-rank size — convenient for
    the MoE expert loaders which need both. Falls back to the even ``(per*rank, per)``
    when weights are unset. ``u = logical_total // axis_total`` converts the logical
    apportionment to axis units (pack/group/1)."""
    weights = weights or get_cu_weights(tp_size)
    if weights is None:
        per = axis_total // tp_size
        return per * rank, per
    if logical_total % axis_total != 0:
        raise ValueError(f"logical_total={logical_total} not divisible by axis_total={axis_total}")
    u = logical_total // axis_total
    offs = partition_offsets(logical_total, weights, align)
    start, end = offs[rank], offs[rank + 1]
    if start % u or end % u:
        raise ValueError(f"offset not divisible by units/elt {u}; align must be a multiple of u")
    return start // u, (end - start) // u


def _selftest() -> None:
    # 1) Even fallback when unset.
    assert get_cu_weights.cache_clear() is None  # reset cache
    os.environ.pop(_ENV, None)
    get_cu_weights.cache_clear()
    assert get_cu_weights(2) is None
    assert het_size(11008, 0, 2, align=128) == 5504  # even: 11008/2
    assert het_offset(11008, 1, 2, align=128) == 5504

    # 2) 64:56 split of a Llama-ish intermediate, align=group_size=128.
    w = (64, 56)
    sizes = partition_sizes(11008, w, align=128)
    assert sum(sizes) == 11008, sizes
    assert all(s % 128 == 0 for s in sizes), sizes
    assert sizes == [5888, 5120], sizes          # 53.5% / 46.5% ~ ideal 53.3 / 46.7
    offs = partition_offsets(11008, w, align=128)
    assert offs == [0, 5888, 11008], offs

    # 3) Exact-sum + alignment property sweep over realistic dims.
    for total in (4096, 8192, 11008, 14336, 28672):
        for align in (16, 64, 128):
            if total % align:
                continue
            s = partition_sizes(total, w, align)
            assert sum(s) == total, (total, align, s)
            assert all(x % align == 0 for x in s), (total, align, s)
            # bigger card never gets less work
            assert s[0] >= s[1], (total, align, s)

    # 4) Three-way asymmetric rig.
    s3 = partition_sizes(12288, (64, 56, 40), align=128)
    assert sum(s3) == 12288 and all(x % 128 == 0 for x in s3), s3

    # 4b) Safety guarantee: equal weights reproduce the EVEN split bit-for-bit
    # (so VLLM_TP_CU_WEIGHTS="1,1" is a no-op vs stock vLLM -- the §7.2 equiv test).
    for total in (4096, 8192, 11008, 14336):
        for align in (16, 64, 128):
            if total % (align * 2):
                continue
            assert partition_sizes(total, (1, 1), align) == [total // 2, total // 2]

    # 5) Helper wiring with weights passed explicitly (bypasses env/cache).
    assert het_size(11008, 0, 2, align=128, weights=w) == 5888
    assert het_offset(11008, 1, 2, align=128, weights=w) == 5888

    # 6) Too-small-to-split guard.
    try:
        partition_sizes(128, w, align=128)  # only 1 tile, 2 ranks
        raise AssertionError("expected ValueError for indivisible-at-alignment dim")
    except ValueError:
        pass

    # 7) het_align: lcm(group, 128) — g32/g64/g128 -> 128 (the served model is g32).
    assert het_align(32) == 128 and het_align(64) == 128 and het_align(128) == 128
    assert het_align(256) == 256 and het_align(None) == 128 and het_align(-1) == 128

    # 8) het_axis_offset converts logical->axis (pack=8, group=32, plain=1) correctly,
    #    and reduces to the even offset when weights are unset.
    AL = het_align(32)  # 128, the real model's align
    INTER = 11008
    for units, axis_total in ((1, INTER), (8, INTER // 8), (32, INTER // 32)):
        a0 = het_axis_offset(INTER, AL, axis_total, 0, 2, weights=w)
        a1 = het_axis_offset(INTER, AL, axis_total, 1, 2, weights=w)
        assert a0 == 0 and a1 == 5888 // units, (units, a1)
        assert a1 * units == 5888  # logical consistency
    # unset weights => even offset
    assert het_axis_offset(INTER, AL, INTER // 8, 1, 2, weights=None) == (INTER // 8) // 2

    # 9) het_axis_shard returns (start, size) in axis units; matches het_axis_offset
    #    and reduces to even when unset.
    for units, axis_total in ((1, INTER), (8, INTER // 8), (32, INTER // 32)):
        s0, z0 = het_axis_shard(INTER, AL, axis_total, 0, 2, weights=w)
        s1, z1 = het_axis_shard(INTER, AL, axis_total, 1, 2, weights=w)
        assert s0 == 0 and z0 == 5888 // units and s1 == 5888 // units, (units, s0, z0, s1)
        assert z0 + z1 == axis_total and z1 == 5120 // units
    se, ze = het_axis_shard(INTER, AL, INTER // 8, 1, 2, weights=None)
    assert ze == (INTER // 8) // 2 and se == ze  # even

    print("het_tp self-tests passed")


if __name__ == "__main__":
    _selftest()
