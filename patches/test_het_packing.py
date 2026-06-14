"""Pin the packed/grouped-axis offset convention for heterogeneous TP weight loading.

The risk (HET_TP_PATCH.md §4a / §8): vLLM's v2 loaders narrow `loaded_weight` in
*param-axis units* (`self.data.shape[dim]`), NOT logical element units. For 4-bit
weights the sharded axis is often PACKED (8 int4 per int32) and scales are GROUPED
(one per `group_size`). So a het offset computed in logical elements would land at the
wrong byte and silently corrupt weights. This test fixes the conversion:

    axis_total = logical_total // units_per_elt        # units_per_elt = pack(8) or group
    align_axis = align_logical // units_per_elt        # align_logical = group_size (128)
    offset_axis = partition_offsets(axis_total, weights, align_axis)[rank]

and asserts, on CPU (no torch/GPU), that for every realistic AWQ / compressed-tensors
layout the het shards: partition the axis exactly (gapless, non-overlapping), stay
consistent with the logical split (offset_axis * units == offset_logical), reconstruct
the full tensor, and degenerate to the stock even split when weights are equal.

Run:  python patches/test_het_packing.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from het_tp import partition_offsets, partition_sizes  # noqa: E402

GROUP = 128          # AWQ-INT4 group_size; also our het `align_logical`
PACK = 8             # int4 nibbles per int32
W = (64, 56)         # the 9070 XT : 9070 CU split


def axis_units(total_logical: int, units_per_elt: int) -> int:
    assert total_logical % units_per_elt == 0
    return total_logical // units_per_elt


def het_axis_offsets(total_logical, units_per_elt, align_logical, weights):
    """The proposed loader arithmetic, in PARAM-AXIS units."""
    a_total = axis_units(total_logical, units_per_elt)
    a_align = align_logical // units_per_elt
    assert align_logical % units_per_elt == 0, (
        f"align_logical={align_logical} must be divisible by units_per_elt="
        f"{units_per_elt}; pick align = group_size (a multiple of pack & group)."
    )
    return partition_offsets(a_total, weights, a_align), a_align


def check_axis(name, total_logical, units_per_elt, align_logical=GROUP, weights=W):
    """One sharded axis: assert het offsets are correct in axis units and consistent
    with the logical split, and reconstruct the axis."""
    # logical split (what the *math* sees)
    log_off = partition_offsets(total_logical, weights, align_logical)
    log_sz = partition_sizes(total_logical, weights, align_logical)
    # axis split (what the *loader* must emit)
    ax_off, a_align = het_axis_offsets(total_logical, units_per_elt, align_logical, weights)
    a_total = axis_units(total_logical, units_per_elt)

    # 1) axis shards partition [0, a_total) exactly
    assert ax_off[0] == 0 and ax_off[-1] == a_total, (name, ax_off, a_total)
    for r in range(len(weights)):
        assert ax_off[r + 1] > ax_off[r], (name, "empty shard", ax_off)

    # 2) logical<->axis consistency: axis offset * units == logical offset
    for r in range(len(weights) + 1):
        assert ax_off[r] * units_per_elt == log_off[r], (
            name, "logical/axis mismatch", r, ax_off[r] * units_per_elt, log_off[r])
    for r in range(len(weights)):
        ax_sz = ax_off[r + 1] - ax_off[r]
        assert ax_sz * units_per_elt == log_sz[r], (name, "size mismatch", r)

    # 3) reconstruction: narrow a concrete axis-length buffer per rank, concat == whole
    buf = list(range(a_total))
    recon = []
    for r in range(len(weights)):
        recon += buf[ax_off[r]: ax_off[r + 1]]
    assert recon == buf, (name, "reconstruction failed")

    # 4) group alignment: every shard boundary lands on a whole group (scales valid)
    for r in range(len(weights) + 1):
        assert log_off[r] % GROUP == 0, (name, "shard crosses a quant group", log_off[r])

    print(f"  ok  {name:<46} logical={log_sz} axis_align={a_align}")


def check_even_equivalence(total_logical, units_per_elt):
    """weights=(1,1) must reproduce the stock even loader: start == tp_rank*shard_size."""
    ax_off, _ = het_axis_offsets(total_logical, units_per_elt, GROUP, (1, 1))
    a_total = axis_units(total_logical, units_per_elt)
    shard = a_total // 2
    for rank in range(2):
        assert ax_off[rank] == rank * shard, (total_logical, units_per_elt, ax_off)


def main():
    INTER = 11008  # FFN intermediate (the dim we split)
    HID = 4096

    print("AWQ layout (qweight (K, N//8): OUTPUT/N is packed):")
    # gate_up_proj = ColumnParallel, splits OUTPUT N=intermediate, packed along N
    check_axis("awq col gate_up: qweight N (packed)", INTER, PACK)
    check_axis("awq col gate_up: scales  N (unpacked)", INTER, 1)
    check_axis("awq col gate_up: qzeros  N (packed)", INTER, PACK)
    # down_proj = RowParallel, splits INPUT K=intermediate; K unpacked in AWQ qweight,
    # but scales/zeros are grouped along K (one row per group)
    check_axis("awq row down: qweight K (unpacked)", INTER, 1)
    check_axis("awq row down: scales  K//group", INTER, GROUP)
    check_axis("awq row down: qzeros  K//group", INTER, GROUP)

    print("compressed-tensors layout (w_q (N, K//8): INPUT/K is packed):")
    check_axis("ct col gate_up: w_q N (unpacked)", INTER, 1)
    check_axis("ct row down: w_q K (packed)", INTER, PACK)
    check_axis("ct row down: scales K//group", INTER, GROUP)

    print("cross-layer invariant (gate_up N-split == down K-split, logical):")
    # the §0 invariant: both layers must place intermediate channel i on the same rank
    gate_up_logical = partition_offsets(INTER, W, GROUP)   # used for N (gate_up output)
    down_logical = partition_offsets(INTER, W, GROUP)      # used for K (down input)
    assert gate_up_logical == down_logical, (gate_up_logical, down_logical)
    print(f"  ok  identical logical boundaries {gate_up_logical}")

    print("even-equivalence (VLLM_TP_CU_WEIGHTS unset / '1,1'):")
    for units in (1, PACK, GROUP):
        for total in (4096, 8192, 11008, 14336):
            if total % (units * 2):
                continue
            check_even_equivalence(total, units)
    print("  ok  equal weights reproduce tp_rank*shard_size")

    print("\nALL het-packing convention tests passed")


if __name__ == "__main__":
    main()
