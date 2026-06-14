"""Runtime validation of the het-TP loader edits against the REAL vLLM parameter
classes (no model, no GPU, no distributed init needed).

Builds an AWQ-style packed column weight, stamps `param.het`, fakes a 2-rank TP world
by setting tp_rank/tp_size directly, and asserts `load_column_parallel_weight` /
`load_row_parallel_weight` narrow to the correct proportional slice — and that
`VLLM_TP_CU_WEIGHTS="1,1"` reproduces the stock even split.

Run inside the gfx1201 container with parameter.py + het_tp.py mounted over
site-packages (see the docker invocation in the chat). CPU tensors only.
"""
import os
import torch

import vllm.model_executor.parameter as _pm
# No real distributed world here; params read tp rank/size at __init__. Stub them
# (we overwrite p.tp_rank / p.tp_size per-rank right after construction).
_pm.get_tensor_model_parallel_rank = lambda *a, **k: 0
_pm.get_tensor_model_parallel_world_size = lambda *a, **k: 1
from vllm.model_executor.parameter import PackedvLLMParameter, RowvLLMParameter
from vllm.distributed import het_tp


def _mk_packed_param(per_rank_shape, output_dim, input_dim, packed_dim, het):
    p = PackedvLLMParameter(
        data=torch.zeros(per_rank_shape, dtype=torch.int32),
        input_dim=input_dim, output_dim=output_dim,
        packed_dim=packed_dim, packed_factor=8,
        weight_loader=lambda *a, **k: None,
    )
    p.tp_rank = het["rank"]
    p.tp_size = het["tp"]
    if het.get("stamp"):
        p.het = het["stamp"]
    return p


def test_awq_column_packed_N():
    # AWQ qweight (K, N//8): output/N packed along dim 1. Split N proportionally.
    K, N, PACK = 8, 11008, 8
    weights = (64, 56)
    align = het_tp.het_align(32)            # g32 -> 128
    sizes = het_tp.partition_sizes(N, weights, align)   # [5888, 5120]
    offs = het_tp.partition_offsets(N, weights, align)  # [0, 5888, 11008]
    # global checkpoint weight, axis units (K, N//8); fill so each N-col is identifiable
    glob = torch.arange(K * (N // PACK), dtype=torch.int32).reshape(K, N // PACK)

    os.environ["VLLM_TP_CU_WEIGHTS"] = "64,56"
    het_tp.get_cu_weights.cache_clear()
    for rank in (0, 1):
        per_axis = sizes[rank] // PACK
        p = _mk_packed_param((K, per_axis), output_dim=1, input_dim=0, packed_dim=1,
                             het={"rank": rank, "tp": 2, "stamp": (N, align)})
        p.load_column_parallel_weight(glob)
        exp = glob.narrow(1, offs[rank] // PACK, per_axis)
        assert torch.equal(p.data, exp), f"col rank{rank} mismatch"
    print("  ok  AWQ column packed-N: proportional slices correct")

    # reconstruction: shards concatenated == whole
    parts = []
    for rank in (0, 1):
        per_axis = sizes[rank] // PACK
        p = _mk_packed_param((K, per_axis), 1, 0, 1,
                             het={"rank": rank, "tp": 2, "stamp": (N, align)})
        p.load_column_parallel_weight(glob)
        parts.append(p.data)
    assert torch.equal(torch.cat(parts, dim=1), glob), "reconstruction failed"
    print("  ok  AWQ column packed-N: shards reconstruct the full weight")


def test_row_grouped_scale():
    # scales (K//group, N): row-parallel splits K//group (grouped). u = group.
    K, N, G = 11008, 16, 32
    weights = (64, 56)
    align = het_tp.het_align(G)             # 128
    offs = het_tp.partition_offsets(K, weights, align)
    glob = torch.arange((K // G) * N, dtype=torch.int32).reshape(K // G, N)
    os.environ["VLLM_TP_CU_WEIGHTS"] = "64,56"
    het_tp.get_cu_weights.cache_clear()
    for rank in (0, 1):
        per_axis = (offs[rank + 1] - offs[rank]) // G
        p = RowvLLMParameter(data=torch.zeros((per_axis, N), dtype=torch.int32),
                             input_dim=0, weight_loader=lambda *a, **k: None)
        p.tp_rank, p.tp_size = rank, 2
        p.het = (K, align)
        p.load_row_parallel_weight(glob)
        exp = glob.narrow(0, offs[rank] // G, per_axis)
        assert torch.equal(p.data, exp), f"row rank{rank} mismatch"
    print("  ok  row grouped-scale: proportional group slices correct")


def test_merged_column_gate_up():
    # MergedColumnParallelLinear gate_up: one packed param holds [gate|up], each the
    # full intermediate N, het-split. Mirrors the FIXED caller (uses het
    # output_partition_sizes) + load_merged_column_weight's het loaded_weight offset.
    K, N, PACK = 8, 18944, 8
    weights = (64, 56)
    align = het_tp.het_align(128)                 # 128
    os.environ["VLLM_TP_CU_WEIGHTS"] = "64,56"
    het_tp.get_cu_weights.cache_clear()
    # global checkpoint for each sub-output, axis units (K, N//8); distinct contents
    gate = torch.arange(K * (N // PACK), dtype=torch.int32).reshape(K, N // PACK)
    up = gate + 1_000_000
    for rank in (0, 1):
        sizes = het_tp.partition_sizes(N, weights, align)        # logical per-sub
        opp = [sizes[rank], sizes[rank]]                          # output_partition_sizes
        per_pk = sizes[rank] // PACK
        merged = _mk_packed_param((K, 2 * per_pk), output_dim=1, input_dim=0,
                                  packed_dim=1, het={"rank": rank, "tp": 2,
                                                     "stamp": (N, align)})
        for sid, sub in ((0, gate), (1, up)):
            # FIXED caller: het cumulative offset/size (logical), method adjusts packing
            soff = sum(opp[:sid]); ssz = opp[sid]
            merged.load_merged_column_weight(loaded_weight=sub, shard_id=sid,
                                             shard_offset=soff, shard_size=ssz,
                                             tp_rank=rank)
        # expected: each sub-slot holds this rank's het slice of that sub
        off_pk = het_tp.partition_offsets(N, weights, align)[rank] // PACK
        exp_gate = gate.narrow(1, off_pk, per_pk)
        exp_up = up.narrow(1, off_pk, per_pk)
        assert torch.equal(merged.data[:, :per_pk], exp_gate), f"gate rank{rank}"
        assert torch.equal(merged.data[:, per_pk:], exp_up), f"up rank{rank}"
    print("  ok  merged gate_up: het sub-outputs land on matching boundaries")


def test_even_equivalence():
    # VLLM_TP_CU_WEIGHTS unset => het stamp is a no-op == stock tp_rank*shard_size.
    os.environ.pop("VLLM_TP_CU_WEIGHTS", None)
    het_tp.get_cu_weights.cache_clear()
    K, N, PACK = 8, 4096, 8
    glob = torch.arange(K * (N // PACK), dtype=torch.int32).reshape(K, N // PACK)
    for rank in (0, 1):
        per_axis = (N // 2) // PACK
        # stamp present but env unset -> het_axis_offset falls back to even
        p = _mk_packed_param((K, per_axis), 1, 0, 1,
                             het={"rank": rank, "tp": 2, "stamp": (N, 128)})
        p.load_column_parallel_weight(glob)
        exp = glob.narrow(1, rank * per_axis, per_axis)
        assert torch.equal(p.data, exp), f"even rank{rank} mismatch"
    print("  ok  even-equivalence: unset env == stock tp_rank*shard_size")


if __name__ == "__main__":
    test_awq_column_packed_N()
    test_row_grouped_scale()
    test_merged_column_gate_up()
    test_even_equivalence()
    print("\nALL het-loader runtime tests passed")
