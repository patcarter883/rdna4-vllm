"""Isolated grouped-MoE micro-benchmark: our _run_grouped_moe vs stock Triton
fused_experts, per-M, at the Qwen3.6 expert shape. Fast iteration for kernel
tuning (no model load)."""
import sys, time
import torch
import w4a8_fp8_wmma
from w4a8_fp8_wmma.moe_experts import _wna16_moe_to_op_layout, _run_grouped_moe
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def main():
    dev = "cuda"
    torch.manual_seed(0)
    import os as _o
    E = int(_o.environ.get("BENCH_E","128")); hidden = int(_o.environ.get("BENCH_H","2304"))
    inter = int(_o.environ.get("BENCH_I","896")); g = 32; top_k = 8
    N13, K13 = 2 * inter, hidden
    N2, K2 = hidden, inter
    w13 = torch.randint(0, 256, (E, N13, K13 // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device=dev)
    s13 = torch.rand(E, N13, K13 // g, device=dev, dtype=torch.float16) * 0.02 + 0.005
    s2 = torch.rand(E, N2, K2 // g, device=dev, dtype=torch.float16) * 0.02 + 0.005
    z13 = torch.randint(0, 256, (E, N13 // 2, K13 // g), dtype=torch.uint8, device=dev)
    z2 = torch.randint(0, 256, (E, N2 // 2, K2 // g), dtype=torch.uint8, device=dev)
    w13o, s13o, z13o = _wna16_moe_to_op_layout(w13, s13, z13)
    w2o, s2o, z2o = _wna16_moe_to_op_layout(w2, s2, z2)
    qc = int4_w4a16_moe_quant_config(w1_scale=s13, w2_scale=s2, w1_zp=z13,
                                     w2_zp=z2, block_shape=[0, g])
    ver = int(__import__("os").environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_VERSION", "5"))
    print(f"E={E} hidden={hidden} inter={inter} g={g} top_k={top_k} ver={ver} "
          f"dev={torch.cuda.get_device_name(0)}")
    print(f"{'M':>6} {'ours_ms':>9} {'stock_ms':>9} {'ours/stock':>10}")
    import os as _os2
    _ms = _os2.environ.get("BENCH_MS")
    Ms = [int(m) for m in _ms.split(",")] if _ms else [1, 2, 4, 8, 16, 32, 64, 256, 1024]
    for M in Ms:
        x = torch.randn(M, hidden, dtype=torch.float16, device=dev) * 0.5
        tids = torch.stack(
            [torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
        tw = torch.rand(M, top_k, dtype=torch.float32, device=dev)
        ours = lambda: _run_grouped_moe(
            x, w13o, w2o, s13o, s2o, z13o, z2o, tw, tids, MoEActivation.SILU,
            E, None, False, ver, out_dtype=x.dtype)
        stock = lambda: fused_experts(
            x, w13, w2, topk_weights=tw, topk_ids=tids,
            activation=MoEActivation.SILU, apply_router_weight_on_input=False,
            global_num_experts=E, expert_map=None, quant_config=qc)
        try:
            mo = bench(ours)
        except Exception as e:
            print(f"{M:>6}  ours FAILED: {type(e).__name__}: {e}"); continue
        ms = bench(stock)
        print(f"{M:>6} {mo:>9.3f} {ms:>9.3f} {mo/ms:>9.2f}x")


if __name__ == "__main__":
    main()
