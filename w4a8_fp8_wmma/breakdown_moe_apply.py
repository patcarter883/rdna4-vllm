"""Per-step breakdown of the MoE apply vs stock, to find where decode time goes.

Times moe_align / gemm1(v7) / silu / gemm2-scatter(v7) individually and the full
_run_grouped_moe, vs stock fused_experts, at the Qwen3.6/Mellum2 expert shape.
Needs vLLM -> run in the container. HIP_VISIBLE_DEVICES=0.
"""
import os, time, torch
import w4a8_fp8_wmma
from w4a8_fp8_wmma.moe_experts import _wna16_moe_to_op_layout, _run_grouped_moe
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation, apply_moe_activation
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

mmq = w4a8_fp8_wmma.mmq_fp8_moe_gemm
scat = w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter


def t_ms(fn, it=80, wu=25):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3


def main():
    dev = "cuda"; torch.manual_seed(0)
    E, hidden, inter, g, top_k = 128, 2304, 896, 32, 8
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
    qc = int4_w4a16_moe_quant_config(w1_scale=s13, w2_scale=s2, w1_zp=z13, w2_zp=z2,
                                     block_shape=[0, g])
    print(f"{'M':>4} {'align':>7} {'gemm1':>7} {'silu':>7} {'gemm2sc':>8} "
          f"{'sum':>7} {'apply':>7} {'stock':>7} {'ap/stk':>7}")
    for M in [1, 2, 4, 8, 16, 32]:
        x = torch.randn(M, hidden, dtype=torch.float16, device=dev) * 0.5
        tids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
        tw = torch.rand(M, top_k, dtype=torch.float32, device=dev)
        bm = 8
        def do_align():
            return moe_align_block_size(tids, bm, E, None, pad_sorted_ids=True,
                                        ignore_invalid_experts=True)
        sti, eid, ntp = do_align()
        P = sti.numel()
        x16 = x.contiguous()
        def g1():
            return mmq(x16, w13o, s13o, sti, eid, ntp, top_k, bm, kernel="gemv", w_zeros=z13o)
        out1 = g1()
        buf2 = torch.empty((P, inter), dtype=torch.float16, device=dev)
        def silu():
            apply_moe_activation(MoEActivation.SILU, buf2, out1)
        silu()
        tw_flat = tw.reshape(-1).contiguous()
        outacc = torch.zeros((M, hidden), dtype=torch.float32, device=dev)
        def g2():
            outacc.zero_()
            scat(buf2, w2o, s2o, sti, eid, ntp, tw_flat, outacc, top_k, bm, kernel="gemv", w_zeros=z2o)
        ours = lambda: _run_grouped_moe(x, w13o, w2o, s13o, s2o, z13o, z2o, tw, tids,
                                        MoEActivation.SILU, E, None, False, "wmma", out_dtype=x.dtype)
        stock = lambda: fused_experts(x, w13, w2, topk_weights=tw, topk_ids=tids,
                                      activation=MoEActivation.SILU,
                                      apply_router_weight_on_input=False,
                                      global_num_experts=E, expert_map=None, quant_config=qc)
        ta = t_ms(do_align); t1 = t_ms(g1); ts = t_ms(silu); t2 = t_ms(g2)
        tap = t_ms(ours); tst = t_ms(stock)
        print(f"{M:>4} {ta:7.3f} {t1:7.3f} {ts:7.3f} {t2:8.3f} {ta+t1+ts+t2:7.3f} "
              f"{tap:7.3f} {tst:7.3f} {tap/tst:6.2f}x")


if __name__ == "__main__":
    main()
