"""Probe gemm2 (the MoE down-projection) at decode -- the weak link (126 GB/s vs
gemm1's 273). Times v7-scatter / v7-nonscatter / v6-scatter / v6-nonscatter and a
NWARPS sweep at the gemm2 decode shape + routing. Container (needs vllm)."""
import os, time, torch
import w4a8_fp8_wmma
from w4a8_fp8_wmma.moe_experts import _wna16_moe_to_op_layout
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size

mmq = w4a8_fp8_wmma.mmq_fp8_moe_gemm
scat = w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter
PEAK = 640.0


def t_ms(fn, it=100, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e3


def main():
    dev = "cuda"; torch.manual_seed(0)
    E, hidden, inter, g, top_k = 128, 2304, 896, 32, 8
    # gemm2: x=(P,inter) @ w2=(E,hidden,inter) -> (P,hidden)
    N2, K2 = hidden, inter
    w2 = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device=dev)
    s2 = torch.rand(E, N2, K2 // g, device=dev, dtype=torch.float16) * 0.02 + 0.005
    z2 = torch.randint(0, 256, (E, N2 // 2, K2 // g), dtype=torch.uint8, device=dev)
    w2o, s2o, z2o = _wna16_moe_to_op_layout(w2, s2, z2)
    wbytes = N2 * K2 * 0.5
    print(f"gemm2 shape N={N2} K={K2} g={g} | {'M':>3} {'nact':>5} "
          f"{'v7sc':>7} {'v7ns':>7} {'v6sc':>7} {'v6ns':>7} {'v7sc_nw8':>9} {'v7sc_nw16':>10}")
    for M in [1, 4, 8, 16, 32]:
        tids = torch.stack([torch.randperm(E, device=dev)[:top_k] for _ in range(M)]).to(torch.int32)
        bm = 16
        sti, eid, ntp = moe_align_block_size(tids, bm, E, None, pad_sorted_ids=True,
                                             ignore_invalid_experts=True)
        P = sti.numel()
        nact = len(set(eid[eid >= 0].tolist())) if (eid >= 0).any() else 0
        buf2 = torch.randn(P, inter, dtype=torch.float16, device=dev) * 0.3
        tw_flat = torch.rand(M * top_k, dtype=torch.float32, device=dev)
        outacc = torch.zeros((M, hidden), dtype=torch.float32, device=dev)
        ident = torch.arange(P, dtype=torch.int32, device=dev)

        def v7sc():
            outacc.zero_()
            scat(buf2, w2o, s2o, sti, eid, ntp, tw_flat, outacc, top_k, bm, version=7, w_zeros=z2o)
        def v7ns():
            return mmq(buf2, w2o, s2o, ident, eid, ntp, 1, bm, version=7, w_zeros=z2o)
        def v6sc():
            outacc.zero_()
            scat(buf2, w2o, s2o, sti, eid, ntp, tw_flat, outacc, top_k, bm, version=6, w_zeros=z2o)
        def v6ns():
            return mmq(buf2, w2o, s2o, ident, eid, ntp, 1, bm, version=6, w_zeros=z2o)
        def v7sc_nw(nw):
            os.environ["VLLM_W4A8_MOE_GEMV_NWARPS"] = str(nw)
            outacc.zero_()
            scat(buf2, w2o, s2o, sti, eid, ntp, tw_flat, outacc, top_k, bm, version=7, w_zeros=z2o)
        r = {}
        for k, fn in [("v7sc", v7sc), ("v7ns", v7ns), ("v6sc", v6sc), ("v6ns", v6ns)]:
            try: r[k] = t_ms(fn)
            except Exception as e: r[k] = float("nan")
        os.environ["VLLM_W4A8_MOE_GEMV_NWARPS"] = "8"; r["nw8"] = t_ms(v7sc)
        os.environ["VLLM_W4A8_MOE_GEMV_NWARPS"] = "16"; r["nw16"] = t_ms(v7sc)
        os.environ.pop("VLLM_W4A8_MOE_GEMV_NWARPS", None)
        gbs = lambda ms: nact * wbytes / (ms * 1e-3) / 1e9
        print(f"{'':>16} {M:>3} {nact:>5} {r['v7sc']:7.3f} {r['v7ns']:7.3f} "
              f"{r['v6sc']:7.3f} {r['v6ns']:7.3f} {r['nw8']:9.3f} {r['nw16']:10.3f}  "
              f"| v7sc={gbs(r['v7sc']):.0f}GB/s")


if __name__ == "__main__":
    main()
