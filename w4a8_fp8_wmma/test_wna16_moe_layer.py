"""Layer-level validation of the MoeWNA16 MoE hook (int4 AWQ-asym, the path
Qwen3.6-35B-A3B takes on gfx1201). Builds registered MoeWNA16-format weights
(uint8 standard format + uint8 zero points), computes a REFERENCE via stock
Triton `fused_experts` (what MoeWNA16Method.apply calls), and compares against
our zero-copy `.view(int32)` + grouped op. Layout/zp bug -> rel~1; correct ->
fp8 noise (rel <~5%). Tiny VRAM.
"""
import os
import sys
import torch

import w4a8_fp8_wmma  # noqa: F401
from w4a8_fp8_wmma.moe_experts import _wna16_moe_to_op_layout, _run_grouped_moe
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


def make_case(E, hidden, inter, top_k, g, T, dev, has_zp):
    N13, K13 = 2 * inter, hidden
    N2, K2 = hidden, inter
    w13 = torch.randint(0, 256, (E, N13, K13 // 2), dtype=torch.uint8, device=dev)
    w2 = torch.randint(0, 256, (E, N2, K2 // 2), dtype=torch.uint8, device=dev)
    s13 = torch.rand(E, N13, K13 // g, device=dev, dtype=torch.float16) * 0.02 + 0.005
    s2 = torch.rand(E, N2, K2 // g, device=dev, dtype=torch.float16) * 0.02 + 0.005
    z13 = z2 = None
    if has_zp:
        z13 = torch.randint(0, 256, (E, N13 // 2, K13 // g), dtype=torch.uint8, device=dev)
        z2 = torch.randint(0, 256, (E, N2 // 2, K2 // g), dtype=torch.uint8, device=dev)
    x = torch.randn(T, hidden, dtype=torch.float16, device=dev) * 0.5
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    topk_weights = torch.rand(T, top_k, dtype=torch.float32, device=dev)
    return w13, w2, s13, s2, z13, z2, x, topk_ids, topk_weights


def stock_ref(x, w13, w2, s13, s2, z13, z2, tw, tids, E, g):
    qc = int4_w4a16_moe_quant_config(
        w1_scale=s13, w2_scale=s2, w1_zp=z13, w2_zp=z2, block_shape=[0, g])
    return fused_experts(
        x, w13, w2, topk_weights=tw, topk_ids=tids, activation=MoEActivation.SILU,
        apply_router_weight_on_input=False, global_num_experts=E,
        expert_map=None, quant_config=qc)


def ours(x, w13, w2, s13, s2, z13, z2, tw, tids, E, kernel):
    w13o, s13o, z13o = _wna16_moe_to_op_layout(w13, s13, z13)
    w2o, s2o, z2o = _wna16_moe_to_op_layout(w2, s2, z2)
    return _run_grouped_moe(
        x, w13o, w2o, s13o, s2o, z13o, z2o, tw, tids, MoEActivation.SILU,
        E, None, False, kernel, out_dtype=x.dtype)


def main():
    if not torch.cuda.is_available():
        print("FAIL: no device"); sys.exit(1)
    print("Device:", torch.cuda.get_device_name(0))
    dev = "cuda"
    torch.manual_seed(0)
    # (E, hidden, inter, top_k, g, T, has_zp) ; last = exact Qwen3.6 shape, asym
    cases = [(8, 256, 128, 2, 32, 64, True), (16, 512, 256, 4, 32, 32, True),
             (4, 256, 128, 2, 32, 1, True), (8, 256, 128, 2, 32, 32, False),
             (64, 2304, 896, 8, 32, 16, True)]
    ok_all = True
    for (E, hidden, inter, tk, g, T, zp) in cases:
        w13, w2, s13, s2, z13, z2, x, tids, tw = make_case(
            E, hidden, inter, tk, g, T, dev, zp)
        try:
            ref = stock_ref(x, w13, w2, s13, s2, z13, z2, tw, tids, E, g).float()
        except Exception as ex:
            print(f"  REF FAILED (E={E} h={hidden} zp={zp}): {type(ex).__name__}: {ex}")
            ok_all = False; continue
        # former v0 -> "scalar"; former v5 (A-in-LDS) -> "wmma" + VLLM_W4A8_MOE_A_IN_LDS=1
        for (label, kernel, a_in_lds) in (("scalar", "scalar", False),
                                          ("wmma", "wmma", True)):
            if a_in_lds:
                os.environ["VLLM_W4A8_MOE_A_IN_LDS"] = "1"
            try:
                out = ours(x, w13, w2, s13, s2, z13, z2, tw, tids, E, kernel).float()
            finally:
                if a_in_lds:
                    os.environ.pop("VLLM_W4A8_MOE_A_IN_LDS", None)
            rel = (out - ref).abs().mean().item() / max(ref.abs().mean().item(), 1e-6)
            ok = rel < 0.07
            ok_all &= ok
            print(f"  WNA16 {label} E={E} h={hidden} I={inter} g={g} T={T} zp={zp}: "
                  f"rel_mean={rel:.4f} -> {'PASS' if ok else 'FAIL'}")
    print("=" * 56)
    print("ALL PASSED" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
