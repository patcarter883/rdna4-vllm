"""Layer-level validation of the compressed-tensors MoE hook (symmetric int4).

Non-circular: builds random registered-layout packed weights, computes a
REFERENCE through vLLM's STOCK path (the exact process_weights transpose +
Triton `fused_experts`), and compares against our converter + grouped op. A
layout/nibble-order bug -> gross divergence (rel ~1); correct -> only fp8
activation-quant noise (rel <~3%). Runs in tiny VRAM (no full model).
"""
import sys
import torch

import w4a8_fp8_wmma  # noqa: F401
from w4a8_fp8_wmma.moe_experts import _ct_moe_to_op_layout, _run_grouped_moe
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation


def gptq_pack_k(vals):  # (E,K,N) uint4 -> (E,K//8,N) int32, natural order along K
    E, K, N = vals.shape
    packed = torch.zeros((E, K // 8, N), dtype=torch.int32, device=vals.device)
    for j in range(8):
        packed |= (vals[:, j::8, :] & 0xF) << (j * 4)
    return packed


def make_case(E, hidden, inter, top_k, g, T, dev):
    q13 = torch.randint(0, 16, (E, hidden, 2 * inter), dtype=torch.int32, device=dev)
    q2 = torch.randint(0, 16, (E, inter, hidden), dtype=torch.int32, device=dev)
    w13p = gptq_pack_k(q13)   # (E, hidden//8, 2*inter)
    w2p = gptq_pack_k(q2)     # (E, inter//8, hidden)
    s13 = torch.rand(E, hidden // g, 2 * inter, device=dev, dtype=torch.float16) * 0.02 + 0.005
    s2 = torch.rand(E, inter // g, hidden, device=dev, dtype=torch.float16) * 0.02 + 0.005
    x = torch.randn(T, hidden, dtype=torch.float16, device=dev) * 0.5
    topk_ids = torch.stack(
        [torch.randperm(E, device=dev)[:top_k] for _ in range(T)]).to(torch.int32)
    topk_weights = torch.rand(T, top_k, dtype=torch.float32, device=dev)
    return w13p, w2p, s13, s2, x, topk_ids, topk_weights


def stock_ref(x, w13p, w2p, s13, s2, tw, tids, E, g):
    # exact replica of CompressedTensorsWNA16MoEMethod.process_weights + apply
    w13 = w13p.transpose(1, 2).contiguous().view(torch.uint8)
    w2 = w2p.transpose(1, 2).contiguous().view(torch.uint8)
    qc = int4_w4a16_moe_quant_config(
        w1_scale=s13.transpose(1, 2).contiguous(),
        w2_scale=s2.transpose(1, 2).contiguous(),
        w1_zp=None, w2_zp=None, block_shape=[0, g])
    return fused_experts(
        x, w13, w2, topk_weights=tw, topk_ids=tids, activation=MoEActivation.SILU,
        apply_router_weight_on_input=False, global_num_experts=E,
        expert_map=None, quant_config=qc)


def ours(x, w13p, w2p, s13, s2, tw, tids, E, version):
    w13_op, s13_op = _ct_moe_to_op_layout(w13p, s13)
    w2_op, s2_op = _ct_moe_to_op_layout(w2p, s2)
    return _run_grouped_moe(
        x, w13_op, w2_op, s13_op, s2_op, None, None, tw, tids,
        MoEActivation.SILU, E, None, False, version, out_dtype=x.dtype)


def main():
    if not torch.cuda.is_available():
        print("FAIL: no device"); sys.exit(1)
    print("Device:", torch.cuda.get_device_name(0))
    dev = "cuda"
    torch.manual_seed(0)
    cases = [(8, 256, 128, 2, 32, 64), (16, 512, 256, 4, 32, 32),
             (4, 256, 128, 2, 32, 1), (64, 2304, 896, 8, 32, 16)]  # last ~ Qwen3.6
    ok_all = True
    for (E, hidden, inter, tk, g, T) in cases:
        w13p, w2p, s13, s2, x, tids, tw = make_case(E, hidden, inter, tk, g, T, dev)
        try:
            ref = stock_ref(x, w13p, w2p, s13, s2, tw, tids, E, g).float()
        except Exception as e:
            print(f"  REF FAILED (E={E} h={hidden}): {type(e).__name__}: {e}")
            ok_all = False; continue
        for v in (0, 5):
            out = ours(x, w13p, w2p, s13, s2, tw, tids, E, v).float()
            diff = (out - ref).abs()
            refm = ref.abs().mean().item()
            rel = diff.mean().item() / max(refm, 1e-6)
            ok = rel < 0.05
            ok_all &= ok
            print(f"  CT-layer v{v} E={E} h={hidden} I={inter} tk={tk} g={g} "
                  f"T={T}: rel_mean={rel:.4f} |ref|={refm:.4f} -> "
                  f"{'PASS' if ok else 'FAIL'}")
    print("=" * 56)
    print("ALL PASSED" if ok_all else "FAIL")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
