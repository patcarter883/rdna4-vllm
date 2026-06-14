import torch  # MUST come before `from . import _C` so libtorch is dlopen'd first

from . import _C  # noqa: F401


def mmq_fp8_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    version: int = 0,
    w_zeros: torch.Tensor | None = None,
) -> torch.Tensor:
    """W4A8-FP8 MMQ kernel for gfx1201 (RDNA4).

    Args:
        x: (M, K) fp16 activations.
        w_packed: (N, K/8) int32 weights, 8 uint4b8 per int32, low nibble first.
        scales: (N, K/32) fp16 per-group weight scales.
        version: 0 = scalar fp8 reference (always correct, slow);
                 1 = WMMA + LDS staging (on-device WIP).
        w_zeros: optional (N/8, K/32) int32 packed per-group zero points for
                 asymmetric (AWQ) quant; None for symmetric uint4b8 (zp=8).

    Returns:
        (M, N) fp16 output.
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, w_packed, scales, w_zeros, version)


def mmq_fp8_moe_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int,
    version: int = 0,
    w_zeros: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped (MoE) W4A8-FP8 GEMM for gfx1201.

    Computes, for each padded sorted row, out[row] = x[token(row)] @ W[expert]^T
    (fp8-expanded), matching vLLM's fused_moe contract. The topk-weight scaling +
    scatter-reduce over the (token, expert-slot) rows is a separate epilogue.

    Args:
        x: (T, K) fp16 token activations.
        w_packed: (E, N, K/8) int32 stacked per-expert weights (8 uint4 per int32).
        scales: (E, N, K/group) fp16 per-expert per-group scales.
        sorted_token_ids: (P,) int32, expert-sorted padded token slots.
        expert_ids: (P/block_m,) int32, expert per block.
        num_tokens_post_padded: (1,) int32 valid padded length.
        top_k: experts per token (activation row = sorted_token_id // top_k).
        block_m: moe_align block size (multiple of 16; v5 needs <=64).
        version: 0 = scalar golden, 5 = fp8 WMMA.
        w_zeros: (E, N/8, K/group) int32 packed zeros (AWQ); None for uint4b8.

    Returns:
        (P, N) fp16 output in the padded sorted layout.
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, block_m, version)


def mmq_fp8_moe_gemm_scatter(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    top_k: int,
    block_m: int,
    version: int = 0,
    w_zeros: torch.Tensor | None = None,
) -> None:
    """Fused gemm2 + topk-weight + indirect atomic scatter (gfx1201).

    Per padded sorted row r (gathered by IDENTITY from x), computes the gemm2
    result and accumulates it into ``output`` IN PLACE via
    ``output[sorted_token_ids[r]//top_k] += topk_weights[sorted_token_ids[r]] *
    (x[r] @ W[expert]^T)`` with global atomicAdd (cross-block, top_k-wide
    collisions). Replaces the (P,N) materialise + torch scatter-reduce.

    Args:
        x: (P, K) fp16 post-activation buffer (one row per padded (token,slot)).
        w_packed: (E, N, K/8) int32.  scales: (E, N, K/group) fp16.
        sorted_token_ids: (P,) int32 gemm1 (token,slot) ids.
        expert_ids: (P/block_m,) int32.  num_tokens_post_padded: (1,) int32.
        topk_weights: (M*top_k,) fp32 flattened router weights.
        output: (M, N) fp32, **pre-zeroed**, mutated in place.
        w_zeros: (E, N/8, K/group) int32 (AWQ) or None (symmetric uint4b8 zp=8).
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm_scatter(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, output, top_k, block_m, version)


def mmq_fp8_moe_gather_reduce(out2, sorted_token_ids, topk_weights,
                              num_tokens_post_padded, top_k):
    """Contention-free MoE reduce: gemm2 NON-scatter output (P,N) fp16 ->
    per-token weighted gather-reduce -> (M, N) fp32. Replaces the gemm2 atomic
    scatter (no top_k-contended global atomics); HIP-graph safe. M = topk_weights
    .numel() // top_k.

    Args:
        out2: (P, N) fp16, the gemm2 non-scatter result (identity-gathered rows).
        sorted_token_ids: (P,) int32, the gemm1 (token,slot) ids.
        topk_weights: (M*top_k,) fp32 flattened router weights.
        num_tokens_post_padded: (1,) int32.
    Returns: (M, N) fp32.
    """
    return torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gather_reduce(
        out2, sorted_token_ids, topk_weights, num_tokens_post_padded, top_k)
