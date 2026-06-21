import torch  # MUST come before `from . import _C` so libtorch is dlopen'd first

from . import _C  # noqa: F401


# Descriptive kernel names map to the opaque int the torch op carries over the ABI
# (kernel_names.h). Names are the public surface; no caller writes the int. The former
# MoE v5/v6 are ONE name "wmma" — A-residence is an env knob (VLLM_W4A8_MOE_A_IN_LDS),
# not a name. Hard-break: a removed numeric value (e.g. 5) raises, never mis-dispatches.
_MOE_KERNELS = {"scalar": 0, "wmma": 6, "gemv": 7}

# Dense mmq_fp8_gemm kernels — names ↔ the opaque DenseKernel ints (kernel_names.h).
# Enum values equal the old version ints; the v3 gap is absent. Served names are
# reference_scalar(0) / prefill_wmma(5) / prefill_wmma_ashuffle(10) / decode_gemv(11);
# the rest are retired research kernels kept for benches. (The register-direct
# v15/16/17 are SEPARATE ops, not values here.)
_DENSE_KERNELS = {
    "reference_scalar": 0, "rocwmma_v1": 1, "rocwmma_tiled": 2, "rocwmma_pipe": 4,
    "prefill_wmma": 5, "prefill_wmma_b128": 6, "wmma_tiled_tuned": 7, "wmma_dbuf": 8,
    "wmma_dbuf2": 9, "prefill_wmma_ashuffle": 10, "decode_gemv": 11,
    "splitk_smallm": 12, "regdirect_shuffle": 13, "nsplit_smallm": 14,
}


def _resolve_dense_kernel(kernel) -> int:
    """Map a descriptive dense kernel name to its op id; reject unknowns/legacy ints."""
    if isinstance(kernel, str):
        try:
            return _DENSE_KERNELS[kernel]
        except KeyError:
            raise ValueError(
                f"unknown dense kernel {kernel!r}; known: {sorted(_DENSE_KERNELS)}"
            ) from None
    raise TypeError(
        f"dense kernel must be a name in {sorted(_DENSE_KERNELS)} (the integer "
        f"'version' was removed); got {kernel!r}")


def _resolve_moe_kernel(kernel) -> int:
    """Map a descriptive MoE kernel name to its op id; reject unknowns/legacy ints."""
    if isinstance(kernel, str):
        try:
            return _MOE_KERNELS[kernel]
        except KeyError:
            raise ValueError(
                f"unknown moe kernel {kernel!r}; known: {sorted(_MOE_KERNELS)}") from None
    raise TypeError(
        f"moe kernel must be a name in {sorted(_MOE_KERNELS)} (the integer 'version' was "
        f"removed); got {kernel!r}")


# FakeTensor / meta registration for all W4A8 ops lives in ONE place — the
# comprehensive block further down (dense regdirect ops + the MoE ops). Without a
# fake (abstract) impl, Dynamo can't trace these custom ops and vLLM's full
# torch.compile + cudagraph path aborts, leaving the layer --enforce-eager only.
# (Do not add a second registration here: register_fake raises if an op already
# has a fake impl — a duplicate block crashes import at startup.)


def mmq_fp8_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    kernel: str = "reference_scalar",
    w_zeros: torch.Tensor | None = None,
    weight_is_e2m1: bool = False,
) -> torch.Tensor:
    """W4A8-FP8 MMQ kernel for gfx1201 (RDNA4).

    Args:
        x: (M, K) fp16 activations.
        w_packed: (N, K/8) int32 weights, 8 4-bit codes per int32, low nibble first.
        scales: (N, K/32) fp16 per-group weight scales.
        kernel: descriptive dense-kernel name (see _DENSE_KERNELS); default
                "reference_scalar" (the scalar fp8 golden). Served names are
                "prefill_wmma" / "prefill_wmma_ashuffle" / "decode_gemv".
        w_zeros: optional (N/8, K/32) int32 packed per-group zero points for
                 asymmetric (AWQ) quant; None for symmetric uint4b8 (zp=8).
        weight_is_e2m1: when True the 4-bit codes are decoded as mxfp4 (OCP E2M1)
                 instead of uniform int4; scales are the E8M0 group exponents folded
                 to fp16 (see mxfp4/convert.py), and w_zeros must be None (symmetric).

    Returns:
        (M, N) fp16 output.
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(
        x, w_packed, scales, w_zeros, _resolve_dense_kernel(kernel), weight_is_e2m1)


def mmq_fp8_moe_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int,
    kernel: str = "wmma",
    w_zeros: torch.Tensor | None = None,
    weight_is_e2m1: bool = False,
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
        block_m: moe_align block size (multiple of 16; wmma needs <=64).
        kernel: "scalar" = golden reference, "wmma" = fp8 WMMA (tiled, default),
                "gemv" = scalar-dot decode GEMV. A-residence (former v5 vs v6) is
                the env knob VLLM_W4A8_MOE_A_IN_LDS, not a kernel name.
        w_zeros: (E, N/8, K/group) int32 packed zeros (AWQ); None for uint4b8.

    Returns:
        (P, N) fp16 output in the padded sorted layout.
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, block_m, _resolve_moe_kernel(kernel),
        weight_is_e2m1)


def mmq_fp8_moe_gemm1_silu(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    scales: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    block_m: int,
    kernel: str = "wmma",
    w_zeros: torch.Tensor | None = None,
    weight_is_e2m1: bool = False,
) -> torch.Tensor:
    """Fused gemm1 + silu_and_mul for the gated MoE (gfx1201).

    Runs the gated gemm1 (w13, N = 2*inter = [gate | up]) AND the silu_and_mul
    activation in one kernel, returning the post-activation ``(P, inter)`` directly
    -- dropping the separate silu launch and the (P, 2*inter) out1 / (P, inter)
    buf2 HBM round-trip the unfused ``mmq_fp8_moe_gemm`` + ``silu_and_mul`` pays.

    Bit-exact to ``mmq_fp8_moe_gemm(x, w13, ..., kernel)`` followed by
    ``torch.ops._C.silu_and_mul`` (each half rounded to fp16 exactly as gemm1's
    store, silu in fp32 then fp16, final half*half multiply).

    Args:
        x: (T, K) fp16 token activations.
        w_packed: (E, 2*inter, K/8) int32 stacked w13 ([gate|up] along dim 1).
        scales: (E, 2*inter, K/group) fp16.
        kernel: "wmma" (the only fused-epilogue path). A-residence (former v5 A+B
                LDS vs v6 A-out-of-LDS, served default) is the env knob
                VLLM_W4A8_MOE_A_IN_LDS. The silu-fused path keeps its own
                v5/v6 kernels (no tiled equivalent).
        w_zeros: (E, (2*inter)/8, K/group) int32 (AWQ) or None (uint4b8 zp=8).

    Returns:
        (P, inter) fp16 post-activation in the padded sorted layout.
    """
    if w_zeros is None:
        w_zeros = torch.empty(0, dtype=torch.int32, device=x.device)
    return torch.ops.w4a8_fp8_wmma.mmq_fp8_moe_gemm1_silu(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, top_k, block_m, _resolve_moe_kernel(kernel),
        weight_is_e2m1)


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
    kernel: str = "wmma",
    w_zeros: torch.Tensor | None = None,
    weight_is_e2m1: bool = False,
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
        num_tokens_post_padded, topk_weights, output, top_k, block_m,
        _resolve_moe_kernel(kernel), weight_is_e2m1)


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


# ---------------------------------------------------------------------------
# Fake / meta kernels for torch.compile (pt2-compliance).
#
# vLLM 0.22.69's full-graph aot_compile traces the model forward; the moment our
# kernel actually engages (single-layout default / autotuned crossover), Dynamo
# hits these custom ops and — without an output-shape (fake) impl — raises
# "unsupported operator: w4a8_fp8_wmma.*". These fakes return an empty tensor of
# the real output shape/dtype so Dynamo can infer metadata WITHOUT running the
# kernel; paired with the pt2_compliant_tag on each m.def (bindings.cpp). No GPU
# work runs in a fake. Output contracts mirror the CUDA impls exactly:
#   mmq_fp8_gemm:            (M, N) f16        N = w_packed.size(0)   [(N, K//8)]
#   mmq_regdirect_fp8/_f16/_w4a16: (M, N) f16   N = the `N` arg
#   mmq_fp8_moe_gemm:        (P, N) f16        P = sorted_ids, N = w_packed.size(1)
#   mmq_fp8_moe_gemm1_silu:  (P, inter) f16    inter = w_packed.size(1)//2  [(E,2*inter,K//8)]
#   mmq_fp8_moe_gemm_scatter: () — in-place into `output` (mutating, dormant)
#   mmq_fp8_moe_gather_reduce:(M, N) f32       M = topk_weights.numel()//top_k
_register_fake = getattr(getattr(torch, "library", None), "register_fake", None)
if _register_fake is not None:  # torch >= 2.4 (base is 2.10)
    @_register_fake("w4a8_fp8_wmma::mmq_fp8_gemm")
    def _fake_mmq_fp8_gemm(x, w_packed, scales, w_zeros, kernel, weight_is_e2m1=False):
        return x.new_empty((x.shape[0], w_packed.shape[0]), dtype=torch.float16)

    @_register_fake("w4a8_fp8_wmma::mmq_regdirect_fp8")
    def _fake_mmq_regdirect_fp8(x, w_rep, scales, w_zeros, N):
        return x.new_empty((x.shape[0], N), dtype=torch.float16)

    @_register_fake("w4a8_fp8_wmma::mmq_regdirect_f16")
    def _fake_mmq_regdirect_f16(x, w_rep, scales, w_zeros, N):
        return x.new_empty((x.shape[0], N), dtype=torch.float16)

    @_register_fake("w4a8_fp8_wmma::mmq_regdirect_w4a16")
    def _fake_mmq_regdirect_w4a16(x, w_rep, scales, w_zeros, N):
        return x.new_empty((x.shape[0], N), dtype=torch.float16)

    @_register_fake("w4a8_fp8_wmma::mmq_fp8_moe_gemm")
    def _fake_mmq_fp8_moe_gemm(x, w_packed, scales, w_zeros, sorted_token_ids,
                               expert_ids, num_tokens_post_padded, top_k,
                               block_m, kernel, weight_is_e2m1=False):
        return x.new_empty((sorted_token_ids.shape[0], w_packed.shape[1]),
                           dtype=torch.float16)

    @_register_fake("w4a8_fp8_wmma::mmq_fp8_moe_gemm1_silu")
    def _fake_mmq_fp8_moe_gemm1_silu(x, w_packed, scales, w_zeros, sorted_token_ids,
                                     expert_ids, num_tokens_post_padded, top_k,
                                     block_m, kernel, weight_is_e2m1=False):
        return x.new_empty((sorted_token_ids.shape[0], w_packed.shape[1] // 2),
                           dtype=torch.float16)

    @_register_fake("w4a8_fp8_wmma::mmq_fp8_moe_gemm_scatter")
    def _fake_mmq_fp8_moe_gemm_scatter(x, w_packed, scales, w_zeros, sorted_token_ids,
                                       expert_ids, num_tokens_post_padded, topk_weights,
                                       output, top_k, block_m, kernel, weight_is_e2m1=False):
        return None  # in-place accumulate into `output`; nothing returned

    @_register_fake("w4a8_fp8_wmma::mmq_fp8_moe_gather_reduce")
    def _fake_mmq_fp8_moe_gather_reduce(out2, sorted_token_ids, topk_weights,
                                        num_tokens_post_padded, top_k):
        return out2.new_empty((topk_weights.shape[0] // top_k, out2.shape[1]),
                              dtype=torch.float32)
