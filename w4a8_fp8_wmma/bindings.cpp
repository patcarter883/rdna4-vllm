// torch bindings for the W4A8-FP8 MMQ HIP custom op (gfx1201 / RDNA4).
//
// Exposes torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, w_packed, scales, w_zeros,
// kernel) -> out.
//
//   kernel: an opaque DenseKernel id (kernel_names.h); enum values equal the old
//   version ints (0 = reference_scalar golden, 5 = prefill_wmma, 10 = ashuffle,
//   11 = decode_gemv, ...). Descriptive NAMES live above the ABI in __init__.py.
//
// Tensor contracts (match vLLM compressed_tensors_wNa16 / the gfx1151 ref):
//   x         : (M, K)        fp16,  CUDA, contiguous
//   w_packed  : (N, K/8)      int32, CUDA, contiguous  (8 uint4b8 per int32)
//   scales    : (N, K/32)     fp16,  CUDA, contiguous  (group_size=32)
//   w_zeros   : (N/8, K/32)   int32, CUDA, contiguous, optional (empty=symmetric)
//   out       : (M, N)        fp16,  CUDA, contiguous, allocated here

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel_names.h"   // w4a8::{Dense,Moe}Kernel + *_valid() (descriptive ids; opaque int ABI)

void launch_mmq_fp8_gemm_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    at::Tensor& out,
    int64_t kernel,
    bool weight_is_e2m1);

void launch_mmq_regdirect_fp8_gfx1201(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, int64_t);

void launch_mmq_regdirect_f16_gfx1201(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, int64_t);

void launch_mmq_regdirect_w4a16_gfx1201(
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
    at::Tensor&, int64_t);

void launch_mmq_fp8_moe_gemm_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    at::Tensor& out,
    int64_t top_k,
    int64_t block_m,
    int64_t version,
    bool weight_is_e2m1);

void launch_mmq_fp8_moe_gemm1_silu_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    at::Tensor& out,
    int64_t top_k,
    int64_t block_m,
    int64_t version,
    bool weight_is_e2m1);

void launch_mmq_fp8_moe_gemm_scatter_gfx1201(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    const at::Tensor& topk_weights,
    at::Tensor& output,
    int64_t top_k,
    int64_t block_m,
    int64_t version,
    bool weight_is_e2m1);

void launch_moe_gather_reduce_gfx1201(
    const at::Tensor& out2,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& topk_weights,
    const at::Tensor& num_tokens_post_padded,
    at::Tensor& out,
    int64_t top_k);

namespace {

constexpr int64_t kPackFactor = 8;

at::Tensor mmq_fp8_gemm_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    int64_t kernel,
    bool weight_is_e2m1) {

    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && scales.is_cuda(),
                "x, w_packed, scales must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");
    TORCH_CHECK(x.dim() == 2 && w_packed.dim() == 2 && scales.dim() == 2,
                "all inputs must be 2D");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() && scales.is_contiguous(),
                "all inputs must be contiguous");

    const int64_t M = x.size(0);
    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(0);

    TORCH_CHECK(w_packed.size(1) * kPackFactor == K,
                "w_packed last dim mismatch: expected K/8 = ", K / kPackFactor,
                " got ", w_packed.size(1));
    TORCH_CHECK(scales.size(0) == N, "scales.size(0) must equal N=", N);
    const int64_t group_size = K / scales.size(1);
    TORCH_CHECK(scales.size(1) * group_size == K,
                "K not divisible by scales' K dim");
    TORCH_CHECK(group_size % 16 == 0 && group_size <= 128,
                "group_size must be a multiple of 16 and <= 128; got ", group_size);
    TORCH_CHECK(w4a8::dense_kernel_valid(kernel),
                "dense kernel id invalid (got ", kernel, "); must be a DenseKernel "
                "value 0..14 (rejecting the retired v3 gap) — see kernel_names.h. "
                "Names live above the torch ABI in __init__.py.");

    if (w_zeros.defined() && w_zeros.numel() > 0) {
        TORCH_CHECK(!weight_is_e2m1, "mxfp4 (E2M1) weights are symmetric; w_zeros must be empty");
        TORCH_CHECK(w_zeros.is_cuda() && w_zeros.scalar_type() == at::kInt,
                    "w_zeros must be CUDA int32 (packed)");
        TORCH_CHECK(w_zeros.dim() == 2 && w_zeros.size(0) * 8 == N && w_zeros.size(1) == scales.size(1),
                    "w_zeros shape mismatch: expected (", N / 8, ", ", scales.size(1), ")");
        TORCH_CHECK(w_zeros.is_contiguous(), "w_zeros must be contiguous");
    }

    auto out = at::empty({M, N}, x.options());
    launch_mmq_fp8_gemm_gfx1201(x, w_packed, scales, w_zeros, out, kernel, weight_is_e2m1);
    return out;
}

// mmq_regdirect_fp8 (was v15): register-direct WMMA with pre-repacked (N/16,K/16,32) weights.
at::Tensor mmq_regdirect_fp8_forward(
    const at::Tensor& x, const at::Tensor& w_rep, const at::Tensor& scales,
    const at::Tensor& w_zeros, int64_t N) {
    const int64_t M = x.size(0);
    auto out = at::empty({M, N}, x.options());
    launch_mmq_regdirect_fp8_gfx1201(x, w_rep, scales, w_zeros, out, N);
    return out;
}

// mmq_regdirect_f16 (was v16): f16-WMMA twin of mmq_regdirect_fp8 (same w_rep contract).
at::Tensor mmq_regdirect_f16_forward(
    const at::Tensor& x, const at::Tensor& w_rep, const at::Tensor& scales,
    const at::Tensor& w_zeros, int64_t N) {
    const int64_t M = x.size(0);
    auto out = at::empty({M, N}, x.options());
    launch_mmq_regdirect_f16_gfx1201(x, w_rep, scales, w_zeros, out, N);
    return out;
}

// mmq_regdirect_w4a16 (was v17): true W4A16 -- fp16 activations direct (no act-quant).
at::Tensor mmq_regdirect_w4a16_forward(
    const at::Tensor& x, const at::Tensor& w_rep, const at::Tensor& scales,
    const at::Tensor& w_zeros, int64_t N) {
    const int64_t M = x.size(0);
    auto out = at::empty({M, N}, x.options());
    launch_mmq_regdirect_w4a16_gfx1201(x, w_rep, scales, w_zeros, out, N);
    return out;
}

at::Tensor mmq_fp8_moe_gemm_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel,
    bool weight_is_e2m1) {

    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && scales.is_cuda(),
                "x, w_packed, scales must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be (T, K)");
    TORCH_CHECK(w_packed.dim() == 3 && scales.dim() == 3,
                "w_packed (E,N,K/8) and scales (E,N,K/group) must be 3D");
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                expert_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "routing tensors must be int32");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() &&
                scales.is_contiguous(), "x, w_packed, scales must be contiguous");

    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(1);
    const int64_t P = sorted_token_ids.size(0);
    TORCH_CHECK(w_packed.size(2) * kPackFactor == K,
                "w_packed last dim must be K/8=", K / kPackFactor);
    TORCH_CHECK(P % block_m == 0, "P=", P, " not divisible by block_m=", block_m);
    TORCH_CHECK(expert_ids.size(0) == P / block_m,
                "expert_ids must have P/block_m=", P / block_m, " entries");
    TORCH_CHECK(w4a8::moe_kernel_valid(kernel),
                "moe kernel id must be 0 (scalar) / 6 (wmma) / 7 (gemv); got ", kernel);

    if (w_zeros.defined() && w_zeros.numel() > 0) {
        TORCH_CHECK(w_zeros.scalar_type() == at::kInt && w_zeros.dim() == 3 &&
                    w_zeros.size(1) * 8 == N && w_zeros.size(2) == scales.size(2),
                    "w_zeros must be (E, N/8, K/group) int32");
    }

    auto out = at::empty({P, N}, x.options());
    launch_mmq_fp8_moe_gemm_gfx1201(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, out, top_k, block_m, kernel, weight_is_e2m1);
    return out;
}

// Fused gemm1 + silu_and_mul. Runs the gated gemm1 (w13, N=2*inter = [gate|up])
// and the silu_and_mul activation in ONE kernel, returning the post-activation
// (P, inter) directly -- no separate (P,2*inter) out1 / (P,inter) buf2 HBM
// round-trip and no separate silu launch. Bit-exact to mmq_fp8_moe_gemm(w13) +
// torch.ops._C.silu_and_mul. v5/v6 only.
at::Tensor mmq_fp8_moe_gemm1_silu_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel,
    bool weight_is_e2m1) {

    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && scales.is_cuda(),
                "x, w_packed, scales must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be (T, K)");
    TORCH_CHECK(w_packed.dim() == 3 && scales.dim() == 3,
                "w_packed (E,2*inter,K/8) and scales (E,2*inter,K/group) must be 3D");
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                expert_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "routing tensors must be int32");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() &&
                scales.is_contiguous(), "x, w_packed, scales must be contiguous");
    TORCH_CHECK(static_cast<w4a8::MoeKernel>(kernel) == w4a8::MoeKernel::Wmma,
                "fused gemm1+silu needs the 'wmma' kernel; got id ", kernel);

    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(1);          // 2*inter
    const int64_t P = sorted_token_ids.size(0);
    TORCH_CHECK(N % 2 == 0, "w13 N must be 2*inter (even); got ", N);
    const int64_t inter = N / 2;
    TORCH_CHECK(w_packed.size(2) * kPackFactor == K,
                "w_packed last dim must be K/8=", K / kPackFactor);
    TORCH_CHECK(P % block_m == 0, "P=", P, " not divisible by block_m=", block_m);
    TORCH_CHECK(expert_ids.size(0) == P / block_m,
                "expert_ids must have P/block_m=", P / block_m, " entries");

    if (w_zeros.defined() && w_zeros.numel() > 0) {
        TORCH_CHECK(w_zeros.scalar_type() == at::kInt && w_zeros.dim() == 3 &&
                    w_zeros.size(1) * 8 == N && w_zeros.size(2) == scales.size(2),
                    "w_zeros must be (E, (2*inter)/8, K/group) int32");
    }

    auto out = at::empty({P, inter}, x.options());
    launch_mmq_fp8_moe_gemm1_silu_gfx1201(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, out, top_k, block_m, kernel, weight_is_e2m1);
    return out;
}

// Fused gemm2 + topk-weight + indirect atomic scatter. `x` is the (P, inter)
// post-activation buffer; the result is accumulated IN PLACE into the caller's
// pre-zeroed (M, N) fp32 `output` (one row per real token). No (P,N) materialise,
// no torch scatter. `sorted_token_ids` are the gemm1 (token,slot) ids.
void mmq_fp8_moe_gemm_scatter_forward(
    const at::Tensor& x,
    const at::Tensor& w_packed,
    const at::Tensor& scales,
    const at::Tensor& w_zeros,
    const at::Tensor& sorted_token_ids,
    const at::Tensor& expert_ids,
    const at::Tensor& num_tokens_post_padded,
    const at::Tensor& topk_weights,
    at::Tensor& output,
    int64_t top_k,
    int64_t block_m,
    int64_t kernel,
    bool weight_is_e2m1) {

    TORCH_CHECK(x.is_cuda() && w_packed.is_cuda() && scales.is_cuda(),
                "x, w_packed, scales must be CUDA");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "x must be fp16");
    TORCH_CHECK(w_packed.scalar_type() == at::kInt, "w_packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "scales must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be (P, K)");
    TORCH_CHECK(w_packed.dim() == 3 && scales.dim() == 3,
                "w_packed (E,N,K/8) and scales (E,N,K/group) must be 3D");
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                expert_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "routing tensors must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat &&
                output.scalar_type() == at::kFloat,
                "topk_weights and output must be fp32");
    TORCH_CHECK(x.is_contiguous() && w_packed.is_contiguous() &&
                scales.is_contiguous() && topk_weights.is_contiguous() &&
                output.is_contiguous(),
                "x, w_packed, scales, topk_weights, output must be contiguous");
    TORCH_CHECK(output.dim() == 2 && output.size(1) == w_packed.size(1),
                "output must be (M, N) with N=", w_packed.size(1));

    const int64_t K = x.size(1);
    const int64_t N = w_packed.size(1);
    const int64_t P = sorted_token_ids.size(0);
    const int64_t M = output.size(0);
    TORCH_CHECK(w_packed.size(2) * kPackFactor == K,
                "w_packed last dim must be K/8=", K / kPackFactor);
    TORCH_CHECK(P % block_m == 0, "P=", P, " not divisible by block_m=", block_m);
    TORCH_CHECK(expert_ids.size(0) == P / block_m,
                "expert_ids must have P/block_m=", P / block_m, " entries");
    TORCH_CHECK(topk_weights.numel() == M * top_k,
                "topk_weights must have M*top_k=", M * top_k, " entries");
    TORCH_CHECK(w4a8::moe_kernel_valid(kernel),
                "moe kernel id must be 0 (scalar) / 6 (wmma) / 7 (gemv); got ", kernel);

    if (w_zeros.defined() && w_zeros.numel() > 0) {
        TORCH_CHECK(w_zeros.scalar_type() == at::kInt && w_zeros.dim() == 3 &&
                    w_zeros.size(1) * 8 == N && w_zeros.size(2) == scales.size(2),
                    "w_zeros must be (E, N/8, K/group) int32");
    }

    launch_mmq_fp8_moe_gemm_scatter_gfx1201(
        x, w_packed, scales, w_zeros, sorted_token_ids, expert_ids,
        num_tokens_post_padded, topk_weights, output, top_k, block_m, kernel, weight_is_e2m1);
}

// Contention-free MoE reduce: gemm2 NON-scatter (P,N) -> weighted gather-reduce
// to (M,N) fp32. Replaces the atomic scatter at scale (no top_k-contended atomics).
at::Tensor mmq_fp8_moe_gather_reduce_forward(
    const at::Tensor& out2,                    // (P, N) fp16
    const at::Tensor& sorted_token_ids,        // (P,) int32
    const at::Tensor& topk_weights,            // (M*top_k,) fp32
    const at::Tensor& num_tokens_post_padded,  // (1,) int32
    int64_t top_k) {

    TORCH_CHECK(out2.is_cuda() && out2.scalar_type() == at::kHalf && out2.dim() == 2,
                "out2 must be (P,N) fp16 CUDA");
    TORCH_CHECK(sorted_token_ids.scalar_type() == at::kInt &&
                num_tokens_post_padded.scalar_type() == at::kInt,
                "sorted_token_ids / num_tokens_post_padded must be int32");
    TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights must be fp32");
    TORCH_CHECK(out2.is_contiguous() && topk_weights.is_contiguous(),
                "out2 and topk_weights must be contiguous");
    TORCH_CHECK(topk_weights.numel() % top_k == 0, "topk_weights.numel() must be M*top_k");

    const int64_t N = out2.size(1);
    const int64_t M = topk_weights.numel() / top_k;
    auto out = at::zeros({M, N}, out2.options().dtype(at::kFloat));
    launch_moe_gather_reduce_gfx1201(out2, sorted_token_ids, topk_weights,
                                     num_tokens_post_padded, out, top_k);
    return out;
}

}  // namespace

// pt2_compliant_tag: marks every op as safe for torch.compile / Dynamo full-graph
// capture (vLLM 0.22.69's aot_compile is fullgraph-strict and otherwise raises
// "unsupported operator: w4a8_fp8_wmma.*" the moment the kernel actually engages).
// The matching fake/meta kernels (output-shape inference) live in __init__.py.
TORCH_LIBRARY(w4a8_fp8_wmma, m) {
    m.def("mmq_fp8_gemm(Tensor x, Tensor w_packed, Tensor scales, Tensor w_zeros, int kernel, bool weight_is_e2m1=False) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_regdirect_fp8(Tensor x, Tensor w_rep, Tensor scales, Tensor w_zeros, int N) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_regdirect_f16(Tensor x, Tensor w_rep, Tensor scales, Tensor w_zeros, int N) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_regdirect_w4a16(Tensor x, Tensor w_rep, Tensor scales, Tensor w_zeros, int N) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_fp8_moe_gemm(Tensor x, Tensor w_packed, Tensor scales, Tensor w_zeros, "
          "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, "
          "int top_k, int block_m, int kernel, bool weight_is_e2m1=False) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_fp8_moe_gemm1_silu(Tensor x, Tensor w_packed, Tensor scales, Tensor w_zeros, "
          "Tensor sorted_token_ids, Tensor expert_ids, Tensor num_tokens_post_padded, "
          "int top_k, int block_m, int kernel, bool weight_is_e2m1=False) -> Tensor",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_fp8_moe_gemm_scatter(Tensor x, Tensor w_packed, Tensor scales, "
          "Tensor w_zeros, Tensor sorted_token_ids, Tensor expert_ids, "
          "Tensor num_tokens_post_padded, Tensor topk_weights, Tensor(a!) output, "
          "int top_k, int block_m, int kernel, bool weight_is_e2m1=False) -> ()",
          {at::Tag::pt2_compliant_tag});
    m.def("mmq_fp8_moe_gather_reduce(Tensor out2, Tensor sorted_token_ids, "
          "Tensor topk_weights, Tensor num_tokens_post_padded, int top_k) -> Tensor",
          {at::Tag::pt2_compliant_tag});
}

TORCH_LIBRARY_IMPL(w4a8_fp8_wmma, CUDA, m) {
    m.impl("mmq_fp8_gemm", &mmq_fp8_gemm_forward);
    m.impl("mmq_regdirect_fp8", &mmq_regdirect_fp8_forward);
    m.impl("mmq_regdirect_f16", &mmq_regdirect_f16_forward);
    m.impl("mmq_regdirect_w4a16", &mmq_regdirect_w4a16_forward);
    m.impl("mmq_fp8_moe_gemm", &mmq_fp8_moe_gemm_forward);
    m.impl("mmq_fp8_moe_gemm1_silu", &mmq_fp8_moe_gemm1_silu_forward);
    m.impl("mmq_fp8_moe_gemm_scatter", &mmq_fp8_moe_gemm_scatter_forward);
    m.impl("mmq_fp8_moe_gather_reduce", &mmq_fp8_moe_gather_reduce_forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "W4A8-FP8 MMQ kernel for gfx1201. "
              "torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm(x, w_packed, scales, w_zeros, kernel)";
}
