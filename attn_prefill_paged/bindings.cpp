// Torch bindings for the native PAGED/chunked-prefill flash-attention HIP kernel on gfx1201.
// torch.ops.attn_prefill_paged.flash_prefill_paged — framework-agnostic (minisgl OR vLLM), opaque to
// torch.compile via the fake in op.py. Mirrors the attn_decode / attn_hip TORCH_LIBRARY pattern.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_flash_prefill_paged(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                                double, int64_t, int64_t, int64_t, int64_t);
void launch_flash_prefill_paged_fp8(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                    const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                                    double, double, double, int64_t, int64_t, int64_t, int64_t);

namespace {

at::Tensor flash_prefill_paged(const at::Tensor& q, const at::Tensor& k_cache,
                               const at::Tensor& v_cache, const at::Tensor& block_table,
                               const at::Tensor& cu_seqlens_q, const at::Tensor& context_lens,
                               double scale, int64_t causal, int64_t sliding_window,
                               int64_t max_seqlen_q, int64_t kv_block_stride) {
  TORCH_CHECK(q.dim() == 3, "q must be [total_q_tokens, num_q_heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "k/v_cache must be [num_blocks, block_size, num_kv_heads, head_dim]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "bf16-only");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && cu_seqlens_q.scalar_type() == at::kInt
              && context_lens.scalar_type() == at::kInt, "block_table/cu_seqlens_q/context_lens int32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");  // k/v_cache may be strided (kv_block_stride)
  auto out = at::empty_like(q);
  launch_flash_prefill_paged(q, k_cache, v_cache, block_table, cu_seqlens_q, context_lens, out,
                             scale, causal, sliding_window, kv_block_stride, max_seqlen_q);
  return out;
}

at::Tensor flash_prefill_paged_fp8(const at::Tensor& q, const at::Tensor& k_cache,
                                   const at::Tensor& v_cache, const at::Tensor& block_table,
                                   const at::Tensor& cu_seqlens_q, const at::Tensor& context_lens,
                                   double scale, double k_descale, double v_descale, int64_t causal,
                                   int64_t sliding_window, int64_t max_seqlen_q,
                                   int64_t kv_block_stride) {
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4, "bad q/k/v_cache dims");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(k_cache.scalar_type() == at::kFloat8_e4m3fn && v_cache.scalar_type() == at::kFloat8_e4m3fn,
              "fp8: k/v_cache must be float8_e4m3fn");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && cu_seqlens_q.scalar_type() == at::kInt
              && context_lens.scalar_type() == at::kInt, "block_table/cu_seqlens_q/context_lens int32");
  TORCH_CHECK(q.is_contiguous(), "q must be contiguous");  // k/v_cache may be strided
  auto out = at::empty_like(q);
  launch_flash_prefill_paged_fp8(q, k_cache, v_cache, block_table, cu_seqlens_q, context_lens, out,
                                 scale, k_descale, v_descale, causal, sliding_window, kv_block_stride,
                                 max_seqlen_q);
  return out;
}

}  // namespace

TORCH_LIBRARY(attn_prefill_paged, m) {
  m.def("flash_prefill_paged(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, "
        "Tensor cu_seqlens_q, Tensor context_lens, float scale, int causal, int sliding_window, "
        "int max_seqlen_q, int kv_block_stride=0) -> Tensor");
  m.def("flash_prefill_paged_fp8(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, "
        "Tensor cu_seqlens_q, Tensor context_lens, float scale, float k_descale, float v_descale, "
        "int causal, int sliding_window, int max_seqlen_q, int kv_block_stride=0) -> Tensor");
}

TORCH_LIBRARY_IMPL(attn_prefill_paged, CUDA, m) {
  m.impl("flash_prefill_paged", flash_prefill_paged);
  m.impl("flash_prefill_paged_fp8", flash_prefill_paged_fp8);
}
