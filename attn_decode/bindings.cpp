// Torch bindings for the native flash-DECODE attention HIP kernel on gfx1201.
//
// Framework-agnostic op under attn_decode:: (callable as torch.ops.attn_decode.* from minisgl OR
// vLLM). Separate library namespace from the prefill kernel (attn_hip::) so both .so can coexist;
// they fold into one attn_hip library when the prefill + decode work merges. Opaque custom op + a
// registered fake/meta (op.py) so torch.compile/Inductor steps over it without graph-breaking.
//
// v0 is bf16 dense (non-paged) decode. See attn_decode_kernels.hip for scope + provenance.

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

// ---- launchers defined in attn_decode_kernels.hip ----
void launch_flash_decode(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                         double, int64_t);
void launch_flash_decode_paged(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, const at::Tensor&, at::Tensor&, double, int64_t);
void launch_flash_decode_paged_fp8(const at::Tensor&, const at::Tensor&, const at::Tensor&,
                                   const at::Tensor&, const at::Tensor&, at::Tensor&, double, double,
                                   double, int64_t);

namespace {

at::Tensor flash_decode(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                        double scale, int64_t sliding_window) {
  TORCH_CHECK(q.dim() == 3, "attn_decode: q must be [B, num_q_heads, head_dim]");
  TORCH_CHECK(k.dim() == 4 && v.dim() == 4, "attn_decode: k/v must be [B, Skv, num_kv_heads, head_dim]");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "attn_decode v0 is bf16-only");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "attn_decode v0 expects contiguous q/k/v");
  auto out = at::empty_like(q);
  launch_flash_decode(q, k, v, out, scale, sliding_window);
  return out;
}

at::Tensor flash_decode_paged(const at::Tensor& q, const at::Tensor& k_cache,
                              const at::Tensor& v_cache, const at::Tensor& block_table,
                              const at::Tensor& context_lens, double scale, int64_t sliding_window) {
  TORCH_CHECK(q.dim() == 3, "attn_decode: q must be [B, num_q_heads, head_dim]");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "attn_decode: k/v_cache must be [num_blocks, block_size, num_kv_heads, head_dim]");
  TORCH_CHECK(block_table.dim() == 2 && context_lens.dim() == 1, "bad block_table/context_lens");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "attn_decode v0 is bf16-only");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
              "block_table/context_lens must be int32");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous(),
              "attn_decode v0 expects contiguous inputs");
  auto out = at::empty_like(q);
  launch_flash_decode_paged(q, k_cache, v_cache, block_table, context_lens, out, scale,
                            sliding_window);
  return out;
}

at::Tensor flash_decode_paged_fp8(const at::Tensor& q, const at::Tensor& k_cache,
                                  const at::Tensor& v_cache, const at::Tensor& block_table,
                                  const at::Tensor& context_lens, double scale, double k_descale,
                                  double v_descale, int64_t sliding_window) {
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 4 && v_cache.dim() == 4, "bad q/k/v_cache dims");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "attn_decode: q must be bf16");
  TORCH_CHECK(k_cache.scalar_type() == at::kFloat8_e4m3fn && v_cache.scalar_type() == at::kFloat8_e4m3fn,
              "attn_decode fp8: k/v_cache must be float8_e4m3fn");
  TORCH_CHECK(block_table.scalar_type() == at::kInt && context_lens.scalar_type() == at::kInt,
              "block_table/context_lens must be int32");
  TORCH_CHECK(q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous(),
              "attn_decode fp8 expects contiguous inputs");
  auto out = at::empty_like(q);
  launch_flash_decode_paged_fp8(q, k_cache, v_cache, block_table, context_lens, out, scale,
                                k_descale, v_descale, sliding_window);
  return out;
}

}  // namespace

TORCH_LIBRARY(attn_decode, m) {
  m.def("flash_decode(Tensor q, Tensor k, Tensor v, float scale, int sliding_window) -> Tensor");
  m.def("flash_decode_paged(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, "
        "Tensor context_lens, float scale, int sliding_window) -> Tensor");
  m.def("flash_decode_paged_fp8(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_table, "
        "Tensor context_lens, float scale, float k_descale, float v_descale, int sliding_window) "
        "-> Tensor");
}

TORCH_LIBRARY_IMPL(attn_decode, CUDA, m) {
  m.impl("flash_decode", flash_decode);
  m.impl("flash_decode_paged", flash_decode_paged);
  m.impl("flash_decode_paged_fp8", flash_decode_paged_fp8);
}
