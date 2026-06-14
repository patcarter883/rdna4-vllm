// Torch bindings for the fused CCA decode conv + state-update HIP op (gfx1201).
//
// Exposes torch.ops.zaya_cca.conv_state_decode(qk_new, conv_states, slot,
// is_pad, w0, b0, w1, b1) -> qk_out, and mutates conv_states in place.
// Registered as an opaque custom op so torch.compile steps over it (no eager
// graph-break to thousands of pointwise launches).

#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>

void launch_cca_conv_state_decode(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& slot,
    const at::Tensor& is_pad,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1,
    at::Tensor& qk_out);

void launch_cca_decode_qk(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& slot,
    const at::Tensor& is_pad,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1,
    const at::Tensor& temp_eff,
    at::Tensor& qk_out,
    int64_t num_q,
    int64_t gqa,
    int64_t latent_q,
    double sqrt_d);

namespace {

at::Tensor cca_decode_qk(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& slot,
    const at::Tensor& is_pad,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1,
    const at::Tensor& temp_eff,
    int64_t num_q,
    int64_t gqa,
    int64_t latent_q,
    double sqrt_d) {
  TORCH_CHECK(qk_new.is_cuda() && qk_new.scalar_type() == at::kFloat &&
              qk_new.is_contiguous(), "qk_new must be contiguous fp32 CUDA");
  TORCH_CHECK(conv_states.scalar_type() == at::kFloat, "conv_states must be fp32");
  TORCH_CHECK(slot.scalar_type() == at::kLong, "slot must be int64");
  TORCH_CHECK(is_pad.scalar_type() == at::kBool, "is_pad must be bool");
  auto qk_out = at::empty_like(qk_new);
  launch_cca_decode_qk(qk_new, conv_states, slot, is_pad, w0.contiguous(),
                       b0.contiguous(), w1.contiguous(), b1.contiguous(),
                       temp_eff.contiguous(), qk_out, num_q, gqa, latent_q,
                       sqrt_d);
  return qk_out;
}

at::Tensor conv_state_decode(
    const at::Tensor& qk_new,
    at::Tensor& conv_states,
    const at::Tensor& slot,
    const at::Tensor& is_pad,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1) {

  TORCH_CHECK(qk_new.is_cuda() && conv_states.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(qk_new.scalar_type() == at::kFloat, "qk_new must be fp32");
  TORCH_CHECK(conv_states.scalar_type() == at::kFloat, "conv_states must be fp32");
  TORCH_CHECK(qk_new.is_contiguous(), "qk_new must be contiguous");
  // conv_states may be a non-contiguous cache view; the kernel uses its strides.
  TORCH_CHECK(slot.scalar_type() == at::kLong, "slot must be int64");
  TORCH_CHECK(is_pad.scalar_type() == at::kBool, "is_pad must be bool");

  auto qk_out = at::empty_like(qk_new);
  launch_cca_conv_state_decode(qk_new, conv_states, slot, is_pad,
                               w0.contiguous(), b0.contiguous(),
                               w1.contiguous(), b1.contiguous(), qk_out);
  return qk_out;
}

}  // namespace

TORCH_LIBRARY(zaya_cca, m) {
  m.def(
      "conv_state_decode(Tensor qk_new, Tensor(a!) conv_states, Tensor slot, "
      "Tensor is_pad, Tensor w0, Tensor b0, Tensor w1, Tensor b1) -> Tensor");
  m.def(
      "cca_decode_qk(Tensor qk_new, Tensor(a!) conv_states, Tensor slot, "
      "Tensor is_pad, Tensor w0, Tensor b0, Tensor w1, Tensor b1, "
      "Tensor temp_eff, int num_q, int gqa, int latent_q, float sqrt_d) "
      "-> Tensor");
}

TORCH_LIBRARY_IMPL(zaya_cca, CUDA, m) {
  m.impl("conv_state_decode", &conv_state_decode);
  m.impl("cca_decode_qk", &cca_decode_qk);
}
