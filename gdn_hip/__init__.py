"""gdn_hip — native HIP kernels for the Gated Delta Net linear-attention path on gfx1201 (RDNA4).

A standalone, framework-agnostic torch.ops extension shared by minisgl-rdna4 and vllm-gfx1201:
AOT-compiled once (no Triton JIT/autotune), it replaces the fla-Triton GDN kernels
(chunk_gated_delta_rule + the decode SSM kernel + causal_conv1d + rmsnorm_gated). Import `op` to
load the .so and register the ops as torch.ops.gdn_hip.*.
"""
from .op import (  # noqa: F401
    causal_conv1d_fwd,
    causal_conv1d_update,
    gdn_decode,
    gdn_prefill,
    rmsnorm_gated,
)
