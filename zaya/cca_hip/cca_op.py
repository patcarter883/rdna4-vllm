"""Python entry point for the fused CCA decode conv+state HIP op.

Importing this registers ``torch.ops.zaya_cca.conv_state_decode`` (via the C++
TORCH_LIBRARY in the compiled ``zaya_cca_C`` extension) and a **fake/meta**
implementation. The fake function is what lets torch.compile / Inductor treat
the op as opaque without graph-breaking — without a registered meta, Inductor
can't infer the output shape/dtype and shatters the graph (the exact failure
this kernel exists to avoid).
"""
import glob
import os

import torch

# Pure-TORCH_LIBRARY extension (no pybind PyInit) → load via torch.ops, not import.
_so = glob.glob(os.path.join(os.path.dirname(__file__), "zaya_cca_C*.so"))
if not _so:
    raise ImportError("zaya_cca_C extension not built; run setup.py build_ext --inplace")
torch.ops.load_library(_so[0])


@torch.library.register_fake("zaya_cca::conv_state_decode")
def _conv_state_decode_fake(qk_new, conv_states, slot, is_pad, w0, b0, w1, b1):
    # Mutates conv_states in place (declared Tensor(a!) in the schema);
    # returns qk_out shaped/dtyped like qk_new.
    return torch.empty_like(qk_new)


@torch.library.register_fake("zaya_cca::cca_decode_qk")
def _cca_decode_qk_fake(
    qk_new, conv_states, slot, is_pad, w0, b0, w1, b1, temp_eff,
    num_q, gqa, latent_q, sqrt_d,
):
    return torch.empty_like(qk_new)


conv_state_decode = torch.ops.zaya_cca.conv_state_decode
cca_decode_qk = torch.ops.zaya_cca.cca_decode_qk
