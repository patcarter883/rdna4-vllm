"""Build the native GDN (gated delta net) HIP custom ops for gfx1201 (RDNA4).

Builds torch.ops.gdn_hip.{gdn_decode,gdn_prefill,causal_conv1d_update,causal_conv1d_fwd,
rmsnorm_gated} as ONE AOT-compiled .so — no per-shape Triton JIT/autotune ever again. Mirrors the
proven zaya_cca / w4a8_fp8_wmma cpp_extension recipe.

Usage (inside the combined ROCm image; CPU-only, no GPU needed to compile):
    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="gdn_hip",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="gdn_hip_C",
            sources=["bindings.cpp", "gdn_kernels.hip"],
            include_dirs=["/opt/rocm-7.2.1/include"],  # rocwmma headers (gdn_prefill_wmma matrix-core)
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    f"--offload-arch={TARGET_ARCH}",
                    "-Wno-unused-result",
                    "-Wno-unused-variable",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
