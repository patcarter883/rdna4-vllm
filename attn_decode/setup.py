"""Build the native flash-DECODE attention HIP op for gfx1201 (RDNA4).

Builds torch.ops.attn_decode.flash_decode as ONE AOT-compiled .so — no Triton JIT/autotune. Pure
FMA + warp-shuffle (no WMMA / no rocwmma needed). Mirrors the gdn_hip / attn_hip recipe.

Usage (inside the combined ROCm image; CPU-only, no GPU needed to compile):
    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="attn_decode",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="attn_decode_C",
            sources=["bindings.cpp", "attn_decode_kernels.hip"],
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
