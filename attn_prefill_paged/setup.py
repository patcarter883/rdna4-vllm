"""Build the native PAGED/chunked-prefill flash-attention HIP op for gfx1201.

    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="attn_prefill_paged",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="attn_prefill_paged_C",
            sources=["bindings.cpp", "attn_prefill_paged_kernels.hip"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": ["-O3", "-std=c++17", f"--offload-arch={TARGET_ARCH}",
                         "-Wno-unused-result", "-Wno-unused-variable"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
