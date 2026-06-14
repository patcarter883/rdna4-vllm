"""Build the fused CCA decode conv+state HIP custom op for gfx1201 (RDNA4).

Builds torch.ops.zaya_cca.conv_state_decode. Mirrors the kernel agent's
w4a8_fp8_wmma cpp_extension recipe.

Usage (in the vllm-gfx1201 container, or the TheRock build env):
    GPU_ARCHS=gfx1201 python setup.py build_ext --inplace
"""
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TARGET_ARCH = os.environ.get("GPU_ARCHS", "gfx1201").split(";")[0]

setup(
    name="zaya_cca",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="zaya_cca_C",
            sources=["bindings.cpp", "cca_kernel.hip"],
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
