"""Build script for the W4A8-FP8 MMQ HIP custom op targeting gfx1201 (RDNA4).

Builds a Python extension `w4a8_fp8_wmma._C` exposing
`torch.ops.w4a8_fp8_wmma.mmq_fp8_gemm`.

Usage (on the gfx1201 host, with the TheRock build env active):
    source /home/pat/code/vllm-rocm714-gfx1250/activate-build-env.sh
    cd vllm/csrc/quantization/w4a8_fp8_wmma
    python setup.py build_ext --inplace
    python test_correctness.py

GPU_ARCHS (set by activate-build-env.sh) overrides the default arch.
"""
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Accept one or more RDNA4 archs (";" or "," separated). Default builds a fat
# binary covering BOTH RDNA4 dies — gfx1200 (Navi44: RX 9060 XT / 9060) and
# gfx1201 (Navi48: RX 9070 XT / 9070) — so one .so runs on either card. AMD code
# objects are arch-exact, so a single-arch build will NOT load on the other die.
TARGET_ARCHS = [a for a in os.environ.get("GPU_ARCHS", "gfx1200;gfx1201")
                .replace(",", ";").split(";") if a]

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17", "-fPIC"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        *[f"--offload-arch={a}" for a in TARGET_ARCHS],
        "-Wno-unused-result",
        "-Wno-unused-variable",
    ],
}

setup(
    name="w4a8_fp8_wmma",
    version="0.1.0",
    description=(
        "W4A8-FP8 MMQ kernel for AMD RDNA4 (gfx1201). Expands packed int4 "
        "weights to fp8 e4m3 in-register and feeds the FP8 WMMA units. "
        "v0 scalar fp8 reference; v1 WMMA is on-device work-in-progress."
    ),
    packages=find_packages(),
    package_data={"w4a8_fp8_wmma": ["crossover_cache.json"]},
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="w4a8_fp8_wmma._C",
            sources=["bindings.cpp", "w4a8_fp8_wmma_kernel.hip", "moe_kernel.hip"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    # vLLM loads general plugins in EVERY process (incl. the EngineCore worker),
    # so this registers our kernel into the dispatcher where the model actually
    # loads — not just the parent process.
    entry_points={
        "vllm.general_plugins": [
            "w4a8_fp8_wmma_register = w4a8_fp8_wmma.register:register",
        ],
    },
)
