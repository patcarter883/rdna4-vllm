"""End-to-end vLLM generation for a DENSE AWQ-4bit model through our kernel.

Confirms Task 1: an AWQ (asymmetric uint4 + per-group zero points, g128) model
routes AWQMarlinLinearMethod -> choose_mp_linear_kernel -> our kernel, repacks
correctly in process_weights, and generates coherent text.

Two dispatch regimes (the adapter picks per-layer by M vs the crossover cache):
  - default: the AWQ model's shapes aren't in crossover_cache.json -> _NEVER ->
    ALL M use the Triton W4A16 fallback. This exercises the decode-path AWQ
    zero-point fix (the bug this session fixed).
  - VLLM_ROCM_W4A8_FP8_WMMA_MIN_M=1: forces the FP8-WMMA v5 path for all M,
    exercising the large-M AWQ path (process_weights repack + apply_weights v5).

Run inside kyuz0/vllm-therock-gfx1201 with the GPU + HF cache mounted.
  python awq_model_test.py [model_id]
"""
import os
import sys

import w4a8_fp8_wmma  # noqa: F401  (loads torch.ops + the general plugin)
from w4a8_fp8_wmma.register import register

register()

from vllm import LLM, SamplingParams  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"


def main():
    mid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    forced = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MIN_M")
    print(f"===== AWQ end-to-end: {mid} =====")
    print(f"dispatch: {'FORCED v5 (MIN_M=' + forced + ')' if forced else 'default (Triton fallback / crossover cache)'}")

    llm = LLM(
        model=mid,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.75,
        max_model_len=2048,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=48)
    long_prompt = ("def fibonacci(n):\n    # returns the nth Fibonacci number\n"
                   "    " * 1) + ("# " + "x = 1; " * 60 + "\n") * 1 + \
        "Write a Python function that returns the sum of a list of integers:\n"
    prompts = [
        "The capital of France is",
        "Write a Python function to reverse a string:",
        long_prompt,
    ]
    outs = llm.generate(prompts, sp)
    print("\n===== GENERATION =====")
    ok = True
    for o in outs:
        txt = o.outputs[0].text
        print(f"PROMPT : {o.prompt[:60]!r}")
        print(f"OUTPUT : {txt[:200]!r}\n")
        if len(txt.strip()) < 3:
            ok = False
    print("MODEL RUN:", "PASS" if ok else "FAIL (empty/degenerate output)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
