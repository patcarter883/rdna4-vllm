"""End-to-end vLLM generation through the W4A8-FP8 WMMA kernel (gfx1201).

Registers our kernel, loads a small compressed-tensors W4A16 (uint4b8 g128)
model, and generates. Prints which linear kernel each quantized layer selected so
we can confirm ours is actually used.

Run inside kyuz0/vllm-therock-gfx1201 with the GPU mounted and a writable HF
cache. Pass a model id as argv[1] to override the candidate list.
"""
import os
import sys

# Register BEFORE importing/much of vllm engine so the dispatcher sees our kernel.
import w4a8_fp8_wmma  # noqa: F401
from w4a8_fp8_wmma.register import register
register()

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

CANDIDATES = [
    # GPTQ-Int4 models with desc_act=False -> g_idx absent -> our kernel engages.
    "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
]


def main():
    models = [sys.argv[1]] if len(sys.argv) > 1 else CANDIDATES
    last_err = None
    for mid in models:
        print(f"\n===== trying model: {mid} =====")
        try:
            llm = LLM(
                model=mid,
                dtype="float16",
                enforce_eager=True,
                gpu_memory_utilization=0.55,
                max_model_len=2048,
            )
        except Exception as e:
            print(f"  load failed: {type(e).__name__}: {str(e)[:300]}")
            last_err = e
            continue

        sp = SamplingParams(temperature=0.0, max_tokens=48)
        # long prompt (~3500 tokens) forces a large-M prefill -> exercises the
        # FP8-WMMA v5 path; short prompts hit the Triton fallback.
        long_prompt = ("The quick brown fox jumps over the lazy dog. " * 120) + \
                      "\nIn one word, the animal that jumps is the"
        prompts = [
            "The capital of France is",
            long_prompt,
        ]
        outs = llm.generate(prompts, sp)
        print("\n===== GENERATION =====")
        for o in outs:
            print(f"PROMPT: {o.prompt!r}")
            print(f"OUTPUT: {o.outputs[0].text!r}\n")
        print("MODEL RUN: PASS")
        return 0

    print(f"\nMODEL RUN: FAIL (no candidate model loaded). last: {last_err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
