"""End-to-end vLLM generation for an AWQ-4bit MoE model through our kernel.

Confirms Task 6 (MoE vLLM wiring): an AWQ MoE model's expert layers route
AWQMarlinMoEMethod -> select_wna16_moe_backend (our patched oracle) ->
W4A8Fp8WmmaExperts, convert their weights to our op layout, and generate
coherent text via the grouped FP8-WMMA GEMM (gemm1 -> silu_and_mul -> gemm2 ->
topk scatter-reduce).

What to look for in the log:
  "[w4a8_fp8_wmma] AWQ MoE hook installed (gfx12x): ..."     (at register())
  "[w4a8_fp8_wmma] AWQ MoE -> W4A8Fp8WmmaExperts (g=..)"      (first expert layer)
  (vLLM may also log "Using 'MARLIN' WNA16 MoE backend." from the stock oracle —
   that is the pre-override selection; our hook then replaces it with our experts
   class, so generation actually runs on the grouped FP8-WMMA path.)

Compare against the Marlin baseline by disabling just the MoE hook:
  VLLM_ROCM_W4A8_FP8_WMMA_MOE=0 python moe_model_test.py <model>

Debug with the scalar golden grouped kernel instead of the WMMA tile:
  VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL=scalar python moe_model_test.py <model>

Run inside kyuz0/vllm-therock-gfx1201 with the GPU + HF cache mounted. The 35B
final-goal model is too big for 16GB; use a small cached AWQ MoE, e.g.
  cyankiwi/Mellum2-12B-A2.5B-base-AWQ-INT4   (~6GB)
  Qwen/Qwen1.5-MoE-A2.7B-Chat-AWQ            (if cached)
  python moe_model_test.py <model_id>
"""
import os
import sys

import w4a8_fp8_wmma  # noqa: F401  (loads torch.ops + the general plugin)
from w4a8_fp8_wmma.register import register

register()

from vllm import LLM, SamplingParams  # noqa: E402

DEFAULT_MODEL = "cyankiwi/Mellum2-12B-A2.5B-base-AWQ-INT4"


def main():
    mid = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    moe_on = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE", "1") == "1"
    kernel = os.environ.get("VLLM_ROCM_W4A8_FP8_WMMA_MOE_KERNEL", "wmma")
    print(f"===== AWQ MoE end-to-end: {mid} =====")
    print(f"MoE hook: {f'ON (grouped FP8-WMMA, {kernel})' if moe_on else 'OFF (Marlin baseline)'}")

    llm = LLM(
        model=mid,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        trust_remote_code=True,
    )

    sp = SamplingParams(temperature=0.0, max_tokens=48)
    prompts = [
        "The capital of France is",
        "Write a Python function to reverse a string:",
        "Explain in one sentence why the sky is blue:",
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
    print("MOE MODEL RUN:", "PASS" if ok else "FAIL (empty/degenerate output)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
