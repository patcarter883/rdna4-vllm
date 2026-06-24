"""End-to-end token-diff: gdn_hip GDN path vs the fla-Triton GDN path on Qwen3.5-4B.

Greedy-decodes a set of prompts twice in the SAME process is not possible (the GDN path is chosen
at engine-init time from VLLM_GDN_HIP + the state dtype is fixed at cache-alloc), so this script
runs ONE engine and dumps the greedy token ids; invoke it twice (VLLM_GDN_HIP=0 then =1) and diff
the two json dumps. fp32-vs-fp32 (the gdn_hip path forces fp32 state) vs the fla bf16/fp32 Triton
path: expect near-identical greedy ids (a late divergence under sampling noise is acceptable; an
early/structural divergence means a mapping bug — conv [q|k|v] split, GQA head map, or strides).

Run inside vllm22-w4a8:gdnhip under a 1-card lease (see the runner in the PR body).
"""
import json
import os
import sys

from vllm import LLM, SamplingParams

MODEL = os.environ.get("GDN_DIFF_MODEL", "Qwen/Qwen3.5-4B")
OUT = os.environ.get("GDN_DIFF_OUT", "/tmp/gdn_diff_out.json")

PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "Write a short haiku about the ocean.",
    "List three prime numbers greater than ten:",
    "Once upon a time, in a distant galaxy,",
    "The derivative of x squared with respect to x is",
]


def main() -> None:
    flag = os.environ.get("VLLM_GDN_HIP", "0")
    print(f"=== token-diff: VLLM_GDN_HIP={flag} model={MODEL} ===", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=48)
    outs = llm.generate(PROMPTS, sp)
    dump = []
    for o in outs:
        comp = o.outputs[0]
        dump.append({"prompt": o.prompt, "token_ids": list(comp.token_ids), "text": comp.text})
    with open(OUT, "w") as f:
        json.dump({"flag": flag, "model": MODEL, "results": dump}, f, indent=2)
    print(f"wrote {OUT} ({len(dump)} prompts)", flush=True)


if __name__ == "__main__":
    main()
