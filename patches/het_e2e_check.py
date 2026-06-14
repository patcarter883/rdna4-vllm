"""Greedy-equivalence check for heterogeneous TP.

Loads the model at TP=2, greedy-decodes fixed prompts, writes token ids to JSON. Run
once with VLLM_TP_CU_WEIGHTS unset (even baseline) and once ="64,56" (het). The token
ids MUST be identical — het sharding only changes *which rank holds which channels*, not
the math. Any divergence => a het offset/stamp bug. (Order 64,56 vs 56,64 is irrelevant
to correctness; it only affects the perf balance.)

Env: HET_MODEL (hf id), HET_TAG (output suffix), HET_OUT (dir, default /out).
"""
import json
import os

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "In one sentence, explain what a GPU does:",
    "List three prime numbers:",
    "2 + 2 =",
]


def main():
    # Must be under __main__ guard: vLLM uses spawn for TP workers, which re-imports
    # this module in each child.
    model = os.environ["HET_MODEL"]
    tag = os.environ.get("HET_TAG", "out")
    out = os.environ.get("HET_OUT", "/out")
    tp = int(os.environ.get("HET_TP", "2"))
    maxlen = int(os.environ.get("HET_MAXLEN", "2048"))
    util = float(os.environ.get("HET_GPUUTIL", "0.85"))

    kwargs = dict(
        model=model,
        tensor_parallel_size=tp,
        dtype="float16",
        max_model_len=maxlen,
        gpu_memory_utilization=util,
        enforce_eager=True,          # correctness check; skip graph capture
        trust_remote_code=True,
    )
    # Only for multimodal models (e.g. Qwen3.6): skip the ViT dummy forward in the
    # memory profiler, which hangs on RDNA4 via the AOTriton SDPA block-(0,0,0) hazard
    # (DIARY "Blocker 3"). Harmless to omit for text-only models.
    if os.environ.get("HET_LIMIT_MM", "0") == "1":
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0,
                        max_tokens=int(os.environ.get("HET_MAXTOK", "64")))
    outs = llm.generate(PROMPTS, sp)
    result = [
        {"prompt": o.prompt,
         "token_ids": list(o.outputs[0].token_ids),
         "text": o.outputs[0].text}
        for o in outs
    ]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"het_{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{tag}] wrote {path}")
    for r in result:
        print(f"[{tag}] {r['token_ids'][:8]}... :: {r['text'][:48]!r}")


if __name__ == "__main__":
    main()
