#!/usr/bin/env python3
"""TP=2 throughput benchmark for the gfx1201 stack (offline vLLM LLM API).

Reproduces the reported figure for cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit on two
gfx1201 cards: ~298 decode tok/s, ~1887 total (prefill+decode) tok/s, on a WARM
Triton cache. Does a warmup generate() (which JIT-compiles decode + MoE kernels)
then times a second generate() so the number isn't compile-contaminated.

Run INSIDE the image (it needs the gfx1201 vLLM/aiter/torch), e.g.:

  docker compose --profile tp2-baseline run --rm --entrypoint python3 \
    tp2-baseline /workspace/test/bench_tp2.py

or copy it into a running container. Honors the same env as the compose profiles
(ROCR_VISIBLE_DEVICES, CU_NUM, VLLM_ROCM_USE_W4A8_FP8_WMMA, ...).

Env knobs: MODEL, TP, N_PROMPTS, OUT_TOKENS, GPU_MEM_UTIL, MAX_MODEL_LEN.
"""
import os
import time


def main():
    from vllm import LLM, SamplingParams

    model = os.environ.get("MODEL", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit")
    tp = int(os.environ.get("TP", "2"))
    n_prompts = int(os.environ.get("N_PROMPTS", "32"))
    out_tokens = int(os.environ.get("OUT_TOKENS", "128"))
    gpu_mem = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
    max_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))

    # A non-trivial shared prefix so prefill is meaningful (~hundreds of tokens).
    base = ("You are a careful systems engineer. Explain, step by step and with "
            "concrete detail, the following topic for an expert audience. ") * 8
    prompts = [f"{base}\n\nTopic #{i}: low-level GPU kernel scheduling on RDNA4."
               for i in range(n_prompts)]
    sp = SamplingParams(temperature=0.0, max_tokens=out_tokens, ignore_eos=True)

    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        dtype="float16",
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_len,
        max_num_batched_tokens=max_len,
    )

    print("=== warmup generate (JIT-compiles decode/MoE; not timed) ===", flush=True)
    llm.generate(prompts[:4], sp)

    print("=== timed generate ===", flush=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0

    # vLLM gives us the prompt token counts; outputs are fixed at out_tokens.
    prompt_toks = sum(len(o.prompt_token_ids) for o in outs)
    gen_toks = sum(len(c.token_ids) for o in outs for c in o.outputs)
    total = prompt_toks + gen_toks
    print(f"\nprompts={len(outs)}  wall={dt:.2f}s")
    print(f"prefill_tok={prompt_toks}  decode_tok={gen_toks}  total_tok={total}")
    print(f"out_tok_s={gen_toks / dt:.1f}   (decode throughput)")
    print(f"total_tok_s={total / dt:.1f}   (prefill+decode throughput)")


if __name__ == "__main__":   # vLLM v1 spawns the engine core — guard is mandatory
    main()
