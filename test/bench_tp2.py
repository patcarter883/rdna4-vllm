#!/usr/bin/env python3
"""TP=2 throughput benchmark for the gfx1201 stack (offline vLLM LLM API).

Reproduces the reported figure for cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit on two
gfx1201 cards: ~298 decode tok/s, ~1887 total (prefill+decode) tok/s, on a WARM
Triton cache. Does a warmup generate() (which JIT-compiles decode + MoE kernels)
then times a second generate() so the number isn't compile-contaminated.

Run INSIDE the image (it needs the gfx1201 vLLM/aiter/torch). The turnkey way is
the `bench` compose profile, which mounts this script + a results dir and wires
the same TP=2 / het-TP / W4A8 env as `serve`:

  ./scripts/bench.sh                 # wrapper: stamps git SHA, appends results.jsonl
  # or directly:
  docker compose --profile bench run --rm bench

or copy it into a running container. Honors the same env as the compose profiles
(ROCR_VISIBLE_DEVICES, CU_NUM, VLLM_ROCM_USE_W4A8_FP8_WMMA, ...).

The published headline (298 dec / 1887 total tok/s) is the STOCK path; the `bench`
profile defaults USE_W4A8=0 to reproduce it. Set USE_W4A8=1 to bench the kernel.

Env knobs: MODEL, TP, N_PROMPTS, OUT_TOKENS, GPU_MEM_UTIL, MAX_MODEL_LEN,
MAX_NUM_BATCHED (per-forward batch; decoupled from MAX_MODEL_LEN so a production
context length doesn't force a single giant prefill forward at init).
Traceability: set BENCH_RESULTS_JSONL=<path> to append one JSON record per run
(throughput + full config + BENCH_GIT_SHA + timestamp) so a published number maps
back to a commit. The `bench` profile sets both for you.
"""
import json
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
    # Decouple the per-forward token batch from the context length. max_model_len
    # only sizes KV cache; max_num_batched_tokens drives the prefill forward (and
    # the engine-init memory-profiling dummy run) -- tying them meant a production
    # context like 60000 demanded a single 60000-token forward and OOM'd init on
    # 16 GB / TP=2. Default to a chunked-prefill batch so a large max_model_len
    # validates the real KV config without that spike. Set =max_len to recover the
    # old coupled behaviour.
    max_batched = int(os.environ.get("MAX_NUM_BATCHED", "4096"))

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
        max_num_batched_tokens=max_batched,
        enable_chunked_prefill=True,
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
    out_tok_s = gen_toks / dt
    total_tok_s = total / dt
    print(f"\nprompts={len(outs)}  wall={dt:.2f}s")
    print(f"prefill_tok={prompt_toks}  decode_tok={gen_toks}  total_tok={total}")
    print(f"out_tok_s={out_tok_s:.1f}   (decode throughput)")
    print(f"total_tok_s={total_tok_s:.1f}   (prefill+decode throughput)")

    # Traceability: append one self-describing record so a published number maps
    # back to a commit + the exact config that produced it. Opt-in via env.
    results_path = os.environ.get("BENCH_RESULTS_JSONL")
    if results_path:
        record = {
            "git_sha": os.environ.get("BENCH_GIT_SHA", "unknown"),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": model,
            "tp": tp,
            "w4a8": os.environ.get("VLLM_ROCM_USE_W4A8_FP8_WMMA", "?"),
            "cu_weights": os.environ.get("VLLM_TP_CU_WEIGHTS", "even"),
            "n_prompts": n_prompts,
            "out_tokens": out_tokens,
            "max_model_len": max_len,
            "max_num_batched_tokens": max_batched,
            "gpu_mem_util": gpu_mem,
            "wall_s": round(dt, 3),
            "prefill_tok": prompt_toks,
            "decode_tok": gen_toks,
            "decode_tok_s": round(out_tok_s, 1),
            "total_tok_s": round(total_tok_s, 1),
        }
        os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"\nappended result -> {results_path}")


if __name__ == "__main__":   # vLLM v1 spawns the engine core — guard is mandatory
    main()
