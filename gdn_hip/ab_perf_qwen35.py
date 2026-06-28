"""A/B: GDN WMMA chunked-prefill vs the scalar recurrent prefill on Qwen3.5-4B (GDN hybrid).

Both runs use VLLM_GDN_HIP=1 (native HIP GDN path); the differentiator is
VLLM_GDN_HIP_RECURRENT_ONLY (unset -> WMMA prefill default; =1 -> scalar recurrent prefill). The
route is fixed at engine init, so run ONE engine per invocation and dump (a) greedy token ids for a
coherence check and (b) a long-prompt prefill timing. Invoke twice with different env + GDN_AB_OUT
and diff the two json dumps with ab_compare.py.

The WMMA win is a PREFILL effect, so the perf vehicle is a long prompt with max_tokens=1 (prefill
dominates, ~one decode step). enforce_eager isolates the GDN prefill kernel (no cudagraph noise).

Run inside vllm22-w4a8:gdnwmma under a 1-card lease. __main__ guard for TP-worker re-import safety.
"""
import json
import os
import time

from vllm import LLM, SamplingParams

MODEL = os.environ.get("GDN_AB_MODEL", "Qwen/Qwen3.5-4B")
OUT = os.environ.get("GDN_AB_OUT", "/tmp/gdn_ab.json")
PREFILL_LEN_REPS = int(os.environ.get("GDN_AB_REPS", "230"))   # ~230*12 -> ~2.8k-token prompt (< 4096)
N_ITERS = int(os.environ.get("GDN_AB_ITERS", "10"))
N_WARMUP = int(os.environ.get("GDN_AB_WARMUP", "3"))

COHERENCE_PROMPTS = [
    "The capital of France is",
    "Explain in one sentence why the sky appears blue.",
    "Write a short haiku about the ocean.",
    "List three prime numbers greater than ten:",
    "Once upon a time, in a distant galaxy,",
    "The derivative of x squared with respect to x is",
]
# A long prompt so prefill dominates (the regime where WMMA chunked prefill beats the recurrence).
LONG_PROMPT = ("In a comprehensive technical report, the engineer documented every subsystem. "
               * PREFILL_LEN_REPS).strip() + " Summarize the above in one word:"


def main() -> None:
    gdn = os.environ.get("VLLM_GDN_HIP", "0")
    rec_only = os.environ.get("VLLM_GDN_HIP_RECURRENT_ONLY", "0")
    path = "recurrent" if rec_only == "1" else "wmma"
    print(f"=== GDN A/B: VLLM_GDN_HIP={gdn} RECURRENT_ONLY={rec_only} -> prefill={path} "
          f"model={MODEL} ===", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # (a) coherence: greedy ids on the short prompts
    sp_co = SamplingParams(temperature=0.0, max_tokens=48)
    co = []
    for o in llm.generate(COHERENCE_PROMPTS, sp_co):
        c = o.outputs[0]
        co.append({"prompt": o.prompt, "token_ids": list(c.token_ids), "text": c.text})

    # (b) prefill timing: long prompt, 1 output token (prefill-dominated)
    sp_pf = SamplingParams(temperature=0.0, max_tokens=1)
    n_prompt_tokens = len(llm.get_tokenizer().encode(LONG_PROMPT))
    for _ in range(N_WARMUP):
        llm.generate([LONG_PROMPT], sp_pf, use_tqdm=False)
    times = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        llm.generate([LONG_PROMPT], sp_pf, use_tqdm=False)
        times.append(time.perf_counter() - t0)
    times.sort()
    mean = sum(times) / len(times)
    median = times[len(times) // 2]
    print(f"prefill[{path}] prompt_tokens={n_prompt_tokens} "
          f"mean={mean*1e3:.2f}ms median={median*1e3:.2f}ms min={times[0]*1e3:.2f}ms", flush=True)

    with open(OUT, "w") as f:
        json.dump({
            "gdn_hip": gdn, "recurrent_only": rec_only, "prefill_path": path, "model": MODEL,
            "prompt_tokens": n_prompt_tokens, "iters": N_ITERS,
            "prefill_mean_ms": mean * 1e3, "prefill_median_ms": median * 1e3,
            "prefill_min_ms": times[0] * 1e3, "coherence": co,
        }, f, indent=2)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
