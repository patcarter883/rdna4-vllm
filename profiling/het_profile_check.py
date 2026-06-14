"""Offline torch-profiler driver for the het-TP COMM-bubble A/B.

Runs the model at TP=2 under vLLM's torch profiler around a decode-heavy workload, so the
per-rank kineto traces expose the TP all-reduce / collective time. Run once with
VLLM_TP_CU_WEIGHTS unset (even baseline) and once ="64,56" (het). Under EVEN split the bigger
(64-CU) card finishes its shard first and spin-waits at the all-reduce barrier -> inflated
collective time on one rank; HET=64,56 balances the work so the imbalance shrinks.

Uses the OFFLINE LLM path (not the server /start_profile route, which this fork's `vllm serve`
doesn't expose) and the SAME LLM() config as patches/het_e2e_check.py, so the Triton cache from
the equivalence runs is warm.

Env: HET_MODEL, HET_TAG, HET_OUT (abs dir for traces, default /profiles), HET_TP (2),
HET_MAXLEN (2048), HET_GPUUTIL (0.85), HET_NSEQ (16), HET_DECODE_TOKS (128), HET_LIMIT_MM.
"""
import os
import time

from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig

PROMPT = "Count slowly and explain each step as you go. Start now"


def main():
    # __main__ guard required: TP workers re-import this module under spawn.
    model = os.environ["HET_MODEL"]
    tag = os.environ.get("HET_TAG", "out")
    out = os.environ.get("HET_OUT", "/profiles")  # absolute; mounted to host
    tp = int(os.environ.get("HET_TP", "2"))
    maxlen = int(os.environ.get("HET_MAXLEN", "2048"))
    util = float(os.environ.get("HET_GPUUTIL", "0.85"))
    nseq = int(os.environ.get("HET_NSEQ", "16"))
    dtoks = int(os.environ.get("HET_DECODE_TOKS", "128"))

    pc = ProfilerConfig(profiler="torch", torch_profiler_dir=out,
                        torch_profiler_with_stack=False, torch_profiler_use_gzip=True)
    kwargs = dict(
        model=model, tensor_parallel_size=tp, dtype="float16",
        max_model_len=maxlen, gpu_memory_utilization=util,
        enforce_eager=True, trust_remote_code=True, profiler_config=pc,
    )
    if os.environ.get("HET_LIMIT_MM", "0") == "1":
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}

    llm = LLM(**kwargs)

    # Warmup (excluded from the trace): pays first-call JIT for these shapes.
    llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)

    # Decode-heavy wave: nseq concurrent seqs, ignore_eos -> exactly dtoks decode steps each.
    prompts = [f"{PROMPT} (worker {i}):" for i in range(nseq)]
    sp = SamplingParams(temperature=0.0, max_tokens=dtoks, ignore_eos=True)

    llm.start_profile(profile_prefix=tag)
    t0 = time.time()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.time() - t0
    llm.stop_profile()
    time.sleep(8)  # let the per-rank kineto traces serialize/flush to HET_OUT

    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[{tag}] {toks} tokens in {wall:.2f}s = {toks/wall:.1f} tok/s aggregate "
          f"({nseq} seqs x {dtoks} decode)")
    print(f"[{tag}] traces dir: {out}")


if __name__ == "__main__":
    main()
