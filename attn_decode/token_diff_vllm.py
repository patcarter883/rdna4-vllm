"""End-to-end token-diff: vLLM triton_attn decode vs the native attn_decode HIP path.

The route is locked at first-forward (VLLM_ATTN_DECODE_HIP read once), so — like gdn_hip's
token_diff — run ONE engine per invocation and dump greedy token ids; invoke twice (=0 then =1) and
diff the json dumps. A dense full-attention model (Qwen3.5-4B) is the vehicle so the decode kernel
fires on EVERY layer. Expect near-identical greedy ids; an early/structural divergence means a wiring
bug (KV layout / block_table / seq_lens / sliding-window / output write), not sampling noise.

Run inside vllm22-w4a8:attndec under a 1-card lease. __main__ guard for TP-worker re-import safety.
"""
import json
import os

from vllm import LLM, SamplingParams

MODEL = os.environ.get("ATTN_DIFF_MODEL", "Qwen/Qwen3.5-4B")
OUT = os.environ.get("ATTN_DIFF_OUT", "/tmp/attn_diff")
# Small token budget forces CHUNKED prefill (later chunks have prefix>0 → the extend path through
# attn_prefill_paged); prefix caching makes a shared-prefix request a cache-hit extend.
MAX_BATCHED = int(os.environ.get("ATTN_DIFF_MAX_BATCHED", "64"))
PREFIX_CACHE = os.environ.get("ATTN_DIFF_PREFIX_CACHE", "1") == "1"
KV_DTYPE = os.environ.get("ATTN_DIFF_KV_DTYPE", "auto")   # "fp8" to exercise the fp8-KV HIP path

_LONG = ("You are a careful assistant. " * 20).strip()  # ~120 tokens -> chunks at budget 64
PROMPTS = [
    "The capital of France is",
    "List three prime numbers greater than ten:",
    "Q: What is 17 plus 26? A:",
    _LONG + " In one sentence, summarize your role:",          # forces chunked prefill (cold+extend)
    _LONG + " Name a primary color:",                          # SHARED prefix with the previous ->
    _LONG + " Name a primary color:",                          #   prefix-cache hit (extend, prefix>0)
]


def main() -> None:
    flag = os.environ.get("VLLM_ATTN_DECODE_HIP", "0")
    print(f"=== token-diff: VLLM_ATTN_DECODE_HIP={flag} model={MODEL} "
          f"(chunked={MAX_BATCHED}, prefix_cache={PREFIX_CACHE}) ===", flush=True)
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
        enforce_eager=True,          # eager so the patched forward runs (no cudagraph capture)
        trust_remote_code=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=MAX_BATCHED,
        enable_prefix_caching=PREFIX_CACHE,
        kv_cache_dtype=KV_DTYPE,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(PROMPTS, sp)
    ids = [list(o.outputs[0].token_ids) for o in outs]
    path = f"{OUT}_{flag}.json"
    json.dump({"flag": flag, "model": MODEL, "ids": ids}, open(path, "w"))
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
