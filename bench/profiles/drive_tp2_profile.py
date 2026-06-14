"""Drive a decode workload around vLLM's torch profiler for the TP=2 FP8 server.

Warmup -> POST /start_profile -> 16 concurrent completions (one max-num-seqs wave,
short prompt + N decode tokens) -> POST /stop_profile (flushes one kineto trace
per rank to VLLM_TORCH_PROFILER_DIR=/profiles -> host bench/profiles/tp2-torch/).
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "http://localhost:8000"
N_SEQ = 16
OUT_TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 128


def post(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        body = r.read().decode()
        return r.status, body


def one_completion(i):
    payload = {
        "model": "model",
        "prompt": f"Count slowly and explain each step as you go. Start now (worker {i}):",
        "max_tokens": OUT_TOKENS,
        "temperature": 0.0,
        "ignore_eos": True,  # force exactly OUT_TOKENS decode steps
    }
    t0 = time.time()
    st, body = post("/v1/completions", payload)
    d = json.loads(body)
    n = d["usage"]["completion_tokens"] if st == 200 else -1
    return i, st, n, time.time() - t0


def main():
    print(f"workload: {N_SEQ} seqs x {OUT_TOKENS} decode tokens")
    # 1) Warmup (excluded from the trace) so lazy init / first-call JIT is paid.
    print("warmup...", post("/v1/completions", {
        "model": "model", "prompt": "Hello", "max_tokens": 8, "temperature": 0})[0])
    # 2) Start profiler on all ranks.
    print("start_profile:", post("/start_profile")[0])
    # 3) Fire one concurrent wave.
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_SEQ) as ex:
        results = list(ex.map(one_completion, range(N_SEQ)))
    wall = time.time() - t0
    # 4) Stop profiler -> flushes traces (can take a few s to serialize).
    print("stop_profile:", post("/stop_profile")[0])
    toks = sum(r[2] for r in results if r[2] > 0)
    print(f"done: {toks} tokens in {wall:.2f}s = {toks/wall:.1f} tok/s aggregate")
    bad = [r for r in results if r[1] != 200]
    if bad:
        print("FAILURES:", bad[:3])


if __name__ == "__main__":
    main()
