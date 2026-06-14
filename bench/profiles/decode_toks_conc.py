"""Aggregate decode throughput at fixed concurrency against a served ZAYA.

    python decode_toks_conc.py http://localhost:8001 [conc] [max_tokens]

Fires `conc` greedy completions concurrently (distinct prompts to avoid prefix
sharing) and reports aggregate output tok/s = sum(completion_tokens)/wall. This
is the batch-decode regime where per-step GPU compute (e.g. the CCA kernel)
matters, unlike single-stream which is HBM-bandwidth-bound on weight loads.
"""
import sys
import time
import json
import threading
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
conc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 256

base_prompt = (
    "You are a careful reasoner. Solve step by step, then give the final "
    "answer. Question {i}: Compute the value of the following and explain each "
    "step in detail: ({i} * 37 + 19) then divide by 3, then describe a short "
    "story about the number you get. Be thorough and verbose.\n"
)


def call(i, out_list):
    body = json.dumps({
        "model": "model",
        "prompt": base_prompt.format(i=i),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    out_list[i] = resp["usage"]["completion_tokens"]


def run():
    outs = [0] * conc
    threads = [threading.Thread(target=call, args=(i, outs)) for i in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0
    total = sum(outs)
    return total, dt, total / dt


# warmup
run()
best = 0.0
for _ in range(3):
    total, dt, tps = run()
    print(f"  conc={conc} total_out={total:5d} tok  wall={dt:6.2f}s  -> {tps:7.2f} tok/s")
    best = max(best, tps)
print(f"best aggregate: {best:.2f} tok/s  (conc={conc}, max_tokens={max_tokens})")
