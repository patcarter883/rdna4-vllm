"""Measure decode tok/s (single-stream, decode-bound) against a served ZAYA.

    python decode_toks.py http://localhost:8001 [max_tokens]

Sends one greedy completion with a fixed prompt and reports output tok/s from
the server-side usage counts + client wall time. Single concurrency keeps it
decode-bound (the regime the CCA kernel lives in).
"""
import sys
import time
import json
import urllib.request

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512

prompt = (
    "You are a careful reasoner. Solve step by step, then give the final answer.\n"
    "Question: A train leaves city A at 9:00am going 60 km/h. Another leaves "
    "city B, 300 km away, at 10:00am going 90 km/h toward A. At what time do "
    "they meet? Show all the arithmetic.\n"
)


def call(n_tokens):
    body = json.dumps({
        "model": "model",
        "prompt": prompt,
        "max_tokens": n_tokens,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        base + "/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    dt = time.time() - t0
    usage = resp["usage"]
    return dt, usage, resp["choices"][0]["text"]


# warm the server / caches with a short call
call(16)

runs = []
for _ in range(3):
    dt, usage, text = call(max_tokens)
    out = usage["completion_tokens"]
    runs.append((out, dt, out / dt))
    print(f"  out={out:4d} tok  wall={dt:6.2f}s  -> {out/dt:6.2f} tok/s")

best = max(r[2] for r in runs)
print(f"best decode: {best:.2f} tok/s  (max_tokens={max_tokens})")
print("--- sample (first 240 chars) ---")
print(text[:240].replace("\n", " "))
