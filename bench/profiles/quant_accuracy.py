"""bf16-vs-FP8 ACCURACY comparison: do both reach the correct answer?

Bit-identity is the wrong gate for a lossy quant (FP8 changes logits -> greedy
diverges). What matters: does FP8 preserve answer correctness vs bf16. Asks
verifiable questions via chat completions (greedy), extracts the final answer
(content, post-</think>), checks the expected string is present.

  python quant_accuracy.py fp8.json     # against the running server
  python quant_accuracy.py --diff bf16.json fp8.json
"""
import json
import sys
import urllib.request

BASE = "http://localhost:8000"
# (prompt, list of acceptable answer substrings — case-insensitive)
QA = [
    ("What is 137 times 19? End with 'Answer: <number>'.", ["2603"]),
    ("What is the 7th prime number? End with 'Answer: <number>'.", ["17"]),
    ("List the first 8 prime numbers, comma separated.",
     ["2, 3, 5, 7, 11, 13, 17, 19", "2,3,5,7,11,13,17,19"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is the next number in the sequence 2, 6, 12, 20, 30? "
     "End with 'Answer: <number>'.", ["42"]),
    ("What is 15% of 240? End with 'Answer: <number>'.", ["36"]),
    ("What is the square root of 144? End with 'Answer: <number>'.", ["12"]),
    ("Is 91 a prime number? Answer yes or no and give its factors if any.",
     ["not", "no", "7", "13"]),
]
MAX_TOKENS = 2048


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def capture(out_path):
    rows = []
    for i, (q, expected) in enumerate(QA):
        d = post("/v1/chat/completions", {
            "model": "model", "messages": [{"role": "user", "content": q}],
            "max_tokens": MAX_TOKENS, "temperature": 0.0, "seed": 0})
        m = d["choices"][0]["message"]
        content = (m.get("content") or "").strip()
        fin = d["choices"][0]["finish_reason"]
        ok = any(e.lower() in content.lower() for e in expected)
        rows.append({"q": q, "content": content, "expected": expected,
                     "ok": ok, "finish": fin,
                     "n_tok": d["usage"]["completion_tokens"]})
        print(f"[{i}] {'OK ' if ok else 'MISS'} ({fin}, {rows[-1]['n_tok']}t) {q[:44]!r}")
        print(f"     -> {content[:140]!r}\n")
    n_ok = sum(r["ok"] for r in rows)
    print(f"{n_ok}/{len(rows)} correct")
    json.dump(rows, open(out_path, "w"), indent=2)


def diff(a_path, b_path):
    a = json.load(open(a_path)); b = json.load(open(b_path))
    print(f"{'Q':<46} {'bf16':>6} {'fp8':>6}")
    na = nb = 0
    for x, y in zip(a, b):
        na += x["ok"]; nb += y["ok"]
        print(f"{x['q'][:46]:<46} {'OK' if x['ok'] else 'MISS':>6} "
              f"{'OK' if y['ok'] else 'MISS':>6}")
    print(f"\nbf16: {na}/{len(a)} correct   fp8: {nb}/{len(b)} correct")


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        capture(sys.argv[1])
