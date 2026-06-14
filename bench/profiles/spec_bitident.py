"""Capture greedy (temp=0) completions for a fixed prompt set, to a JSON.

Run twice against the same :8000 endpoint with different server configs and diff:
  python spec_bitident.py all_spec.json     # 'all'-mode + ngram spec server
  python spec_bitident.py baseline.json     # plain FP8, no spec, no 'all'
  python spec_bitident.py --diff all_spec.json baseline.json

Greedy text equality at temp=0 (same tokenizer) is the practical bit-identity
gate: spec decode must not change the chosen tokens vs non-spec greedy.
"""
import json
import sys
import urllib.request

BASE = "http://localhost:8000"
PROMPTS = [
    "Explain in three sentences why the sky is blue.",
    "List the first 10 prime numbers, comma separated.",
    "Write a haiku about gradient descent.",
    "What is 137 times 19? Show your reasoning briefly.",
    "Define tensor parallelism in one paragraph.",
    "Continue the sequence and explain: 2, 6, 12, 20, 30, ...",
]
MAX_TOKENS = 160


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def capture(out_path):
    rows = []
    for i, p in enumerate(PROMPTS):
        d = post("/v1/completions", {
            "model": "model", "prompt": p, "max_tokens": MAX_TOKENS,
            "temperature": 0.0, "seed": 0, "ignore_eos": True})
        txt = d["choices"][0]["text"]
        rows.append({"prompt": p, "text": txt,
                     "n_tok": d["usage"]["completion_tokens"]})
        print(f"[{i}] {p[:48]!r}\n     -> {txt[:120]!r}\n")
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"wrote {out_path} ({len(rows)} prompts)")


def diff(a_path, b_path):
    a = json.load(open(a_path)); b = json.load(open(b_path))
    n_match = 0
    for i, (x, y) in enumerate(zip(a, b)):
        same = x["text"] == y["text"]
        n_match += same
        mark = "MATCH" if same else "DIFFER"
        print(f"[{i}] {mark}  {x['prompt'][:46]!r}")
        if not same:
            # show first divergence
            for j, (ca, cb) in enumerate(zip(x["text"], y["text"])):
                if ca != cb:
                    print(f"      first diff @char {j}: "
                          f"{x['text'][j:j+40]!r} vs {y['text'][j:j+40]!r}")
                    break
            else:
                print(f"      prefix-equal, lengths {len(x['text'])} vs {len(y['text'])}")
    print(f"\n{n_match}/{len(a)} prompts bit-identical (greedy text)")


if __name__ == "__main__":
    if sys.argv[1] == "--diff":
        diff(sys.argv[2], sys.argv[3])
    else:
        capture(sys.argv[1])
