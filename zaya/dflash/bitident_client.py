#!/usr/bin/env python3
"""Greedy bit-identity client for the ZAYA CCA spec-decode bit-lossless gate (M5, task A).

Host-side (no GPU): hits a running vLLM OpenAI server with a fixed prompt set at
temperature=0 and dumps, per prompt, the chosen token sequence (via completions
`logprobs.tokens`) + the full text. Two such dumps (num_spec=0 baseline vs an
ngram-spec run) are compared with bitident_diff.py — identical token sequences
prove the CCA 'all'-mode partial-acceptance rollback is bit-lossless.

Also scrapes /metrics for spec-decode acceptance counters so we can confirm the
ngram run actually exercised PARTIAL acceptance (not just the already-validated
full-rejection regime).

    python bitident_client.py --url http://localhost:8001 --out /path/run.json --label all-ngram

No third-party deps (urllib only) so it runs straight on the host.
"""
import argparse
import json
import sys
import time
import urllib.request
import urllib.error

# Prompts chosen to span (a) strongly repetitive sequences where ngram prompt-
# lookup fires and ACCEPTS multiple tokens (exercises partial + full acceptance
# rollback), and (b) ordinary prose (mostly full-rejection) for breadth. Greedy
# (temperature=0) so output is deterministic and comparable token-for-token.
PROMPTS = [
    # counting continuation — ngram matches the "N, N+1," structure, high acceptance
    "Continue this sequence of numbers, comma separated, for a long time: "
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,",
    # even numbers — repetitive structure
    "List even numbers separated by commas: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,",
    # a-b alternation — guarantees n-gram repetition acceptance
    "Repeat the pattern: a b a b a b a b a b a b a b a b a b a b a b a b a b a b a b a b",
    # explicit repetition request — exercises repeated-line acceptance
    "Write the sentence \"the cat sat on the mat\" ten times, one per line.\n"
    "the cat sat on the mat\nthe cat sat on the mat\n",
    # ordinary prose — mostly full rejection (the regime M5 already validated)
    "The history of the Roman Empire began with the founding of the city of Rome. "
    "Over the following centuries,",
    # short factual
    "Q: What are the first five prime numbers?\nA:",
]


def complete(url: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {
            "model": "model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 1,  # -> choices[0].logprobs.tokens = chosen-token strings
            "seed": 0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.load(r)
    ch = resp["choices"][0]
    lp = ch.get("logprobs") or {}
    return {
        "text": ch.get("text", ""),
        "tokens": lp.get("tokens", []),
        "finish_reason": ch.get("finish_reason"),
    }


def scrape_metrics(url: str, timeout: float) -> dict:
    """Pull spec-decode acceptance counters from Prometheus /metrics."""
    out = {}
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
            text = r.read().decode()
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if "spec_decode" in line or "speculative" in line:
            try:
                name, val = line.rsplit(" ", 1)
                out[name.strip()] = float(val)
            except ValueError:
                pass
    return out


def wait_healthy(url: str, deadline_s: float) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to wait for /health first")
    args = ap.parse_args()

    if args.wait > 0 and not wait_healthy(args.url, args.wait):
        print(f"server at {args.url} not healthy within {args.wait}s", file=sys.stderr)
        return 2

    metrics_before = scrape_metrics(args.url, args.timeout)
    runs = {}
    for i, p in enumerate(PROMPTS):
        try:
            runs[str(i)] = complete(args.url, p, args.max_tokens, args.timeout)
        except urllib.error.HTTPError as e:
            print(f"prompt {i} HTTP {e.code}: {e.read().decode()[:300]}", file=sys.stderr)
            return 3
        ntok = len(runs[str(i)]["tokens"])
        print(f"[{args.label}] prompt {i}: {ntok} tokens, finish={runs[str(i)]['finish_reason']}")
    metrics_after = scrape_metrics(args.url, args.timeout)

    payload = {
        "label": args.label,
        "url": args.url,
        "max_tokens": args.max_tokens,
        "prompts": PROMPTS,
        "runs": runs,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{args.label}] wrote {args.out}")
    # surface spec acceptance delta if present
    for k in sorted(metrics_after):
        if "accept" in k or "draft" in k or "num_spec" in k:
            b = metrics_before.get(k, 0.0)
            print(f"  {k}: {metrics_after[k] - b:+g} (now {metrics_after[k]:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
