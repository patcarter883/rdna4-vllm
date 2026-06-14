# RSA serving proxy for ZAYA1 on vLLM

An OpenAI-compatible proxy that applies **Recursive Self-Aggregation (RSA)**
test-time compute to chat completions served by a vLLM backend.

- Generalized RSA: [Venkatraman et al., arXiv:2509.26626](https://arxiv.org/abs/2509.26626)
  — keep a population of N candidate solutions; for T rounds, aggregate random
  subsets of K candidates into improved candidates.
- Markovian RSA: [ZAYA1-8B technical report, arXiv:2605.05365](https://arxiv.org/abs/2605.05365)
  — the same loop, but each candidate is truncated to its final τ tokens (the
  "tail") before aggregation. This is the configuration ZAYA1 used to reach
  91.9% on AIME'25.

Setting `tail_tokens=0` gives full-trace generalized RSA; `tail_tokens=4096`
(the default) reproduces ZAYA1's Markovian RSA.

Nothing in the `vllm/` package is touched — this directory is a standalone
client-side orchestrator.

## How it works

```
client ──► rsa proxy (:8100) ──► vLLM server (:8000)
              │  round 0: N independent rollouts of the request
              │  rounds 1..T-1: N aggregation prompts, each containing
              │      K randomly sampled candidate tails (τ tokens each)
              │  selection: majority vote over \boxed{} answers, or one
              │      final-aggregation call for general (non-math) queries
              └─ returns ONE normal chat completion, with usage summed
                 across every backend call (true cost)
```

Requests are **passed through unchanged** when they contain `tools`, ask for
`n > 1`, or set `"rsa": false` in the body.

## Setup

From the repo root (the proxy needs only `openai`, `fastapi`, `uvicorn`,
`httpx`, `pydantic` — not the vLLM package):

```bash
uv venv --python 3.12
uv pip install -p .venv/bin/python openai fastapi uvicorn httpx pydantic pytest
# Optional but recommended: local token-exact tails + math verifier
uv pip install -p .venv/bin/python transformers sympy
```

## Running

1. Start the backend (this repo's `docker-compose.yml` already does this —
   ZAYA1-8B served as model name `model` on port 8000):

   ```bash
   docker compose up -d vllm
   ```

2. Start the proxy:

   ```bash
   # ZAYA1-report configuration (N=16, K=4, T=2, tau=4096, beta=40000):
   .venv/bin/python -m rsa.server --backend http://localhost:8000/v1 --port 8100

   # Cheaper smoke-test configuration:
   .venv/bin/python -m rsa.server --backend http://localhost:8000/v1 --port 8100 \
     --rsa-n 4 --rsa-k 2 --rsa-t 2 --rsa-tail-tokens 2048 --rsa-max-tokens 8192
   ```

3. Call it like any OpenAI endpoint:

   ```bash
   curl -s http://localhost:8100/v1/chat/completions \
     -H 'Content-Type: application/json' -d '{
       "model": "model",
       "messages": [{"role": "user",
         "content": "What is 17*23? Put your final answer in \\boxed{}."}]
     }' | jq '.choices[0].message.content, .usage, .rsa'
   ```

   The response is a standard chat completion plus a non-standard `rsa`
   object reporting rounds, population, selection method, vote tally, and
   total backend requests.

### Per-request overrides

Any `RSAParams` field can be overridden per request via the `rsa` body key
(`extra_body` in the openai python client):

```python
client.chat.completions.create(
    model="model",
    messages=[...],
    extra_body={"rsa": {"n": 8, "t": 3, "tail_tokens": 0}},  # full-trace RSA
)
# or bypass RSA entirely:
client.chat.completions.create(..., extra_body={"rsa": False})
```

Request-level `temperature` and `max_tokens` override the rollout defaults.

### Parameters

| flag | default | meaning |
|---|---|---|
| `--rsa-n` | 16 | population size N (controls the quality ceiling — prioritize this) |
| `--rsa-k` | 4 | aggregation set size K (gains diminish beyond ~4) |
| `--rsa-t` | 2 | total rounds T (ZAYA1 used 2; the RSA paper sees monotonic gains to 5–10) |
| `--rsa-tail-tokens` | 4096 | τ — candidate tail carried into aggregation prompts; 0 = full trace |
| `--rsa-max-tokens` | 40000 | β — per-rollout completion budget |
| `--rsa-temperature` | 0.8 | rollout sampling temperature |
| `--rsa-selection` | auto | `auto` / `majority` / `final_agg` / `sample` |
| `--rsa-verifier` | off | `off` / `math` / `code` / `auto` — see below |
| `--rsa-max-concurrency` | 16 | concurrent backend requests |
| `--tokenizer` | auto | HF tokenizer name/path for local tails (resolved from the backend's model `root` by default) |

### Tail truncation

Tails are token-exact and never start mid-thought:

1. **Local HF tokenizer** (preferred): loaded lazily via `AutoTokenizer`,
   auto-resolved from the backend's reported HF repo (ZAYA1 ships a standard
   `GemmaTokenizerFast`, no remote code). Slicing happens in-process.
2. **Backend `/tokenize` + `/detokenize`** if `transformers` isn't installed.
3. **Character approximation** (~4 chars/token) as a last resort.

After slicing, the cut is advanced to the next paragraph (or line) boundary
within the leading 10% of the tail — a "look-back cut" — so each aggregation
prompt receives a complete, coherent block of reasoning instead of a severed
sentence. Texts already shorter than τ characters skip tokenization entirely.

### Verifier filter

`--rsa-verifier` (or per-request `"rsa": {"verifier": "math"}`) filters
**provably broken** candidates out of the aggregation sampling pool and the
final vote. It never picks winners — RSA depends on population diversity
(N ≫ K), so verification only excludes, and falls back to the full
population whenever filtering would leave fewer than K candidates.

- `math`: the candidate's `\boxed{}` answer must parse as a numeric/symbolic
  expression (LaTeX `\frac`/`\sqrt` de-sugared, then SymPy). Heuristic —
  exotic-but-valid notation can fail, hence the pool fallback.
- `code`: the candidate's last ```` ```python ```` block must run cleanly in
  a subprocess (isolated interpreter, CPU/memory/process rlimits, 5 s
  timeout). **This executes model-generated code on the proxy host** — the
  sandbox is best-effort, not a security boundary. Only enable it where
  that's acceptable.
- `auto`: both checks; a candidate fails if either applicable check fails.

### Context-length sizing

Aggregation prompts hold K tails of τ tokens plus the query and template:
with the defaults that's ~16K prompt tokens, and rollouts may generate up to
β=40K — so the backend needs roughly `--max-model-len ≥ τ·K + β + slack`
(~60K for the full report configuration, which is what the compose file sets).
If a long aggregation prompt makes `max_tokens` overflow the remaining
context, the proxy retries once letting the server clamp it.

## Debugging: single-question CLI

Bypasses the proxy and prints per-round stats, the vote tally, and usage:

```bash
.venv/bin/python -m rsa.cli --backend http://localhost:8000/v1 \
  -q 'What is 17*23? Put your final answer in \boxed{}.' \
  --rsa-n 4 --rsa-k 2 --rsa-t 2 --rsa-max-tokens 8192
```

## Tests

```bash
.venv/bin/python -m pytest rsa/test_rsa.py -q   # mocked backend, no GPU
```

## Comparison harness and results

`rsa/compare.py` benchmarks RSA against two baselines from the *same* GPU
time: the round-0 population is N independent samples (single-sample pass@1
and self-consistency/majority-vote baselines), and the aggregated answer is
RSA.

```bash
.venv/bin/python -m rsa.compare rsa/aime_mini.json \
  --rsa-n 8 --rsa-k 3 --rsa-t 2 --rsa-tail-tokens 1536 \
  --rsa-max-tokens 5000 --rsa-verifier math
```

Results on 5 AIME problems (ZAYA1-8B bf16, single RX 7900, dedicated;
~165 tok/s aggregate at batch 8; full data in `results/`):

| method | result |
|---|---|
| single sample (pass@1) | 27/40 (67.5%) |
| self-consistency (vote over 8) | 5/5 |
| RSA (N=8, K=3, T=2, τ=1536, β=5000) | 5/5 |
| per-trace accuracy after one aggregation round | 33/40 (82.5%) |

Takeaways: RSA never returned a wrong answer and the aggregation mechanism
visibly works — on the hardest problem only 1/8 round-0 traces produced an
answer, yet 6/8 post-aggregation traces converged on the correct one from a
single good 1,536-token tail. On this difficulty band, however,
self-consistency (`"rsa": {"t": 1}`) matched RSA's final answers at half
the token cost; the aggregation round earns its budget on problems where
round 0 yields zero (or majority-wrong) extractable answers. Cost per
problem at these settings: ~75K completion tokens, ~9 minutes.

## AIME-style sanity check (manual)

Run a handful of known-answer problems through the CLI and compare boxed
answers, e.g. AIME 2024 I problems 1–5. Expectation: RSA accuracy at
(N=8, T=2) ≥ single-sample accuracy at the same temperature. RSA helps most
on problems where independent samples disagree (a large pass@N − pass@1 gap
predicts aggregability, per the RSA paper).

## Out of scope (v1)

- RSA over tool-calling requests (passed through instead)
- True token-by-token streaming (`stream: true` returns the final answer
  pseudo-streamed after aggregation, with SSE keep-alives during the wait)
- Automated benchmark harness
- Verifying answer *correctness* (the verifier checks well-formedness and
  executability, not ground truth); round-level trace ranking is omitted by
  design — see "Verifier filter"
