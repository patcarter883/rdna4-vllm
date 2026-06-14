# RSA latency benchmark harness

The **Step-0 measurement tool** for the ZAYA1 performance work — now targeting
the gfx1201 (RX 9070 XT) TheRock/ROCm-7.14 stack (see
`../ZAYA1_GFX1201_PORT.md`; metric names match vLLM 0.22, e.g.
`vllm:kv_cache_usage_perc`). It measures the metric we are optimizing —
**RSA query latency** — plus answer accuracy (the regression gate for
accuracy-affecting changes) and, when Prometheus is reachable, server-side
peak concurrency and decode throughput.

Like `rsa/`, this is a standalone client-side tool. It imports `rsa.core`,
`rsa.config`, and `rsa.extract`, and touches nothing in the `vllm/` package — so
it is safe to develop and run without rebuilding or restarting the shared
inference container.

## What it reports

Per problem: wall-clock latency, selected answer + correctness, token usage,
selection method, and (with `--prometheus`) the peak `running`/`waiting` seqs
reached during the query. Aggregated: accuracy, latency median/mean/p90/min/max,
total tokens, and server decode tokens/s over the run.

The **peak running seqs** is the capacity check: after expert quantization
(`quant/README.md`) + `--max-num-seqs 16`, a round of 16 rollouts should run
as one wave (`running ≈ 16, waiting ≈ 0`) instead of queueing.

## Setup

Same client-only dependencies as the RSA proxy (`openai`, `httpx`, `pydantic`,
`pytest` — not the vLLM package):

```bash
uv venv --python 3.12
uv pip install -p .venv/bin/python openai fastapi uvicorn httpx pydantic pytest sympy
```

(`sympy` is optional at runtime but the verifier can only check plain
numerics without it — and `rsa/test_rsa.py` requires it.)

## Usage

> Running against the backend **drives load** (each query fans out N rollouts),
> so coordinate a window with anyone else testing on the shared server first.

```bash
# Baseline at the ZAYA1-report config, 3 runs/problem, JSON out, server metrics:
.venv/bin/python -m bench.run \
  --backend http://localhost:8000/v1 \
  --prometheus http://localhost:9090 \
  --rsa-n 16 --rsa-k 4 --rsa-t 2 --repeat 3 \
  --out baseline.json

# Cheap smoke run (built-in problems, small budget):
.venv/bin/python -m bench.run --rsa-n 4 --rsa-k 2 --rsa-t 2 --rsa-max-tokens 4096
```

Compare two runs by diffing the `latency`, `accuracy`, and `server` blocks of
their JSON reports (e.g. `baseline.json` vs `int8.json`).

### Problem sets

`--questions builtin` (default) runs six verifiable arithmetic/number-theory
problems — enough to validate the harness and get a quick latency read, but not
competition-grade reasoning. For the real accuracy gate, pass a JSONL file of
known-answer problems (e.g. AIME'24/'25), one object per line:

```jsonl
{"id": "aime24-i-1", "question": "...", "answer": "204"}
{"id": "aime24-i-2", "question": "...", "answer": "025"}
```

Answers are compared after `rsa.extract.normalize_answer`, so `204`, `$204$`,
and `{204}` all match.

### Useful flags

| flag | default | meaning |
|---|---|---|
| `--repeat` | 1 | runs per problem; report uses the median latency |
| `--warmup` | 0 | unscored runs first, to absorb compile/cache one-time costs |
| `--prometheus` | off | Prometheus base URL for server-side concurrency/throughput |
| `--sample-interval` | 1.0 | gauge poll interval (s) while a query runs |
| `--out` | — | write the full JSON report to this path |

All `--rsa-*` flags from the proxy are accepted (`rsa/config.py`).

## Tests

```bash
.venv/bin/python -m pytest bench/test_bench.py -q   # mocked backend, no GPU
```
