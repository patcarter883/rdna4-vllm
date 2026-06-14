"""RSA latency benchmark harness.

Drives a fixed set of known-answer problems through the RSA loop against a live
vLLM backend, recording per-query wall-clock latency, token usage, the selected
answer (scored for correctness), and — when Prometheus is reachable — the peak
server-side concurrency and token throughput over the run.

This is the Step-0 baseline tool: capture numbers before a change, capture them
again after, compare. Problems run **sequentially** by default because a single
RSA query already fans out N rollouts that saturate the server; running queries
concurrently would distort the per-query latency we are trying to measure.

Example:
    .venv/bin/python -m bench.run --backend http://localhost:8000/v1 \\
        --rsa-n 16 --rsa-k 4 --rsa-t 2 --repeat 3 --out baseline.json

    # Cheaper smoke run against the built-in problems:
    .venv/bin/python -m bench.run --rsa-n 4 --rsa-k 2 --rsa-t 2 \\
        --rsa-max-tokens 4096
"""

import argparse
import asyncio
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass, field

from rsa import extract
from rsa.config import add_rsa_args, params_from_args
from rsa.core import BackendClient, run_rsa

from bench.metrics import COUNTER_QUERIES, GaugeSampler, PromClient
from bench.problems import Problem, load_problems

logger = logging.getLogger("bench")


def score_answer(final_text: str, expected: str) -> tuple[str | None, bool]:
    """Extract the boxed answer and compare it to *expected* (normalized)."""
    extracted = extract.extract_boxed(final_text)
    if extracted is None:
        return None, False
    return extracted, extract.normalize_answer(extracted) == extract.normalize_answer(
        expected
    )


@dataclass
class ProblemResult:
    id: str
    correct: bool
    extracted: str | None
    expected: str
    latency_s: float
    selection_method: str
    n_requests: int
    prompt_tokens: int
    completion_tokens: int
    peak_running: float | None = None
    peak_waiting: float | None = None


@dataclass
class RunReport:
    config: dict
    results: list[ProblemResult] = field(default_factory=list)
    accuracy: float = 0.0
    latency: dict[str, float] = field(default_factory=dict)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    wall_clock_s: float = 0.0
    server: dict = field(default_factory=dict)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (no numpy dependency)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {}
    return {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p90": _percentile(latencies, 90),
        "min": min(latencies),
        "max": max(latencies),
        "n": len(latencies),
    }


async def _run_one(
    client: BackendClient,
    params,
    model: str,
    problem: Problem,
    prom: PromClient | None,
    sample_interval: float,
) -> ProblemResult:
    messages = [{"role": "user", "content": problem.prompt()}]
    sampler = GaugeSampler(prom, sample_interval) if prom else None
    if sampler:
        sampler.start()
    start = time.monotonic()
    result = await run_rsa(client, params, messages, model)
    latency = time.monotonic() - start
    gauges = await sampler.stop() if sampler else {}

    extracted, correct = score_answer(result.final_text, problem.answer)
    logger.info(
        "%-12s %s  %.1fs  ans=%s expected=%s  (%d reqs, %d+%d tok)",
        problem.id,
        "OK " if correct else "XX ",
        latency,
        extracted,
        problem.answer,
        result.usage.n_requests,
        result.usage.prompt_tokens,
        result.usage.completion_tokens,
    )
    return ProblemResult(
        id=problem.id,
        correct=correct,
        extracted=extracted,
        expected=problem.answer,
        latency_s=latency,
        selection_method=result.selection_method,
        n_requests=result.usage.n_requests,
        prompt_tokens=result.usage.prompt_tokens,
        completion_tokens=result.usage.completion_tokens,
        peak_running=gauges.get("running", {}).get("max"),
        peak_waiting=gauges.get("waiting", {}).get("max"),
    )


async def run_benchmark(args: argparse.Namespace) -> RunReport:
    params = params_from_args(args)
    problems = load_problems(args.questions)
    client = BackendClient(args.backend, api_key=args.api_key, tokenizer=args.tokenizer)
    prom = PromClient(args.prometheus) if args.prometheus else None

    try:
        model = args.model or await client.default_model()
        config = {
            "backend": args.backend,
            "model": model,
            "rsa": params.model_dump(),
            "repeat": args.repeat,
            "questions": args.questions or "builtin",
            "num_problems": len(problems),
        }

        # Warm up (compile/cache) without scoring, so the first measured query
        # isn't paying one-time costs.
        for _ in range(args.warmup):
            warm = [{"role": "user", "content": problems[0].prompt()}]
            await run_rsa(client, params, warm, model)
            logger.info("warmup complete")

        before = await prom.snapshot(COUNTER_QUERIES) if prom else {}
        wall_start = time.monotonic()
        results: list[ProblemResult] = []
        for _ in range(args.repeat):
            for problem in problems:
                results.append(
                    await _run_one(
                        client, params, model, problem, prom, args.sample_interval
                    )
                )
        wall_clock = time.monotonic() - wall_start
        after = await prom.snapshot(COUNTER_QUERIES) if prom else {}

        latencies = [r.latency_s for r in results]
        report = RunReport(
            config=config,
            results=results,
            accuracy=sum(r.correct for r in results) / len(results),
            latency=_latency_stats(latencies),
            total_prompt_tokens=sum(r.prompt_tokens for r in results),
            total_completion_tokens=sum(r.completion_tokens for r in results),
            wall_clock_s=wall_clock,
        )
        report.server = _server_summary(before, after, wall_clock, results)
        return report
    finally:
        await client.close()
        if prom:
            await prom.close()


def _server_summary(
    before: dict, after: dict, wall_clock: float, results: list[ProblemResult]
) -> dict:
    summary: dict = {}
    for name in COUNTER_QUERIES:
        if name in before and name in after:
            delta = after[name] - before[name]
            summary[f"{name}_delta"] = delta
            if name == "generation_tokens" and wall_clock > 0:
                summary["server_generation_tokens_per_s"] = delta / wall_clock
    peaks = [r.peak_running for r in results if r.peak_running is not None]
    if peaks:
        summary["peak_running_seqs"] = max(peaks)
    waits = [r.peak_waiting for r in results if r.peak_waiting is not None]
    if waits:
        summary["peak_waiting_seqs"] = max(waits)
    return summary


def _print_summary(report: RunReport) -> None:
    lat = report.latency
    print("\n" + "=" * 72)
    print(f"problems x repeat : {lat.get('n', 0)} runs")
    print(f"accuracy          : {report.accuracy:.1%}")
    if lat:
        print(
            f"latency (s)       : median {lat['median']:.1f}  "
            f"mean {lat['mean']:.1f}  p90 {lat['p90']:.1f}  "
            f"min {lat['min']:.1f}  max {lat['max']:.1f}"
        )
    print(
        f"tokens            : {report.total_prompt_tokens} prompt + "
        f"{report.total_completion_tokens} completion"
    )
    if report.server:
        s = report.server
        if "peak_running_seqs" in s:
            print(
                f"peak server seqs  : running {s['peak_running_seqs']:.0f}  "
                f"waiting {s.get('peak_waiting_seqs', 0):.0f}"
            )
        if "server_generation_tokens_per_s" in s:
            print(
                f"server decode     : "
                f"{s['server_generation_tokens_per_s']:.0f} gen tok/s"
            )
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="http://localhost:8000/v1", help="vLLM base URL"
    )
    parser.add_argument("--model", default=None, help="model id (default: first)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--tokenizer", default=None, help="HF tokenizer for tails")
    parser.add_argument(
        "--questions",
        default="builtin",
        help="'builtin' or a JSONL path ({id, question, answer} per line)",
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="runs per problem (median is robust)"
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="unscored warmup runs before measuring"
    )
    parser.add_argument(
        "--prometheus",
        default=None,
        help="Prometheus base URL (e.g. http://localhost:9090) for server metrics",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="gauge poll interval (s) when --prometheus is set",
    )
    parser.add_argument("--out", default=None, help="write JSON report to this path")
    add_rsa_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    report = asyncio.run(run_benchmark(args))
    _print_summary(report)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {**asdict(report)},
                f,
                indent=2,
            )
        logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
