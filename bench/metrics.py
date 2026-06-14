"""Best-effort server-side metric collection from Prometheus.

The compose stack already scrapes vLLM's ``/metrics`` into Prometheus
(``monitoring/prometheus/prometheus.yml``). This module reads a few of those
series back through the Prometheus HTTP API so a benchmark run can report
server-side throughput and the *peak concurrency* actually achieved — the
number that tells us whether a round of N rollouts ran as one wave or queued.

Everything here is best-effort: if Prometheus is unreachable the helpers return
empty results and the harness still reports client-side latency.
"""

import asyncio
import logging

import httpx

logger = logging.getLogger("bench.metrics")

# Instant-value gauges sampled during a run.
GAUGE_QUERIES: dict[str, str] = {
    "running": "sum(vllm:num_requests_running)",
    "waiting": "sum(vllm:num_requests_waiting)",
    # renamed from vllm:gpu_cache_usage_perc in vLLM 0.22
    "kv_cache_usage_perc": "avg(vllm:kv_cache_usage_perc)",
}

# Monotonic counters; the driver snapshots these before/after to get a delta.
COUNTER_QUERIES: dict[str, str] = {
    "prompt_tokens": "sum(vllm:prompt_tokens_total)",
    "generation_tokens": "sum(vllm:generation_tokens_total)",
}


class PromClient:
    """Thin async client for the Prometheus instant-query API."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._warned = False

    async def close(self) -> None:
        await self.http.aclose()

    async def query(self, expr: str) -> float | None:
        """Return the scalar value of an instant query, or None on failure."""
        try:
            r = await self.http.get("/api/v1/query", params={"query": expr})
            r.raise_for_status()
            result = r.json()["data"]["result"]
            if not result:
                return None
            return float(result[0]["value"][1])
        except (httpx.HTTPError, KeyError, ValueError, IndexError) as e:
            if not self._warned:
                logger.warning("Prometheus query failed (%s); skipping metrics", e)
                self._warned = True
            return None

    async def snapshot(self, queries: dict[str, str]) -> dict[str, float]:
        """Evaluate a name->expr mapping, dropping series that don't resolve."""
        names = list(queries)
        values = await asyncio.gather(*(self.query(queries[n]) for n in names))
        return {n: v for n, v in zip(names, values) if v is not None}


class GaugeSampler:
    """Polls gauge series on an interval and keeps per-series max and last.

    Run it as a background task spanning a benchmark request so the report can
    state the peak ``running``/``waiting`` reached — i.e. whether the server
    cleared a round of rollouts in a single wave.
    """

    def __init__(self, client: PromClient, interval: float = 1.0):
        self.client = client
        self.interval = interval
        self.samples: dict[str, list[float]] = {k: [] for k in GAUGE_QUERIES}
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
        try:
            while True:
                snap = await self.client.snapshot(GAUGE_QUERIES)
                for name, value in snap.items():
                    self.samples[name].append(value)
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> dict[str, dict[str, float]]:
        """Stop sampling and return ``{series: {"max", "last"}}``."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        out: dict[str, dict[str, float]] = {}
        for name, vals in self.samples.items():
            if vals:
                out[name] = {"max": max(vals), "last": vals[-1]}
        return out
