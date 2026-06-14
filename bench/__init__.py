"""RSA latency benchmark harness.

A standalone client-side tool (sibling to ``rsa/``) that measures RSA query
latency and answer accuracy against a live vLLM backend, plus optional
server-side concurrency/throughput from Prometheus. Touches nothing in the
``vllm/`` package. See ``bench/README.md``.
"""
