"""Unit tests for the RSA proxy (mocked backend, no GPU).

Run from repo root:
    .venv/bin/python -m pytest rsa/test_rsa.py -q
"""

import asyncio
import json
import random

import httpx
from fastapi.testclient import TestClient

from rsa import extract, prompts, verify
from rsa.config import RSAParams, ServerConfig, merge_params
from rsa.core import BackendClient, Candidate, advance_to_boundary, run_rsa
from rsa.server import create_app

# ---------------------------------------------------------------- extract


def test_extract_boxed_simple():
    assert extract.extract_boxed(r"The answer is \boxed{42}.") == "42"


def test_extract_boxed_nested():
    assert extract.extract_boxed(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_extract_boxed_last_of_multiple():
    text = r"first \boxed{1} then finally \boxed{2}"
    assert extract.extract_boxed(text) == "2"


def test_extract_boxed_absent_or_unbalanced():
    assert extract.extract_boxed("no box here") is None
    assert extract.extract_boxed(r"\boxed{unclosed") is None
    # Falls back to an earlier balanced occurrence.
    assert extract.extract_boxed(r"\boxed{ok} and \boxed{bad") == "ok"


def test_normalize():
    assert extract.normalize_answer(" $42$ .") == "42"
    assert extract.normalize_answer(r"\left(1, 2\right)") == "(1, 2)"
    assert extract.normalize_answer(r"\text{abc}") == "abc"
    assert extract.normalize_answer("{42}") == "42"
    assert extract.normalize_answer("042") == "42"


def test_majority_vote():
    winner, tally = extract.majority_vote(["42", " $42$", "7", None])
    assert winner == "42"
    assert tally["42"] == 2 and tally["7"] == 1


def test_majority_vote_tie_first_wins():
    winner, _ = extract.majority_vote(["7", "42", "7", "42"])
    assert winner == "7"


def test_majority_vote_nothing():
    assert extract.majority_vote([None, None]) is None


# ---------------------------------------------------------------- config


def test_params_merge_dict():
    defaults = RSAParams()
    merged = merge_params(defaults, {"n": 4, "tail_tokens": 0})
    assert merged.n == 4
    assert merged.tail_tokens == 0
    assert merged.k == defaults.k  # untouched fields keep defaults


def test_params_merge_optout():
    assert merge_params(RSAParams(), False) is None
    assert merge_params(RSAParams(), {"enabled": False}) is None


def test_params_merge_default():
    defaults = RSAParams(n=8)
    assert merge_params(defaults, None).n == 8
    assert merge_params(defaults, True).n == 8


# ---------------------------------------------------------------- core


class FakeBackend:
    """Stands in for BackendClient; returns canned texts in order."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []  # message lists, in request order
        self.budgets = []  # max_tokens seen per request, in order
        self._lock = asyncio.Lock()

    async def complete(self, messages, **kwargs):
        async with self._lock:
            self.calls.append(messages)
            self.budgets.append(kwargs.get("max_tokens"))
            text = self.texts.pop(0) if self.texts else r"fallback \boxed{0}"
        return Candidate(
            text="thinking...\n" + text,
            content=text,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )

    async def complete_n(self, messages, *, n, **kwargs):
        # The real backend prefills once and samples n; the fake delegates to
        # complete n times so tests can inspect each rollout's prompt/budget.
        return [await self.complete(messages, **kwargs) for _ in range(n)]

    async def tail(self, text, tail_tokens, model):
        if tail_tokens and len(text) > tail_tokens:
            return prompts.TRUNCATION_MARKER + text[-tail_tokens:]
        return text

    async def default_model(self):
        return "model"

    async def close(self):
        pass


def test_round_loop_population_and_prompts():
    n, k, t = 4, 2, 3
    texts = [rf"sol {i} \boxed{{42}}" for i in range(n * t)]
    fake = FakeBackend(texts)
    params = RSAParams(n=n, k=k, t=t, tail_tokens=0, max_tokens=100)
    messages = [{"role": "user", "content": "What is 6*7?"}]

    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))

    assert len(result.rounds) == t
    assert all(len(r) == n for r in result.rounds)
    assert result.usage.n_requests == n * t
    assert result.selection_method == "majority_vote"
    assert result.vote_detail["winner"] == "42"

    # Round 0 requests are the original messages, untouched.
    for call in fake.calls[:n]:
        assert call == messages
    # Aggregation requests contain exactly K candidate blocks and the query.
    for call in fake.calls[n:]:
        user = call[-1]["content"]
        assert user.count("=== Candidate") == k
        assert "What is 6*7?" in user


def test_selection_final_agg_fallback_for_general_queries():
    # No boxed answers anywhere -> auto falls back to a final aggregation
    # call, whose content becomes the final text.
    n, t = 3, 1
    texts = [f"opinion {i}" for i in range(n)] + ["the synthesized best answer"]
    fake = FakeBackend(texts)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100)
    messages = [{"role": "user", "content": "Explain X briefly."}]

    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))

    assert result.selection_method == "final_aggregation"
    assert result.final_text == "the synthesized best answer"
    assert result.usage.n_requests == n + 1
    # The selection prompt asks for only the final answer.
    assert "only the final answer" in fake.calls[-1][-1]["content"]


def test_empty_boxed_does_not_count_as_vote():
    # One candidate echoes the literal \boxed{} from the prompt (extracts as
    # empty) and one has a real answer: only 1 real vote, so "auto" must
    # fall back to final aggregation, not declare a majority of one.
    n, t = 2, 1
    texts = [r"they want \boxed{}", r"answer \boxed{391}", "synthesized"]
    fake = FakeBackend(texts)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100)
    messages = [{"role": "user", "content": "17*23? Use \\boxed{}."}]

    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))

    assert result.selection_method == "final_aggregation"
    assert result.final_text == "synthesized"


def test_vote_winner_prefers_finished_candidate():
    class TruncatingBackend(FakeBackend):
        async def complete(self, messages, **kwargs):
            c = await super().complete(messages, **kwargs)
            # First response was cut off by the token budget.
            if len(self.calls) == 1:
                c.finish_reason = "length"
                c.content = r"rambling cut-off reasoning \boxed{42}"
            return c

    n, t = 2, 1
    texts = ["ignored", r"clean solution \boxed{42}"]
    fake = TruncatingBackend(texts)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100)
    messages = [{"role": "user", "content": "6*7? Use \\boxed{}."}]

    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))

    assert result.selection_method == "majority_vote"
    assert result.final_text == r"clean solution \boxed{42}"


def test_tail_truncation_marks_and_bounds():
    fake = FakeBackend([])
    long_text = "x" * 10000
    tail = asyncio.run(fake.tail(long_text, 100, "model"))
    assert tail.startswith(prompts.TRUNCATION_MARKER)
    short = asyncio.run(fake.tail("short", 100, "model"))
    assert short == "short"


def test_system_message_survives_aggregation():
    n, t = 2, 2
    texts = [rf"s{i} \boxed{{1}}" for i in range(n * t)]
    fake = FakeBackend(texts)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100)
    messages = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "Combien font 2+2?"},
    ]
    asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    agg_system = fake.calls[-1][0]
    assert agg_system["role"] == "system"
    assert agg_system["content"].startswith("Answer in French.")


# ------------------------------------------------------- tail boundaries


def test_advance_to_boundary_paragraph():
    tail = "broken half-thought\nstill broken\n\nA clean paragraph." + "x" * 500
    assert advance_to_boundary(tail).startswith("A clean paragraph.")


def test_advance_to_boundary_line_fallback():
    tail = "broken half-thought\nA clean line. " + "x" * 500
    assert advance_to_boundary(tail).startswith("A clean line.")


def test_advance_to_boundary_no_boundary_keeps_cut():
    tail = "y" * 1000
    assert advance_to_boundary(tail) == tail


def test_advance_to_boundary_ignores_distant_breaks():
    # The only break is far past the look-back window; keep the raw cut.
    tail = "x" * 900 + "\n\n" + "y" * 100
    assert advance_to_boundary(tail) == tail


def test_local_tokenizer_tail_slices_and_cuts_at_boundary():
    class StubTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(range(len(text)))  # 1 token per char

        def decode(self, ids, skip_special_tokens=False):
            return "junk\n\n" + "z" * (len(ids) - 6)

    client = BackendClient("http://localhost:9/v1")  # never contacted
    client._tokenizer = StubTokenizer()
    long_text = "a" * 1000

    tail = asyncio.run(client.tail(long_text, 100, "model"))

    assert tail.startswith(prompts.TRUNCATION_MARKER)
    # Boundary cut dropped the leading "junk\n\n".
    assert tail.removeprefix(prompts.TRUNCATION_MARKER).startswith("z")
    short = asyncio.run(client.tail("b" * 50, 100, "model"))
    assert short == "b" * 50
    asyncio.run(client.close())


# ------------------------------------------------------------- verifier


def test_verifier_math_verdicts():
    assert verify.verdict("", r"answer \boxed{42}", "math") == "pass"
    assert verify.verdict("", r"answer \boxed{\frac{1}{2}}", "math") == "pass"
    assert verify.verdict("", r"answer \boxed{((1 +* 2}", "math") == "fail"
    assert verify.verdict("", "no boxed answer", "math") == "unknown"
    assert verify.verdict("", r"\boxed{42}", "off") == "unknown"


def test_verifier_code_verdicts():
    good = "```python\nprint(17 * 23)\n```\nthen \\boxed{391}"
    bad = "```python\nraise ValueError('broken')\n```"
    assert verify.verdict("", good, "code") == "pass"
    assert verify.verdict("", bad, "code") == "fail"
    assert verify.verdict("", "prose only", "code") == "unknown"


def test_verifier_pool_filter_and_fallback():
    cands = ["a", "b", "c", "d"]
    pool = verify.filter_pool(cands, ["pass", "fail", "unknown", "pass"], k=2)
    assert pool == ["a", "c", "d"]
    # Filtering below k falls back to the full population.
    pool = verify.filter_pool(cands, ["fail", "fail", "fail", "pass"], k=2)
    assert pool == cands


def test_verifier_excludes_broken_candidates_from_aggregation():
    # 3 candidates; one has an unparseable boxed answer. With the math
    # verifier on, aggregation sets must never include the broken one.
    n, k, t = 3, 2, 2
    round0 = [r"good \boxed{7}", r"broken \boxed{((1 +* 2}", r"good \boxed{7}"]
    round1 = [rf"agg {i} \boxed{{7}}" for i in range(n)]
    fake = FakeBackend(round0 + round1)
    params = RSAParams(n=n, k=k, t=t, tail_tokens=0, max_tokens=100, verifier="math")
    messages = [{"role": "user", "content": "q? \\boxed{}"}]

    asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))

    for call in fake.calls[n : n * 2]:
        assert "broken" not in call[-1]["content"]


# --------------------------------------------------------------- adaptive


def test_early_stop_off_by_default_runs_full_t():
    # Unanimous answers but early_stop defaults off: still runs all T rounds.
    n, t = 3, 3
    texts = [rf"s{i} \boxed{{42}}" for i in range(n * t)]
    fake = FakeBackend(texts)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100)
    messages = [{"role": "user", "content": "6*7? \\boxed{}"}]
    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert len(result.rounds) == t
    assert result.stopped_early is False
    assert result.usage.n_requests == n * t


def test_early_stop_on_consensus():
    # Round 0 unanimous -> consensus -> stop after round 0 (no aggregation).
    n, t = 4, 3
    texts = [rf"s{i} \boxed{{42}}" for i in range(n)]
    fake = FakeBackend(texts)
    params = RSAParams(
        n=n, k=2, t=t, tail_tokens=0, max_tokens=100,
        early_stop=True, consensus_threshold=0.9,
    )
    messages = [{"role": "user", "content": "6*7? \\boxed{}"}]
    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert len(result.rounds) == 1
    assert result.stopped_early is True
    assert result.usage.n_requests == n  # only round 0 generated


def test_no_early_stop_below_threshold():
    # Round 0 splits 2/2 -> ratio 0.5 < 0.9 -> runs the aggregation round.
    n, t = 4, 2
    round0 = [r"a \boxed{42}", r"b \boxed{42}", r"c \boxed{7}", r"d \boxed{7}"]
    round1 = [rf"agg{i} \boxed{{42}}" for i in range(n)]
    fake = FakeBackend(round0 + round1)
    params = RSAParams(
        n=n, k=2, t=t, tail_tokens=0, max_tokens=100,
        early_stop=True, consensus_threshold=0.9,
    )
    messages = [{"role": "user", "content": "6*7? \\boxed{}"}]
    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert len(result.rounds) == t
    assert result.stopped_early is False


def test_consensus_needs_min_votes():
    # Only 1 extractable answer (others non-boxed): below consensus_min_votes,
    # so no early stop on a "majority of one".
    n, t = 3, 2
    round0 = [r"only one \boxed{42}", "no box here", "prose only"]
    round1 = [rf"agg{i} \boxed{{42}}" for i in range(n)]
    fake = FakeBackend(round0 + round1)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100, early_stop=True)
    messages = [{"role": "user", "content": "6*7? \\boxed{}"}]
    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert len(result.rounds) == t


def test_agg_budget_applies_to_aggregation_only():
    # Round 0 uses the full budget; aggregation rounds use agg_max_tokens.
    n, t = 3, 2
    texts = [rf"s{i} \boxed{{1}}" for i in range(n * t)]
    fake = FakeBackend(texts)
    params = RSAParams(
        n=n, k=2, t=t, tail_tokens=0, max_tokens=500, agg_max_tokens=50,
    )
    messages = [{"role": "user", "content": "1*1? \\boxed{}"}]
    asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert fake.budgets[:n] == [500] * n
    assert fake.budgets[n : n * t] == [50] * n


def test_staged_expansion_tops_up_without_consensus():
    # n_min=2 round-0 rollouts disagree -> top up to the full n=4.
    n, n_min, t = 4, 2, 1
    round0 = [r"a \boxed{1}", r"b \boxed{2}", r"c \boxed{3}", r"d \boxed{4}"]
    fake = FakeBackend(round0)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100, n_min=n_min)
    messages = [{"role": "user", "content": "q? \\boxed{}"}]
    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert len(result.population) == n
    assert len(result.rounds[0]) == n
    assert result.usage.n_requests == n  # n_min + top-up


def test_staged_expansion_stops_at_n_min_on_consensus():
    # n_min rollouts already agree -> no top-up (independent of early_stop).
    n, n_min, t = 4, 2, 1
    round0 = [r"a \boxed{42}", r"b \boxed{42}"]
    fake = FakeBackend(round0)
    params = RSAParams(n=n, k=2, t=t, tail_tokens=0, max_tokens=100, n_min=n_min)
    messages = [{"role": "user", "content": "q? \\boxed{}"}]
    result = asyncio.run(run_rsa(fake, params, messages, "model", rng=random.Random(0)))
    assert len(result.population) == n_min
    assert result.usage.n_requests == n_min


# ---------------------------------------------------------------- server


def _make_client(rsa_texts=None):
    config = ServerConfig(
        defaults=RSAParams(n=2, k=2, t=1, tail_tokens=0, max_tokens=64)
    )
    app = create_app(config)
    client = TestClient(app)
    client.__enter__()  # run lifespan

    app.state.backend = FakeBackend(rsa_texts or [r"a \boxed{5}", r"b \boxed{5}"])

    def passthrough_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"passthrough": True, "path": request.url.path},
        )

    app.state.passthrough = httpx.AsyncClient(
        base_url="http://backend",
        transport=httpx.MockTransport(passthrough_handler),
    )
    return client


def test_server_rsa_response_shape():
    client = _make_client()
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "2+3? Use \\boxed{}."}],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == r"a \boxed{5}"
    # Usage summed across both rollouts.
    assert body["usage"]["prompt_tokens"] == 20
    assert body["usage"]["completion_tokens"] == 40
    assert body["rsa"]["selection"] == "majority_vote"
    assert body["rsa"]["requests"] == 2
    client.__exit__(None, None, None)


def test_server_passthrough_tools_and_optout():
    client = _make_client()
    tool_req = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    r = client.post("/v1/chat/completions", json=tool_req)
    assert r.json() == {"passthrough": True, "path": "/v1/chat/completions"}

    optout = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "rsa": False,
    }
    r = client.post("/v1/chat/completions", json=optout)
    assert r.json()["passthrough"] is True

    r = client.get("/v1/models")
    assert r.json()["passthrough"] is True
    client.__exit__(None, None, None)


def test_server_per_request_override():
    # n=3 via extra body; t=1 so 3 rollouts total.
    texts = [r"x \boxed{9}"] * 3
    client = _make_client(texts)
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "q \\boxed{}?"}],
            "rsa": {"n": 3},
        },
    )
    assert r.status_code == 200
    assert r.json()["rsa"]["population"] == 3
    assert r.json()["rsa"]["requests"] == 3
    client.__exit__(None, None, None)


def test_server_streaming_pseudo_stream():
    client = _make_client()
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "2+3? \\boxed{}"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_raw()).decode()
    events = [
        line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == r"a \boxed{5}"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["total_tokens"] == 60
    client.__exit__(None, None, None)


def test_server_reports_adaptive_metadata():
    client = _make_client()  # defaults: n=2, k=2, t=1
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "model",
            "messages": [{"role": "user", "content": "2+3? \\boxed{}"}],
        },
    )
    body = r.json()
    assert body["rsa"]["rounds"] == 1
    assert body["rsa"]["rounds_configured"] == 1
    assert body["rsa"]["population"] == 2
    assert body["rsa"]["population_configured"] == 2
    assert body["rsa"]["stopped_early"] is False
    client.__exit__(None, None, None)
