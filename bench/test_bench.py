"""Unit tests for the benchmark harness (mocked backend, no GPU).

Run from repo root:
    .venv/bin/python -m pytest bench/test_bench.py -q
"""

import asyncio
import json

from rsa import prompts
from rsa.config import RSAParams
from rsa.core import Candidate

from bench import run
from bench.problems import BUILTIN, Problem, load_problems

# ---------------------------------------------------------------- scoring


def test_score_answer_correct():
    extracted, correct = run.score_answer(r"so \boxed{391}", "391")
    assert extracted == "391"
    assert correct is True


def test_score_answer_normalizes():
    # $042$ and 42 must score equal via rsa.extract.normalize_answer.
    _, correct = run.score_answer(r"\boxed{ $042$ }", "42")
    assert correct is True


def test_score_answer_wrong_and_missing():
    assert run.score_answer(r"\boxed{7}", "42") == ("7", False)
    assert run.score_answer("no box at all", "42") == (None, False)


# ---------------------------------------------------------------- problems


def test_problem_prompt_adds_boxed_instruction():
    p = Problem("x", "Compute 1+1.", "2")
    assert "\\boxed{}" in p.prompt()
    # An existing boxed instruction is not duplicated.
    p2 = Problem("y", "Compute 1+1. Use \\boxed{}.", "2")
    assert p2.prompt().count("\\boxed") == 1


def test_load_builtin():
    assert load_problems("builtin") == BUILTIN
    assert load_problems(None) == BUILTIN


def test_load_jsonl(tmp_path):
    f = tmp_path / "q.jsonl"
    f.write_text(
        "# a comment\n"
        + json.dumps({"id": "a", "question": "Q1", "answer": "1"})
        + "\n\n"
        + json.dumps({"id": "b", "question": "Q2", "answer": "2"})
        + "\n"
    )
    problems = load_problems(str(f))
    assert [p.id for p in problems] == ["a", "b"]
    assert problems[0].answer == "1"


# ---------------------------------------------------------------- stats


def test_percentile_and_stats():
    assert run._percentile([], 90) == 0.0
    assert run._percentile([1.0], 90) == 1.0
    stats = run._latency_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["min"] == 1.0 and stats["max"] == 4.0
    assert stats["median"] == 2.5
    assert stats["n"] == 4


# ---------------------------------------------------------------- _run_one


class FakeBackend:
    """Duck-typed BackendClient returning canned candidates (cf. rsa tests)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self._lock = asyncio.Lock()

    async def complete(self, messages, **kwargs):
        async with self._lock:
            text = self.texts.pop(0) if self.texts else r"\boxed{0}"
        return Candidate(
            text="reasoning...\n" + text,
            content=text,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )

    async def tail(self, text, tail_tokens, model):
        if tail_tokens and len(text) > tail_tokens:
            return prompts.TRUNCATION_MARKER + text[-tail_tokens:]
        return text

    async def default_model(self):
        return "model"

    async def close(self):
        pass


def test_run_one_scores_and_aggregates():
    # All rollouts agree on 391 -> majority vote, correct.
    n = 4
    fake = FakeBackend([rf"work \boxed{{391}}" for _ in range(n)])
    params = RSAParams(n=n, k=2, t=1, tail_tokens=0, max_tokens=64)
    problem = Problem("mult", "Compute 17 * 23.", "391")

    result = asyncio.run(run._run_one(fake, params, "model", problem, None, 1.0))

    assert result.correct is True
    assert result.extracted == "391"
    assert result.selection_method == "majority_vote"
    assert result.n_requests == n
    assert result.latency_s >= 0.0
    assert result.peak_running is None  # no Prometheus client


def test_run_one_marks_wrong_answer():
    n = 3
    fake = FakeBackend([rf"\boxed{{7}}" for _ in range(n)])
    params = RSAParams(n=n, k=2, t=1, tail_tokens=0, max_tokens=64)
    problem = Problem("mult", "Compute 17 * 23.", "391")

    result = asyncio.run(run._run_one(fake, params, "model", problem, None, 1.0))

    assert result.correct is False
    assert result.extracted == "7"
