"""Benchmark problem set: known-answer problems for the RSA latency harness.

The harness measures two things per problem:
- **latency** — wall-clock to run the full RSA loop (the metric we optimize),
- **accuracy** — whether the selected answer matches a known ground truth (the
  regression gate for accuracy-affecting changes like INT8 or mamba dtype).

The built-in set is deliberately *verifiable* arithmetic / number-theory with
unambiguous integer answers, so the accuracy gate is trustworthy without
trusting hand-copied competition answers. Drop real AIME'24/'25 problems into a
JSONL file and pass ``--questions path.jsonl`` to exercise the model the way the
ZAYA1 report did; each line is ``{"id": ..., "question": ..., "answer": ...}``.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# Appended to every question so the selected answer is extractable by
# rsa.extract.extract_boxed (the same path the RSA proxy votes over).
BOXED_INSTRUCTION = " Put your final answer in \\boxed{}."


@dataclass(frozen=True)
class Problem:
    id: str
    question: str
    answer: str  # canonical ground truth (compared after normalization)

    def prompt(self) -> str:
        """The user-message text, ensuring a boxed-answer instruction."""
        if "\\boxed" in self.question:
            return self.question
        return self.question.rstrip() + BOXED_INSTRUCTION


# Built-in smoke set. Every answer is independently checkable; these exist to
# validate the harness end-to-end and give a quick latency read, not to grade
# competition-level reasoning. Use a JSONL file for that.
BUILTIN: list[Problem] = [
    Problem("mult", "Compute 17 * 23.", "391"),
    Problem("sum100", "What is the sum of the integers from 1 to 100?", "5050"),
    Problem("divisors60", "How many positive divisors does 60 have?", "12"),
    Problem("pow2_10", "What is 2 raised to the 10th power?", "1024"),
    Problem("fact5", "Compute 5 factorial (5!).", "120"),
    Problem("mod7_4", "What is the remainder when 7^4 is divided by 100?", "1"),
]


def load_problems(source: str | None) -> list[Problem]:
    """Load problems from ``source``.

    ``None`` or ``"builtin"`` returns the built-in smoke set. Otherwise
    ``source`` is a path to a JSONL file with one ``{"id", "question",
    "answer"}`` object per line (blank lines and ``#`` comments ignored).
    """
    if source is None or source == "builtin":
        return list(BUILTIN)

    path = Path(source)
    problems: list[Problem] = []
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            try:
                problems.append(
                    Problem(
                        id=str(obj["id"]),
                        question=str(obj["question"]),
                        answer=str(obj["answer"]),
                    )
                )
            except KeyError as e:
                raise ValueError(
                    f"{path}:{lineno}: missing required key {e}"
                ) from None
    if not problems:
        raise ValueError(f"{path}: no problems found")
    return problems
