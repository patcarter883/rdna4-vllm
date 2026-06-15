"""Answer extraction, normalization, and majority voting."""

import re
from collections import Counter

_BOXED = "\\boxed{"


def extract_boxed(text: str) -> str | None:
    """Return the content of the last ``\\boxed{...}`` in *text*.

    Uses a brace-depth scan so nested braces (e.g. ``\\boxed{\\frac{1}{2}}``)
    are handled, which a plain regex cannot do. Returns None if no balanced
    boxed expression is found.
    """
    idx = text.rfind(_BOXED)
    while idx != -1:
        start = idx + len(_BOXED)
        depth = 1
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
        # Unbalanced at this occurrence; try an earlier one.
        idx = text.rfind(_BOXED, 0, idx)
    return None


def normalize_answer(s: str) -> str:
    """Cheap string normalization so equivalent answers vote together."""
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\%", "%").replace("\\$", "$")
    s = re.sub(r"\s+", " ", s)
    # Iteratively peel surrounding whitespace, $ delimiters, trailing periods.
    while True:
        peeled = s.strip().strip("$").rstrip(".")
        if peeled == s:
            break
        s = peeled
    # Strip one layer of redundant surrounding braces: {42} -> 42
    if s.startswith("{") and s.endswith("}") and extract_balanced(s) == s[1:-1]:
        s = s[1:-1].strip()
    # Drop leading zeros on plain integers: 042 -> 42
    if re.fullmatch(r"0+\d+", s):
        s = s.lstrip("0") or "0"
    return s


def extract_balanced(s: str) -> str | None:
    """If *s* is a single balanced ``{...}`` group, return its inside."""
    if not (s.startswith("{") and s.endswith("}")):
        return None
    depth = 0
    for i, c in enumerate(s):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[1:-1] if i == len(s) - 1 else None
    return None


def majority_vote(
    answers: list[str | None],
) -> tuple[str, Counter] | None:
    """Majority vote over extracted answers.

    *answers* are raw extracted strings (None where extraction failed).
    Returns ``(winning_normalized_answer, tally)`` or None when no answer
    was extractable. Ties break toward the answer that appeared first.
    """
    normalized = [normalize_answer(a) if a is not None else None for a in answers]
    valid = [a for a in normalized if a]
    if not valid:
        return None
    tally = Counter(valid)
    best_count = max(tally.values())
    # First occurrence wins ties.
    for a in valid:
        if tally[a] == best_count:
            return a, tally
    return None  # unreachable


def consensus_ratio(answers: list[str | None]) -> tuple[str | None, int, int]:
    """Agreement on the top answer, for adaptive early-stopping.

    Returns ``(winner, agree_count, extractable_count)`` using the same vote
    and normalization as :func:`majority_vote` (so consensus and final
    selection agree), or ``(None, 0, 0)`` when nothing is extractable.
    """
    vote = majority_vote(answers)
    if vote is None:
        return None, 0, 0
    winner, tally = vote
    return winner, tally[winner], sum(tally.values())
