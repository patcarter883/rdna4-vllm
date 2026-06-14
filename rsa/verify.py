"""Deterministic candidate verification.

Verdicts are used to FILTER the aggregation sampling pool, never to pick a
single winner: RSA depends on population diversity (N >> K), so verification
only excludes provably broken candidates and falls back to the full
population when it would leave too few.

Verdict semantics:
- "fail":    deterministically broken (code crashes, unparseable answer)
- "pass":    survived every applicable check
- "unknown": no applicable check (verifier off, or nothing checkable)
"""

import logging
import re
import resource
import subprocess
import sys

from rsa import extract
from rsa.config import Verifier

logger = logging.getLogger("rsa.verify")

CODE_BLOCK_RE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
CODE_TIMEOUT_SECONDS = 5
CODE_MEMORY_BYTES = 512 * 1024 * 1024


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CODE_TIMEOUT_SECONDS,) * 2)
    resource.setrlimit(resource.RLIMIT_AS, (CODE_MEMORY_BYTES,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)


def run_code_block(code: str) -> bool:
    """Execute a candidate's python block; True if it exits cleanly.

    Best-effort sandbox: isolated interpreter (-I), CPU/memory/process/file
    rlimits, and a wall-clock timeout. This still executes model-generated
    code on the host — only enable the "code"/"auto" verifier when that is
    acceptable (see README).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            timeout=CODE_TIMEOUT_SECONDS + 1,
            preexec_fn=_limits,
            cwd="/tmp",
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


# sympify ultimately uses eval; refuse anything outside a math-y charset
# (no quotes, underscores, or shell metacharacters) before parsing.
_SAFE_ANSWER = re.compile(r"^[A-Za-z0-9\s+\-*/^(){}\[\].,=<>!|\\]+$")
_FRAC = re.compile(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}")
_SQRT = re.compile(r"\\sqrt\{([^{}]*)\}")


def _desugar_latex(s: str) -> str:
    """Rewrite common LaTeX constructs into sympify-able text."""
    prev = None
    while prev != s:
        prev = s
        s = _FRAC.sub(r"((\1)/(\2))", s)
        s = _SQRT.sub(r"sqrt(\1)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)  # \sin -> sin, etc.
    s = s.replace("^", "**")
    return s


def answer_parses(answer: str) -> bool:
    """True if a boxed answer parses as a numeric/symbolic expression.

    Heuristic by design: exotic-but-valid notations may fail to parse, so
    callers must only use this to filter (with filter_pool's fallback),
    never to reject a request outright.
    """
    s = extract.normalize_answer(answer)
    if not s or len(s) > 200 or not _SAFE_ANSWER.match(s):
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import sympy
    except ImportError:
        # Without sympy, only plain numerics are checkable.
        return True
    try:
        sympy.sympify(_desugar_latex(s), rational=True)
        return True
    except Exception:
        return False


def verdict(text: str, content: str, verifier: Verifier) -> str:
    """Score one candidate. See module docstring for semantics."""
    if verifier == "off":
        return "unknown"
    checked = False

    if verifier in ("code", "auto"):
        blocks = CODE_BLOCK_RE.findall(content) or CODE_BLOCK_RE.findall(text)
        if blocks:
            checked = True
            # The last block is usually the final program.
            if not run_code_block(blocks[-1]):
                return "fail"

    if verifier in ("math", "auto"):
        answer = extract.extract_boxed(content) or extract.extract_boxed(text)
        if answer is not None:
            checked = True
            if not answer_parses(answer):
                return "fail"

    return "pass" if checked else "unknown"


def filter_pool(candidates: list, verdicts: list[str], k: int) -> list:
    """Drop "fail" candidates from the sampling pool.

    Falls back to the full population when filtering would leave fewer
    than *k* candidates — a depleted pool is worse than a noisy one.
    """
    pool = [c for c, v in zip(candidates, verdicts) if v != "fail"]
    if len(pool) < min(k, len(candidates)):
        logger.warning(
            "verifier failed %d/%d candidates; using full population",
            len(candidates) - len(pool),
            len(candidates),
        )
        return candidates
    return pool
