"""Aggregation prompt templates.

Kept deliberately minimal, following the RSA paper's (arXiv:2509.26626)
note that the method works without extensive prompt engineering.
"""

AGGREGATION_SYSTEM = (
    "You are given a problem and several candidate solutions. Some candidates "
    "may be incorrect or contain errors. Examine the candidate solutions and "
    "produce an improved, higher-quality solution to the problem. Reason "
    "carefully; if all candidates are flawed, solve the problem from scratch."
)

AGGREGATION_USER = """\
{query}

Below are {k} candidate solutions:

{candidates}

Examine these candidates and produce an improved, complete solution to the \
problem above. End with your final answer."""

FINAL_SELECTION_SYSTEM = (
    "You are given a problem and several candidate solutions. Select or "
    "synthesize the single best final answer."
)

FINAL_SELECTION_USER = """\
{query}

Below are {k} candidate solutions:

{candidates}

Select or synthesize the single best final answer to the problem above. \
Respond with only the final answer."""

TRUNCATION_MARKER = "[...truncated...]\n"


def render_query(messages: list[dict]) -> str:
    """Render the user-visible conversation into a single query string.

    For the common single-user-message case this is just that message's
    text. Multi-turn conversations are rendered as a ``User:/Assistant:``
    transcript with the final user message last. System messages are
    handled separately (see build_aggregation_messages).
    """
    turns = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(turns) == 1:
        return _content_text(turns[0])
    parts = []
    for m in turns:
        label = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{label}: {_content_text(m)}")
    return "\n\n".join(parts)


def _content_text(message: dict) -> str:
    """Extract plain text from a message's content (str or content parts)."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    # OpenAI content-parts format: [{"type": "text", "text": ...}, ...]
    return "\n".join(
        part.get("text", "") for part in content if part.get("type") == "text"
    )


def render_candidates(tails: list[str]) -> str:
    blocks = []
    for i, tail in enumerate(tails, start=1):
        blocks.append(f"=== Candidate {i} ===\n{tail}")
    return "\n\n".join(blocks)


def _build(
    system_template: str,
    user_template: str,
    query: str,
    candidate_tails: list[str],
    request_system: str | None,
) -> list[dict]:
    system = system_template
    if request_system:
        system = request_system.rstrip() + "\n\n" + system
    user = user_template.format(
        query=query,
        k=len(candidate_tails),
        candidates=render_candidates(candidate_tails),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_aggregation_messages(
    query: str,
    candidate_tails: list[str],
    request_system: str | None = None,
) -> list[dict]:
    return _build(
        AGGREGATION_SYSTEM, AGGREGATION_USER, query, candidate_tails, request_system
    )


def build_final_selection_messages(
    query: str,
    candidate_tails: list[str],
    request_system: str | None = None,
) -> list[dict]:
    return _build(
        FINAL_SELECTION_SYSTEM,
        FINAL_SELECTION_USER,
        query,
        candidate_tails,
        request_system,
    )


def extract_request_system(messages: list[dict]) -> str | None:
    """Concatenate any system/developer messages from the incoming request."""
    parts = [
        _content_text(m) for m in messages if m.get("role") in ("system", "developer")
    ]
    joined = "\n\n".join(p for p in parts if p)
    return joined or None
