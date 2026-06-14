"""RSA orchestrator: backend client, tail truncation, and the round loop."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass

import httpx
import openai

from rsa import extract, prompts, verify
from rsa.config import RSAParams

logger = logging.getLogger("rsa")


def advance_to_boundary(tail: str, max_skip_fraction: float = 0.1) -> str:
    """Advance a sliced tail's start to the next semantic boundary.

    A fixed token cut can land mid-sentence or mid-derivation, handing the
    next round a severed thought it wastes tokens repairing. Look for a
    paragraph break (then a line break) within the leading fraction of the
    tail and start there instead; give up and keep the raw cut if none is
    close enough.
    """
    window = max(int(len(tail) * max_skip_fraction), 1)
    cut = tail.find("\n\n", 0, window)
    if cut != -1:
        return tail[cut + 2 :].lstrip("\n")
    cut = tail.find("\n", 0, window)
    if cut != -1:
        return tail[cut + 1 :]
    return tail


class RSAError(Exception):
    """A whole RSA round failed; maps to HTTP 502 in the server."""


@dataclass
class Candidate:
    text: str  # reasoning + content concatenated (aggregation input)
    content: str  # message.content only (answer-bearing part)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    n_requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, candidate: Candidate) -> None:
        self.prompt_tokens += candidate.prompt_tokens
        self.completion_tokens += candidate.completion_tokens
        self.n_requests += 1


@dataclass
class RSAResult:
    final_text: str
    population: list[Candidate]
    rounds: list[list[Candidate]]
    usage: UsageTotals
    selection_method: str  # "majority_vote" | "final_aggregation" | "sample"
    vote_detail: dict | None = None


class BackendClient:
    """Thin async client for the backend vLLM OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 1800.0,
        tokenizer: str | None = None,
    ):
        self.base_url = base_url
        root = base_url.rstrip("/").removesuffix("/v1")
        self.openai = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0
        )
        self.http = httpx.AsyncClient(base_url=root, timeout=60.0)
        self._tokenize_broken = False
        self._tokenizer_name = tokenizer
        self._tokenizer = None  # None = not tried, False = unavailable
        self._tokenizer_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.openai.close()
        await self.http.aclose()

    async def default_model(self) -> str:
        models = await self.openai.models.list()
        return models.data[0].id

    async def complete(
        self,
        messages: list[dict],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int = 1,
    ) -> Candidate | None:
        """One chat completion; returns None on permanent failure."""
        attempt = 0
        clamped = False
        while True:
            try:
                kwargs = {} if clamped else {"max_tokens": max_tokens}
                resp = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )
                msg = resp.choices[0].message
                reasoning = (
                    getattr(msg, "reasoning", None)
                    or getattr(msg, "reasoning_content", None)
                    or ""
                )
                content = msg.content or ""
                text = (
                    (reasoning.rstrip() + "\n" + content).strip()
                    if reasoning
                    else content
                )
                usage = resp.usage
                return Candidate(
                    text=text,
                    content=content or text,
                    finish_reason=resp.choices[0].finish_reason,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                )
            except openai.BadRequestError as e:
                # Typically max_tokens exceeding remaining context for a long
                # aggregation prompt; retry once letting the server pick.
                if not clamped:
                    logger.warning("400 from backend, retrying clamped: %s", e)
                    clamped = True
                    continue
                logger.error("rollout failed permanently: %s", e)
                return None
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt < max_retries:
                    attempt += 1
                    logger.warning("rollout transport error, retry %d: %s", attempt, e)
                    continue
                logger.error("rollout failed after %d retries: %s", attempt, e)
                return None
            except openai.APIStatusError as e:
                logger.error("rollout failed with status %s: %s", e.status_code, e)
                return None

    async def complete_n(
        self,
        messages: list[dict],
        *,
        n: int,
        model: str,
        temperature: float,
        max_tokens: int,
        max_retries: int = 1,
    ) -> list[Candidate]:
        """One chat request with the native ``n`` parameter: vLLM prefills the
        shared prompt ONCE and samples ``n`` divergent traces in a single batch
        (max GPU saturation, minimal API overhead). Returns ``n`` Candidates, or
        [] on permanent failure. Prompt tokens are attributed once (to the first
        candidate) to reflect the single shared prefill — so the round's reported
        prompt-token total is P, not n*P as in the fan-out path."""
        attempt = 0
        clamped = False
        while True:
            try:
                kwargs = {} if clamped else {"max_tokens": max_tokens}
                resp = await self.openai.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                    **kwargs,
                )
                usage = resp.usage
                prompt_tokens = usage.prompt_tokens if usage else 0
                comp_total = usage.completion_tokens if usage else 0
                num = len(resp.choices) or 1
                out: list[Candidate] = []
                for i, ch in enumerate(resp.choices):
                    msg = ch.message
                    reasoning = (
                        getattr(msg, "reasoning", None)
                        or getattr(msg, "reasoning_content", None)
                        or ""
                    )
                    content = msg.content or ""
                    text = (
                        (reasoning.rstrip() + "\n" + content).strip()
                        if reasoning
                        else content
                    )
                    out.append(
                        Candidate(
                            text=text,
                            content=content or text,
                            finish_reason=ch.finish_reason,
                            prompt_tokens=prompt_tokens if i == 0 else 0,
                            completion_tokens=comp_total // num,
                        )
                    )
                return out
            except openai.BadRequestError as e:
                if not clamped:
                    logger.warning("400 from backend (n=%d), retrying clamped: %s", n, e)
                    clamped = True
                    continue
                logger.error("n-rollout failed permanently: %s", e)
                return []
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                if attempt < max_retries:
                    attempt += 1
                    logger.warning(
                        "n-rollout transport error, retry %d: %s", attempt, e
                    )
                    continue
                logger.error("n-rollout failed after %d retries: %s", attempt, e)
                return []
            except openai.APIStatusError as e:
                logger.error("n-rollout failed with status %s: %s", e.status_code, e)
                return []

    async def _get_tokenizer(self, model: str):
        """Lazily load a local HF tokenizer; False when unavailable."""
        if self._tokenizer is not None:
            return self._tokenizer
        async with self._tokenizer_lock:
            if self._tokenizer is not None:
                return self._tokenizer
            name = self._tokenizer_name
            try:
                if name is None:
                    # vLLM reports the underlying HF repo in the model
                    # card's "root" field (the id may be an alias like
                    # "model" when --served-model-name is used).
                    r = await self.http.get("/v1/models")
                    r.raise_for_status()
                    entry = r.json()["data"][0]
                    name = entry.get("root") or entry["id"]

                def load():
                    from transformers import AutoTokenizer

                    return AutoTokenizer.from_pretrained(name)

                self._tokenizer = await asyncio.to_thread(load)
                logger.info("loaded local tokenizer %r", name)
            except Exception as e:
                self._tokenizer = False
                logger.warning(
                    "local tokenizer unavailable (%s); "
                    "tails will use the backend /tokenize endpoint",
                    e,
                )
        return self._tokenizer

    async def tail(self, text: str, tail_tokens: int, model: str) -> str:
        """Truncate *text* to its final *tail_tokens* tokens.

        Token-exact via a local HF tokenizer when available, else the
        backend's /tokenize + /detokenize endpoints, else a character
        approximation. All paths advance the cut to a semantic boundary
        (paragraph/line break) so aggregation prompts never start
        mid-thought.
        """
        if tail_tokens <= 0:
            return text
        # Any text of <= tail_tokens characters cannot exceed
        # tail_tokens tokens; skip tokenization entirely.
        if len(text) <= tail_tokens:
            return text

        cut = None
        tokenizer = await self._get_tokenizer(model)
        if tokenizer:

            def token_slice():
                ids = tokenizer.encode(text, add_special_tokens=False)
                if len(ids) <= tail_tokens:
                    return None
                return tokenizer.decode(ids[-tail_tokens:], skip_special_tokens=False)

            cut = await asyncio.to_thread(token_slice)
            if cut is None:
                return text
        elif not self._tokenize_broken:
            try:
                r = await self.http.post(
                    "/tokenize",
                    json={
                        "model": model,
                        "prompt": text,
                        "add_special_tokens": False,
                    },
                )
                r.raise_for_status()
                ids = r.json()["tokens"]
                if len(ids) <= tail_tokens:
                    return text
                r = await self.http.post(
                    "/detokenize",
                    json={"model": model, "tokens": ids[-tail_tokens:]},
                )
                r.raise_for_status()
                cut = r.json()["prompt"]
            except (httpx.HTTPError, KeyError) as e:
                self._tokenize_broken = True
                logger.warning(
                    "tokenize endpoint unavailable (%s); "
                    "falling back to char-approximate tails",
                    e,
                )
        if cut is None:
            # Char fallback: ~4 chars/token.
            approx = tail_tokens * 4
            if len(text) <= approx:
                return text
            cut = text[-approx:]
        return prompts.TRUNCATION_MARKER + advance_to_boundary(cut)


async def _run_round(
    client: BackendClient,
    message_sets: list[list[dict]],
    *,
    model: str,
    params: RSAParams,
    semaphore: asyncio.Semaphore,
    usage: UsageTotals,
    round_idx: int,
) -> list[Candidate]:
    async def one(messages: list[dict]) -> Candidate | None:
        async with semaphore:
            return await client.complete(
                messages,
                model=model,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                max_retries=params.max_retries,
            )

    start = time.monotonic()
    results = await asyncio.gather(*(one(m) for m in message_sets))
    population = [c for c in results if c is not None]
    for c in population:
        usage.add(c)
    if not population:
        raise RSAError(f"round {round_idx}: all {len(message_sets)} rollouts failed")
    logger.info(
        "round %d: %d/%d candidates, %d prompt + %d completion tokens, %.1fs",
        round_idx,
        len(population),
        len(message_sets),
        sum(c.prompt_tokens for c in population),
        sum(c.completion_tokens for c in population),
        time.monotonic() - start,
    )
    return population


async def _run_round_shared(
    client: BackendClient,
    messages: list[dict],
    *,
    n: int,
    model: str,
    params: RSAParams,
    usage: UsageTotals,
    round_idx: int,
) -> list[Candidate]:
    """Run a round from a SINGLE shared prompt via the native ``n`` parameter —
    one request, one shared prefill, ``n`` parallel divergent traces. Used for
    the expansion phase (round 0, where all N rollouts share the prompt) and for
    'shared' aggregation."""
    start = time.monotonic()
    population = await client.complete_n(
        messages,
        n=n,
        model=model,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
        max_retries=params.max_retries,
    )
    for c in population:
        usage.add(c)
    if not population:
        raise RSAError(f"round {round_idx}: n={n} shared rollout failed")
    logger.info(
        "round %d: %d/%d candidates (n-parallel), %d prompt + %d completion "
        "tokens, %.1fs",
        round_idx,
        len(population),
        n,
        sum(c.prompt_tokens for c in population),
        sum(c.completion_tokens for c in population),
        time.monotonic() - start,
    )
    return population


async def _tails_for(
    client: BackendClient,
    population: list[Candidate],
    params: RSAParams,
    model: str,
) -> dict[int, str]:
    """Compute each candidate's tail once per round (memoized by identity)."""
    tails = await asyncio.gather(
        *(client.tail(c.text, params.tail_tokens, model) for c in population)
    )
    return {id(c): t for c, t in zip(population, tails)}


async def _verified_pool(
    population: list[Candidate], params: RSAParams
) -> list[Candidate]:
    """The aggregation sampling pool, minus verifiably broken candidates.

    Verdicts run concurrently in threads (code verification shells out and
    can block for seconds per candidate).
    """
    if params.verifier == "off":
        return population
    verdicts = await asyncio.gather(
        *(
            asyncio.to_thread(verify.verdict, c.text, c.content, params.verifier)
            for c in population
        )
    )
    pool = verify.filter_pool(population, list(verdicts), params.k)
    if len(pool) < len(population):
        logger.info(
            "verifier excluded %d/%d candidates from sampling",
            len(population) - len(pool),
            len(population),
        )
    return pool


async def run_rsa(
    client: BackendClient,
    params: RSAParams,
    messages: list[dict],
    model: str,
    rng: random.Random | None = None,
) -> RSAResult:
    """Run the full RSA loop and return the aggregated result.

    *messages* is the incoming OpenAI-style message list. Round 0 sends it
    unchanged; later rounds rebuild aggregation prompts from the rendered
    query plus sampled candidate tails.
    """
    rng = rng or random.Random()
    usage = UsageTotals()
    semaphore = asyncio.Semaphore(params.max_concurrency)
    query = prompts.render_query(messages)
    request_system = prompts.extract_request_system(messages)

    # Expansion (round 0): all N rollouts share the same prompt, so use the
    # native n parameter — vLLM prefills once and samples N divergent traces in
    # one batch (vs N separate requests re-prefilling / relying on prefix cache).
    if params.expand_with_n:
        population = await _run_round_shared(
            client, messages, n=params.n, model=model, params=params,
            usage=usage, round_idx=0,
        )
    else:
        population = await _run_round(
            client, [messages] * params.n, model=model, params=params,
            semaphore=semaphore, usage=usage, round_idx=0,
        )
    rounds = [population]

    for t in range(1, params.t):
        pool = await _verified_pool(population, params)
        tails = await _tails_for(client, pool, params, model)
        if params.aggregate == "shared":
            # One shared aggregation prompt over the full pool, sampled N times
            # via the native n parameter — the aggregation prefill happens ONCE
            # (and is prefix-cacheable) instead of N divergent re-prefills.
            # Diversity comes from temperature, not subset sampling.
            shared = prompts.build_aggregation_messages(
                query, [tails[id(c)] for c in pool], request_system
            )
            population = await _run_round_shared(
                client, shared, n=params.n, model=model, params=params,
                usage=usage, round_idx=t,
            )
        else:
            message_sets = []
            for _ in range(params.n):
                chosen = rng.sample(pool, k=min(params.k, len(pool)))
                message_sets.append(
                    prompts.build_aggregation_messages(
                        query, [tails[id(c)] for c in chosen], request_system
                    )
                )
            population = await _run_round(
                client, message_sets, model=model, params=params,
                semaphore=semaphore, usage=usage, round_idx=t,
            )
        rounds.append(population)

    final_pool = await _verified_pool(population, params)
    final_text, method, vote_detail = await _select(
        client, params, final_pool, query, request_system, model, rng, usage
    )
    logger.info(
        "selection=%s, total: %d requests, %d prompt + %d completion tokens",
        method,
        usage.n_requests,
        usage.prompt_tokens,
        usage.completion_tokens,
    )
    return RSAResult(
        final_text=final_text,
        population=population,
        rounds=rounds,
        usage=usage,
        selection_method=method,
        vote_detail=vote_detail,
    )


async def _select(
    client: BackendClient,
    params: RSAParams,
    population: list[Candidate],
    query: str,
    request_system: str | None,
    model: str,
    rng: random.Random,
    usage: UsageTotals,
) -> tuple[str, str, dict | None]:
    """Pick the final answer text from the final population."""

    def boxed(c: Candidate) -> str | None:
        return extract.extract_boxed(c.content) or extract.extract_boxed(c.text)

    if params.selection == "sample":
        return rng.choice(population).content, "sample", None

    # Normalize up front so empty answers (a literal "\boxed{}" echoed from
    # the prompt) don't count as extractable votes.
    answers = [boxed(c) for c in population]
    normalized = [
        extract.normalize_answer(a) if a is not None else None for a in answers
    ]
    extractable = sum(1 for a in normalized if a)
    want_vote = params.selection == "majority" or (
        params.selection == "auto" and extractable >= 2
    )
    if want_vote:
        vote = extract.majority_vote(answers)
        if vote is not None:
            winner, tally = vote
            # Return the full text of the top-voted candidate so the client
            # receives a complete solution, not just the boxed token —
            # preferring candidates that finished cleanly over ones cut off
            # by the token budget mid-reasoning.
            matching = [c for c, a in zip(population, normalized) if a == winner]
            best = next((c for c in matching if c.finish_reason == "stop"), matching[0])
            return (
                best.content,
                "majority_vote",
                {"winner": winner, "tally": dict(tally)},
            )
        if params.selection == "majority":
            # Nothing extractable; cheap fallback.
            return rng.choice(population).content, "sample", None

    # Fallback (and selection == "final_agg"): one final aggregation call.
    chosen = rng.sample(population, k=min(params.k, len(population)))
    tails = await _tails_for(client, chosen, params, model)
    msgs = prompts.build_final_selection_messages(
        query, [tails[id(c)] for c in chosen], request_system
    )
    final = await client.complete(
        msgs,
        model=model,
        temperature=0.3,
        max_tokens=params.max_tokens,
        max_retries=params.max_retries,
    )
    if final is None:
        # Last resort: don't fail the request over the selection call.
        return rng.choice(population).content, "sample", None
    usage.add(final)
    return final.content, "final_aggregation", None
