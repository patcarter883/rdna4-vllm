"""Compare RSA against single-sample and self-consistency baselines.

One run_rsa() call per problem yields all three comparisons without extra
GPU time: the round-0 population is N independent samples (pass@1 baseline
and majority-vote/self-consistency baseline), and the aggregated result is
RSA. Problems file: JSON list of {"id", "question", "answer"}.

Example:
    .venv/bin/python -m rsa.compare rsa/aime_mini.json \\
        --rsa-n 8 --rsa-k 3 --rsa-t 2 --rsa-tail-tokens 1536 \\
        --rsa-max-tokens 5000 --rsa-verifier math
"""

import argparse
import asyncio
import json
import logging
import random
import time

from rsa import extract
from rsa.config import add_rsa_args, params_from_args
from rsa.core import BackendClient, Candidate, RSAResult, run_rsa

INSTRUCTION = "\n\nPut your final answer in \\boxed{}."


def answer_of(c: Candidate) -> str | None:
    raw = extract.extract_boxed(c.content) or extract.extract_boxed(c.text)
    return extract.normalize_answer(raw) if raw is not None else None


def evaluate(result: RSAResult, target: str) -> dict:
    target = extract.normalize_answer(target)
    round0 = [answer_of(c) for c in result.rounds[0]]
    final_round = [answer_of(c) for c in result.rounds[-1]]
    vote = extract.majority_vote(
        [
            extract.extract_boxed(c.content) or extract.extract_boxed(c.text)
            for c in result.rounds[0]
        ]
    )
    self_consistency = vote[0] if vote else None
    raw_final = extract.extract_boxed(result.final_text)
    rsa_answer = extract.normalize_answer(raw_final) if raw_final is not None else None
    return {
        "target": target,
        "round0_answers": round0,
        "round0_correct": sum(a == target for a in round0),
        "final_round_answers": final_round,
        "final_round_correct": sum(a == target for a in final_round),
        "self_consistency_answer": self_consistency,
        "self_consistency_correct": self_consistency == target,
        "rsa_answer": rsa_answer,
        "rsa_correct": rsa_answer == target,
        "selection": result.selection_method,
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "n_requests": result.usage.n_requests,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problems", help="JSON file: [{id, question, answer}]")
    parser.add_argument("--backend", default="http://localhost:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--out", default="/tmp/rsa_compare_results.json")
    add_rsa_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    params = params_from_args(args)
    with open(args.problems) as f:
        problems = json.load(f)
    client = BackendClient(args.backend, api_key=args.api_key, tokenizer=args.tokenizer)
    rows = []
    try:
        model = args.model or await client.default_model()
        for i, p in enumerate(problems):
            messages = [{"role": "user", "content": p["question"] + INSTRUCTION}]
            start = time.monotonic()
            result = await run_rsa(
                client, params, messages, model, rng=random.Random(i)
            )
            row = evaluate(result, p["answer"])
            row["id"] = p["id"]
            row["wall_seconds"] = round(time.monotonic() - start, 1)
            rows.append(row)
            print(
                f"RESULT [{p['id']}] target={row['target']} "
                f"pass@1={row['round0_correct']}/{params.n} "
                f"SC={'OK' if row['self_consistency_correct'] else 'X'}"
                f"({row['self_consistency_answer']}) "
                f"RSA={'OK' if row['rsa_correct'] else 'X'}"
                f"({row['rsa_answer']}) "
                f"finalround={row['final_round_correct']}/{params.n} "
                f"{row['wall_seconds']}s "
                f"{row['completion_tokens']}tok",
                flush=True,
            )
    finally:
        await client.close()
        with open(args.out, "w") as f:
            json.dump({"params": params.model_dump(), "rows": rows}, f, indent=1)

    n = params.n
    total0 = sum(r["round0_correct"] for r in rows)
    totalf = sum(r["final_round_correct"] for r in rows)
    print("\n=== SUMMARY ===", flush=True)
    print(f"problems: {len(rows)}")
    print(f"single-sample (pass@1 over round 0): {total0}/{len(rows) * n}")
    print(
        "self-consistency (majority of round 0): "
        f"{sum(r['self_consistency_correct'] for r in rows)}/{len(rows)}"
    )
    print(f"RSA final answer: {sum(r['rsa_correct'] for r in rows)}/{len(rows)}")
    print(f"per-trace accuracy after aggregation: {totalf}/{len(rows) * n}")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
