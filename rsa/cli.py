"""Single-question RSA runner for manual testing and debugging.

Example:
    .venv/bin/python -m rsa.cli --backend http://localhost:8000/v1 \\
        -q "What is 17*23? Put your final answer in \\boxed{}." \\
        --rsa-n 4 --rsa-k 2 --rsa-t 2
"""

import argparse
import asyncio
import logging

from rsa.config import add_rsa_args, params_from_args
from rsa.core import BackendClient, run_rsa


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="http://localhost:8000/v1", help="vLLM base URL"
    )
    parser.add_argument("--model", default=None, help="model id (default: first)")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--system", default=None, help="optional system prompt")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="HF tokenizer name/path for local tails (default: auto-resolve)",
    )
    add_rsa_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    params = params_from_args(args)
    client = BackendClient(args.backend, api_key=args.api_key, tokenizer=args.tokenizer)
    try:
        model = args.model or await client.default_model()
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.question})

        result = await run_rsa(client, params, messages, model)

        print("\n" + "=" * 72)
        print(f"selection: {result.selection_method}")
        if result.vote_detail:
            print(f"vote: {result.vote_detail}")
        u = result.usage
        print(
            f"usage: {u.n_requests} requests, {u.prompt_tokens} prompt + "
            f"{u.completion_tokens} completion = {u.total_tokens} tokens"
        )
        print("=" * 72)
        print(result.final_text)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
