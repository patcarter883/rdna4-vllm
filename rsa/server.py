"""OpenAI-compatible RSA proxy server.

Sits in front of a vLLM server and applies Recursive Self-Aggregation to
chat-completion requests, returning the aggregated answer as a normal
chat completion. Requests with tools, with n>1, or with ``"rsa": false``
in the body are passed through to the backend unchanged.

Example:
    .venv/bin/python -m rsa.server --backend http://localhost:8000/v1 \\
        --port 8100 --rsa-n 16 --rsa-k 4 --rsa-t 2 --rsa-tail-tokens 4096
"""

import argparse
import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from rsa.config import RSAParams, ServerConfig, add_rsa_args, params_from_args
from rsa.core import BackendClient, RSAError, RSAResult, run_rsa

logger = logging.getLogger("rsa.server")

KEEPALIVE_SECONDS = 15.0
STREAM_CHUNK_CHARS = 80


def create_app(config: ServerConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = BackendClient(
            config.backend_base_url,
            api_key=config.api_key,
            timeout=config.defaults.request_timeout,
            tokenizer=config.tokenizer,
        )
        app.state.passthrough = httpx.AsyncClient(
            base_url=config.backend_root, timeout=config.defaults.request_timeout
        )
        app.state.default_model = None
        yield
        await app.state.backend.close()
        await app.state.passthrough.aclose()

    app = FastAPI(title="vLLM RSA proxy", lifespan=lifespan)
    app.state.config = config

    @app.get("/healthz")
    async def healthz():
        try:
            r = await app.state.passthrough.get("/v1/models")
            r.raise_for_status()
            return {"status": "ok", "backend": "up"}
        except httpx.HTTPError as e:
            return JSONResponse(
                status_code=503, content={"status": "degraded", "backend": str(e)}
            )

    @app.get("/v1/models")
    async def models():
        return await _passthrough(app, "GET", "/v1/models")

    @app.post("/tokenize")
    async def tokenize(request: Request):
        return await _passthrough(app, "POST", "/tokenize", await request.body())

    @app.post("/detokenize")
    async def detokenize(request: Request):
        return await _passthrough(app, "POST", "/detokenize", await request.body())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        rsa_value = body.pop("rsa", None)
        try:
            params = (
                None
                if body.get("tools") or body.get("n", 1) > 1
                else _merge(app.state.config.defaults, rsa_value)
            )
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        if params is None:
            return await _passthrough_chat(app, body)
        return await _rsa_chat(app, body, params)

    return app


def _merge(defaults: RSAParams, rsa_value) -> RSAParams | None:
    from rsa.config import merge_params

    return merge_params(defaults, rsa_value)


async def _passthrough(
    app: FastAPI, method: str, path: str, body: bytes | None = None
) -> Response:
    client: httpx.AsyncClient = app.state.passthrough
    r = await client.request(
        method,
        path,
        content=body,
        headers={"Content-Type": "application/json"} if body else None,
    )
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


async def _passthrough_chat(app: FastAPI, body: dict) -> Response:
    """Forward a chat request unchanged (including streaming)."""
    client: httpx.AsyncClient = app.state.passthrough
    if body.get("stream"):

        async def relay():
            async with client.stream("POST", "/v1/chat/completions", json=body) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

        return StreamingResponse(relay(), media_type="text/event-stream")
    r = await client.post("/v1/chat/completions", json=body)
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/json"),
    )


async def _resolve_model(app: FastAPI, body: dict) -> str:
    model = body.get("model")
    if model:
        return model
    if app.state.default_model is None:
        app.state.default_model = await app.state.backend.default_model()
    return app.state.default_model


def _response_dict(result: RSAResult, model: str, params: RSAParams) -> dict:
    return {
        "id": f"rsa-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.final_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "rsa": {
            "rounds": len(result.rounds),
            "rounds_configured": params.t,
            "population": len(result.population),
            "population_configured": params.n,
            "stopped_early": result.stopped_early,
            "aggregation_size": params.k,
            "tail_tokens": params.tail_tokens,
            "selection": result.selection_method,
            "vote": result.vote_detail,
            "requests": result.usage.n_requests,
        },
    }


async def _rsa_chat(app: FastAPI, body: dict, params: RSAParams) -> Response:
    model = await _resolve_model(app, body)
    # Request-level sampling settings override RSA defaults for rollouts.
    if body.get("temperature") is not None:
        params = params.model_copy(update={"temperature": body["temperature"]})
    if body.get("max_tokens") is not None:
        params = params.model_copy(update={"max_tokens": body["max_tokens"]})

    if body.get("stream"):
        return StreamingResponse(
            _stream_rsa(app, body, params, model),
            media_type="text/event-stream",
        )

    try:
        result = await run_rsa(app.state.backend, params, body["messages"], model)
    except RSAError as e:
        return JSONResponse(status_code=502, content={"error": str(e)})
    return JSONResponse(content=_response_dict(result, model, params))


async def _stream_rsa(app: FastAPI, body: dict, params: RSAParams, model: str):
    """Pseudo-stream: run RSA fully (with SSE keep-alives), then chunk out."""
    task = asyncio.create_task(
        run_rsa(app.state.backend, params, body["messages"], model)
    )
    while True:
        done, _ = await asyncio.wait({task}, timeout=KEEPALIVE_SECONDS)
        if done:
            break
        yield b": keepalive\n\n"

    try:
        result = task.result()
    except RSAError as e:
        yield _sse({"error": str(e)})
        yield b"data: [DONE]\n\n"
        return

    completion_id = f"rsa-{uuid.uuid4().hex}"

    def chunk(delta: dict, finish_reason: str | None = None, usage=None) -> bytes:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return _sse(payload)

    yield chunk({"role": "assistant", "content": ""})
    text = result.final_text
    for i in range(0, len(text), STREAM_CHUNK_CHARS):
        yield chunk({"content": text[i : i + STREAM_CHUNK_CHARS]})
    usage = None
    if (body.get("stream_options") or {}).get("include_usage"):
        usage = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        }
    yield chunk({}, finish_reason="stop", usage=usage)
    yield b"data: [DONE]\n\n"


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", default="http://localhost:8000/v1", help="vLLM base URL"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "HF tokenizer name/path for local token-exact tails "
            "(default: resolve the backend model's HF repo automatically)"
        ),
    )
    add_rsa_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(name)s %(message)s"
    )
    config = ServerConfig(
        backend_base_url=args.backend,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        log_level=args.log_level,
        tokenizer=args.tokenizer,
        defaults=params_from_args(args),
    )
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
