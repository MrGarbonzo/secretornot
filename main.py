"""SecretOrNot — Privacy Router.

Single-endpoint FastAPI service that classifies prompts as PRIVATE or PUBLIC
and routes them to the appropriate LLM backend.  Fully OpenAI-compatible.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from models import ChatCompletionRequest
from router.decision import decide, resolve_destination
from router.proxy import forward
from audit.logger import log_decision

app = FastAPI(
    title="SecretOrNot",
    description="Privacy router — route what needs to be private, prove it was.",
    version="0.1.0",
)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint with privacy routing."""
    total_start = time.perf_counter()

    # Parse request — pass through everything, only validate what we need
    raw_body = await request.json()
    try:
        parsed = ChatCompletionRequest(**raw_body)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid request: {exc}", "type": "invalid_request_error"}},
        )

    request_id = str(uuid.uuid4())

    # Build classification text from all messages
    classify_text = "\n".join(
        f"{m.role}: {m.content}" for m in parsed.messages if m.content
    )

    # ── Classify ─────────────────────────────────────────────────────────
    classify_start = time.perf_counter()
    result = await decide(classify_text)
    classification_ms = (time.perf_counter() - classify_start) * 1000

    destination = resolve_destination(result)

    # ── Proxy ────────────────────────────────────────────────────────────
    inference_start = time.perf_counter()
    response = await forward(destination, raw_body, is_stream=parsed.stream)
    inference_ms = (time.perf_counter() - inference_start) * 1000

    total_ms = (time.perf_counter() - total_start) * 1000

    # ── Audit ────────────────────────────────────────────────────────────
    log_decision(
        request_id=request_id,
        result=result,
        destination=destination,
        model_requested=parsed.model,
        classification_ms=classification_ms,
        inference_ms=inference_ms,
        total_ms=total_ms,
    )

    return response


@app.get("/health")
async def health():
    return {"ok": True, "service": "SecretOrNot", "version": "0.1.0"}
