"""Forward requests to the chosen LLM endpoint.

Handles both streaming (SSE passthrough) and non-streaming responses.
The client receives an OpenAI-identical response — routing is invisible.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    PUBLIC_LLM_API_KEY,
    PUBLIC_LLM_ENDPOINT,
    PROXY_TIMEOUT_S,
    SECRET_AI_API_KEY,
    SECRET_AI_ENDPOINT,
)
from models import Destination


def _endpoint_for(destination: Destination) -> tuple[str, str]:
    """Return (base_url, api_key) for the given destination."""
    if destination == Destination.SECRET_AI:
        return SECRET_AI_ENDPOINT.rstrip("/"), SECRET_AI_API_KEY
    return PUBLIC_LLM_ENDPOINT.rstrip("/"), PUBLIC_LLM_API_KEY


async def forward(
    destination: Destination,
    body: dict,
    is_stream: bool,
) -> JSONResponse | StreamingResponse:
    """Proxy the request to *destination* and return the response."""
    base_url, api_key = _endpoint_for(destination)
    url = f"{base_url}/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    if is_stream:
        return await _forward_stream(url, headers, body)
    return await _forward_sync(url, headers, body)


async def _forward_sync(url: str, headers: dict, body: dict) -> JSONResponse:
    """Non-streaming: POST, wait for full response, return as JSON."""
    async with httpx.AsyncClient(verify=False, timeout=PROXY_TIMEOUT_S) as client:
        resp = await client.post(url, json=body, headers=headers)
        # Pass through status and body exactly as-is
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _forward_stream(url: str, headers: dict, body: dict) -> StreamingResponse:
    """Streaming: POST with stream=True, SSE passthrough to client."""

    async def event_generator() -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(verify=False, timeout=PROXY_TIMEOUT_S) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
