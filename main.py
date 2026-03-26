"""SecretOrNot — Privacy Router.

Single-endpoint FastAPI service that classifies prompts as PRIVATE or PUBLIC
and routes them to the appropriate LLM backend.  Fully OpenAI-compatible.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

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


@app.post("/classify")
async def classify_only(request: Request):
    """Classify a prompt without routing — for the test UI."""
    body = await request.json()
    text = body.get("text", "")
    if not text.strip():
        return {"classification": "EMPTY", "source": "none", "confidence": 0}

    classify_start = time.perf_counter()
    result = await decide(text)
    classification_ms = (time.perf_counter() - classify_start) * 1000
    destination = resolve_destination(result)

    return {
        "classification": result.classification.value,
        "source": result.layer.value,
        "rule_matched": result.rule_matched,
        "confidence": round(result.confidence, 4),
        "routed_to": destination.value,
        "latency_ms": round(classification_ms, 1),
    }


@app.get("/health")
async def health():
    return {"ok": True, "service": "SecretOrNot", "version": "0.1.0"}


@app.get("/", response_class=HTMLResponse)
async def test_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SecretOrNot</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0a0a; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  header { padding: 20px 24px; border-bottom: 1px solid #222; }
  header h1 { font-size: 20px; font-weight: 600; }
  header h1 span { color: #666; font-weight: 400; }
  #chat { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 20px; }
  .turn { max-width: 720px; width: 100%; margin: 0 auto; }
  .user-msg { background: #1a1a2e; border: 1px solid #2a2a4a; border-radius: 12px; padding: 14px 18px; font-size: 15px; line-height: 1.5; white-space: pre-wrap; }
  .routing-bar { margin-top: 8px; font-size: 12px; color: #666; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 100px; font-weight: 600; font-size: 12px; letter-spacing: 0.5px; }
  .badge.PRIVATE { background: #3b1219; color: #f87171; border: 1px solid #7f1d1d; }
  .badge.PUBLIC { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
  .route { color: #888; }
  .route.secret_ai { color: #f87171; }
  .route.public_llm { color: #4ade80; }
  .llm-response { margin-top: 12px; background: #111; border: 1px solid #2a2a2a; border-radius: 12px; padding: 14px 18px; font-size: 15px; line-height: 1.6; white-space: pre-wrap; }
  .llm-response.error { border-color: #7f1d1d; color: #f87171; }
  .thinking { color: #666; font-style: italic; font-size: 13px; padding: 8px 0; }
  #input-area { padding: 16px 24px; border-top: 1px solid #222; background: #0f0f0f; }
  #input-wrap { max-width: 720px; margin: 0 auto; display: flex; gap: 10px; }
  #prompt { flex: 1; background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 12px 16px; color: #e0e0e0; font-size: 15px; font-family: inherit; resize: none; outline: none; min-height: 44px; max-height: 200px; }
  #prompt:focus { border-color: #555; }
  #prompt::placeholder { color: #555; }
  button { background: #fff; color: #000; border: none; border-radius: 10px; padding: 12px 20px; font-size: 14px; font-weight: 600; cursor: pointer; white-space: nowrap; }
  button:hover { background: #ddd; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>SecretOrNot <span>— privacy router</span></h1>
</header>
<div id="chat"></div>
<div id="input-area">
  <div id="input-wrap">
    <textarea id="prompt" rows="1" placeholder="Type a message..." autofocus></textarea>
    <button id="send" onclick="send()">Send</button>
  </div>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('prompt');
const btn = document.getElementById('send');
let history = [];

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 200) + 'px';
});

async function send() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';
  btn.disabled = true;

  history.push({role: 'user', content: text});

  // Create turn container
  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `<div class="user-msg">${esc(text)}</div>`
    + `<div class="routing-bar thinking">classifying...</div>`
    + `<div class="llm-response thinking">waiting for LLM...</div>`;
  chat.appendChild(turn);
  chat.scrollTop = chat.scrollHeight;

  const routingBar = turn.querySelector('.routing-bar');
  const responseDiv = turn.querySelector('.llm-response');

  // Step 1: Classify
  try {
    const classRes = await fetch('/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const c = classRes.ok ? await classRes.json() : null;
    if (c) {
      routingBar.className = 'routing-bar';
      routingBar.innerHTML = `<span class="badge ${c.classification}">${c.classification}</span>`
        + `<span class="route ${c.routed_to}">&rarr; ${c.routed_to === 'secret_ai' ? 'Secret AI' : 'Public LLM'}</span>`
        + `<span>${c.source}${c.rule_matched ? ': ' + c.rule_matched : ''}</span>`
        + `<span>conf: ${c.confidence}</span>`
        + `<span>${c.latency_ms}ms</span>`;
    }
  } catch(e) {
    routingBar.className = 'routing-bar';
    routingBar.textContent = 'classification failed: ' + e.message;
  }

  chat.scrollTop = chat.scrollHeight;

  // Step 2: Get LLM response via the real router endpoint
  try {
    responseDiv.textContent = '';
    responseDiv.className = 'llm-response';

    const llmRes = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        model: "auto",
        messages: history,
        stream: false
      })
    });

    if (!llmRes.ok) {
      const err = await llmRes.text();
      responseDiv.className = 'llm-response error';
      responseDiv.textContent = `Error ${llmRes.status}: ${err}`;
    } else {
      const data = await llmRes.json();
      const content = data.choices && data.choices[0] && data.choices[0].message
        ? data.choices[0].message.content
        : JSON.stringify(data);
      responseDiv.textContent = content;
      history.push({role: 'assistant', content});
    }
  } catch(e) {
    responseDiv.className = 'llm-response error';
    responseDiv.textContent = 'Request failed: ' + e.message;
  }

  btn.disabled = false;
  input.focus();
  chat.scrollTop = chat.scrollHeight;
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
</script>
</body>
</html>"""
