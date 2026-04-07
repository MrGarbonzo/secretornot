"""SecretOrNot — Privacy Router.

Single-endpoint FastAPI service that classifies prompts as PRIVATE or PUBLIC
and routes them to the appropriate LLM backend.  Fully OpenAI-compatible.
"""

from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from models import ChatCompletionRequest, ChatMessage
from router.decision import decide, decide_conversation, resolve_destination
from router.proxy import forward
from audit.logger import log_decision
from config import AUDIT_LOG_FILE, SECRET_AI_VM_URL, SELF_VM_URL

# ── Attestation cache ───────────────────────────────────────────────
_attestation_cache: dict | None = None


def _get_docker_host_ip() -> str | None:
    """Get the Docker host IP by reading the default gateway from /proc/net/route."""
    import struct
    try:
        with open("/proc/net/route") as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] == "00000000":  # default route
                    return ".".join(str(b) for b in struct.pack("<I", int(fields[2], 16)))
        return None
    except Exception:
        return None


def _find_attestation_host(port: int = 29343) -> str | None:
    """Find a reachable attestation service, trying multiple host candidates."""
    import socket
    candidates = []
    # If running in Docker, try the host gateway
    gw = _get_docker_host_ip()
    if gw:
        candidates.append(gw)
    candidates.extend(["host.docker.internal", "localhost", "127.0.0.1"])
    for host in candidates:
        try:
            with socket.create_connection((host, port), timeout=3):
                return host
        except OSError:
            continue
    return None


def _discover_vm_hostname(host: str, port: int = 29343) -> str | None:
    """Read the TLS cert CN from the attestation service to get the real hostname."""
    import ssl
    try:
        pem = ssl.get_server_certificate((host, port))
        from cryptography.x509 import load_pem_x509_certificate
        from cryptography.x509.oid import NameOID
        cert = load_pem_x509_certificate(pem.encode("ascii"))
        cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
        return cn[0].value if cn else None
    except Exception:
        return None


def _run_attestation() -> dict:
    """Run secretvm-verify against both VMs. Called once and cached."""
    from dataclasses import asdict
    from secretvm.verify import check_secret_vm

    results = {}

    # Verify the Secret AI machine (where private prompts go)
    try:
        ai_result = check_secret_vm(SECRET_AI_VM_URL)
        results["secret_ai"] = asdict(ai_result)
    except Exception as e:
        results["secret_ai"] = {"valid": False, "error": str(e)}

    # Verify the SecretOrNot router VM itself (self-attestation)
    # Auto-discover: find the attestation service on the host, read the TLS
    # cert to get the real VM hostname, then verify using that hostname.
    # This works without any configuration — no need to know the VM URL.
    if SELF_VM_URL:
        if SELF_VM_URL == "auto":
            probe_host = _find_attestation_host()
        else:
            from secretvm.verify import _parse_vm_url
            probe_host, _ = _parse_vm_url(SELF_VM_URL)
            # check reachability
            import socket
            try:
                socket.create_connection((probe_host, 29343), timeout=3).close()
            except OSError:
                probe_host = None

        if probe_host:
            real_hostname = _discover_vm_hostname(probe_host, 29343)
            verify_url = f"https://{real_hostname}" if real_hostname else f"https://{probe_host}"
            try:
                self_result = check_secret_vm(verify_url)
                results["router"] = asdict(self_result)
                # Run code provenance on the router's docker-compose
                results["router"]["provenance"] = _resolve_provenance(verify_url)
            except Exception as e:
                results["router"] = {"valid": False, "error": str(e)}

    return results


def _resolve_provenance(vm_url: str) -> list[dict]:
    """Fetch the docker-compose from a VM and resolve each image to its source commit."""
    import html
    import re
    import requests
    from secretvm.verify import _parse_vm_url
    from code_provenance.compose_parser import parse_compose, parse_image_ref
    from code_provenance.resolver import resolve_image
    from dataclasses import asdict

    host, port = _parse_vm_url(vm_url)
    try:
        resp = requests.get(f"https://{host}:{port}/docker-compose", timeout=15, verify=True)
        resp.raise_for_status()
        # Extract YAML from HTML wrapper
        text = resp.text.strip()
        m = re.search(r"<pre>(.*?)</pre>", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1)
        text = html.unescape(text)
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    except Exception:
        return []

    results = []
    for service, image_str in parse_compose(text):
        ref = parse_image_ref(image_str)
        result = resolve_image(service, ref)
        results.append({
            "service": result.service,
            "image": result.image,
            "repo": result.repo,
            "commit": result.commit,
            "commit_url": result.commit_url,
            "status": result.status,
            "confidence": result.confidence,
            "method": result.resolution_method,
        })
    return results


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

    # ── Classify ─────────────────────────────────────────────────────────
    classify_start = time.perf_counter()
    result = await decide_conversation(parsed.messages)
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
    messages = body.get("messages")
    text = body.get("text", "")

    if messages:
        parsed_messages = [ChatMessage(**m) for m in messages]
        classify_start = time.perf_counter()
        result = await decide_conversation(parsed_messages)
    elif text.strip():
        classify_start = time.perf_counter()
        result = await decide(text)
    else:
        return {"classification": "EMPTY", "source": "none", "confidence": 0}

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


FEEDBACK_FILE = AUDIT_LOG_FILE.replace("audit", "feedback") if "audit" in AUDIT_LOG_FILE else "feedback.jsonl"


@app.post("/feedback")
async def feedback(request: Request):
    """Save correct/incorrect feedback for retraining."""
    body = await request.json()
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "text": body.get("text", ""),
        "model_said": body.get("model_said", ""),
        "correct_label": body.get("correct_label", ""),
        "label": 1 if body.get("correct_label") == "PRIVATE" else 0,
    }
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"ok": True}


@app.get("/health")
async def health():
    return {"ok": True, "service": "SecretOrNot", "version": "0.1.0"}


@app.get("/attestation")
async def attestation():
    """Return cached TEE attestation results for both VMs."""
    global _attestation_cache
    if _attestation_cache is None:
        import asyncio
        _attestation_cache = await asyncio.to_thread(_run_attestation)
    return _attestation_cache


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
  header { padding: 16px 24px; border-bottom: 1px solid #222; display: flex; align-items: center; justify-content: space-between; }
  header .left { display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 20px; font-weight: 600; }
  header h1 span { color: #666; font-weight: 400; }
  .tabs { display: flex; gap: 4px; }
  .tab { background: none; border: 1px solid #333; color: #888; border-radius: 8px; padding: 6px 14px; font-size: 13px; font-weight: 500; cursor: pointer; }
  .tab:hover { color: #ccc; border-color: #555; background: none; }
  .tab.active { background: #1a1a2e; color: #e0e0e0; border-color: #2a2a4a; }
  .header-btn { background: none; border: 1px solid #333; color: #888; border-radius: 8px; padding: 6px 14px; font-size: 13px; }
  .header-btn:hover { color: #ccc; border-color: #555; background: none; }
  .panel { flex: 1; display: none; flex-direction: column; overflow: hidden; }
  .panel.active { display: flex; }
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
  textarea { flex: 1; background: #1a1a1a; border: 1px solid #333; border-radius: 10px; padding: 12px 16px; color: #e0e0e0; font-size: 15px; font-family: inherit; resize: none; outline: none; min-height: 44px; max-height: 200px; }
  textarea:focus { border-color: #555; }
  textarea::placeholder { color: #555; }
  button { background: #fff; color: #000; border: none; border-radius: 10px; padding: 12px 20px; font-size: 14px; font-weight: 600; cursor: pointer; white-space: nowrap; }
  button:hover { background: #ddd; }
  button:disabled { background: #333; color: #666; cursor: not-allowed; }
  /* Classifier test panel */
  #classify-panel { padding: 24px; overflow-y: auto; }
  #classify-inner { max-width: 720px; margin: 0 auto; display: flex; flex-direction: column; gap: 16px; }
  #classify-input { min-height: 100px; }
  #classify-results { display: flex; flex-direction: column; gap: 12px; }
  .classify-row { background: #111; border: 1px solid #2a2a2a; border-radius: 12px; padding: 14px 18px; }
  .classify-row .text { font-size: 14px; color: #aaa; margin-bottom: 8px; white-space: pre-wrap; }
  .classify-row .result { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; font-size: 13px; }
  .feedback-btns { display: flex; gap: 6px; margin-left: auto; }
  .fb-btn { padding: 4px 12px; font-size: 12px; border-radius: 6px; font-weight: 600; }
  .fb-btn.correct { background: #052e16; color: #4ade80; border: 1px solid #14532d; }
  .fb-btn.correct:hover { background: #0a3d1f; }
  .fb-btn.wrong { background: #3b1219; color: #f87171; border: 1px solid #7f1d1d; }
  .fb-btn.wrong:hover { background: #4a1820; }
  .fb-btn.sent { opacity: 0.5; cursor: default; }
  .fb-btn.sent:hover { background: inherit; }
  /* Attestation banner */
  #attestation-banner { max-width: 720px; width: 100%; margin: 0 auto; }
  .attest-card { background: #111; border: 1px solid #2a2a2a; border-radius: 12px; margin-bottom: 12px; overflow: hidden; }
  .attest-header { display: flex; align-items: center; gap: 10px; padding: 12px 16px; cursor: pointer; user-select: none; }
  .attest-header:hover { background: #1a1a1a; }
  .attest-status { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
  .attest-status.pass { background: #4ade80; box-shadow: 0 0 6px #4ade8066; }
  .attest-status.fail { background: #f87171; box-shadow: 0 0 6px #f8717166; }
  .attest-status.loading { background: #facc15; animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  .attest-title { font-size: 13px; font-weight: 600; flex: 1; }
  .attest-summary { font-size: 12px; color: #888; }
  .attest-chevron { color: #555; font-size: 12px; transition: transform 0.2s; }
  .attest-card.open .attest-chevron { transform: rotate(90deg); }
  .attest-details { display: none; padding: 0 16px 12px; font-size: 12px; color: #aaa; line-height: 1.8; }
  .attest-card.open .attest-details { display: block; }
  .attest-check { display: flex; gap: 8px; align-items: center; }
  .attest-check .icon { font-size: 13px; }
  .attest-section-label { color: #666; font-weight: 600; margin-top: 6px; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
  .attest-prov { display: flex; flex-direction: column; gap: 4px; margin-top: 2px; }
  .attest-prov-row { display: flex; gap: 8px; align-items: baseline; font-size: 12px; }
  .attest-prov-svc { color: #888; min-width: 90px; }
  .attest-prov-link { color: #60a5fa; text-decoration: none; word-break: break-all; }
  .attest-prov-link:hover { text-decoration: underline; }
  .attest-prov-conf { font-size: 11px; padding: 1px 6px; border-radius: 4px; }
  .attest-prov-conf.exact { background: #052e16; color: #4ade80; }
  .attest-prov-conf.approximate { background: #1e1b4b; color: #a5b4fc; }
</style>
</head>
<body>
<header>
  <div class="left">
    <h1>SecretOrNot <span>— privacy router</span></h1>
    <div class="tabs">
      <button class="tab active" onclick="switchTab('chat')">Chat</button>
      <button class="tab" onclick="switchTab('classify')">Classifier</button>
    </div>
  </div>
  <button class="header-btn" onclick="newChat()">New Chat</button>
</header>

<!-- Chat panel -->
<div id="chat-panel" class="panel active">
  <div id="chat">
    <div id="attestation-banner"></div>
  </div>
  <div id="input-area">
    <div id="input-wrap">
      <textarea id="prompt" rows="1" placeholder="Type a message..." autofocus></textarea>
      <button id="send" onclick="send()">Send</button>
    </div>
  </div>
</div>

<!-- Classifier test panel -->
<div id="classify-panel" class="panel">
  <div id="classify-inner">
    <div style="color:#888; font-size:13px;">Test DistilBERT classifier directly — single input, no conversation context. Mark results correct or incorrect to save training feedback.</div>
    <textarea id="classify-input" placeholder="Enter text to classify..." rows="3"></textarea>
    <button onclick="classifyTest()">Classify</button>
    <div id="classify-results"></div>
  </div>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('prompt');
const btn = document.getElementById('send');
let history = [];

function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  if (tab === 'chat') {
    document.querySelector('.tab:nth-child(1)').classList.add('active');
    document.getElementById('chat-panel').classList.add('active');
  } else {
    document.querySelector('.tab:nth-child(2)').classList.add('active');
    document.getElementById('classify-panel').classList.add('active');
  }
}

function newChat() {
  history = [];
  // Preserve the attestation banner, clear everything else
  const banner = document.getElementById('attestation-banner');
  chat.innerHTML = '';
  chat.appendChild(banner);
}

input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
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

  const turn = document.createElement('div');
  turn.className = 'turn';
  turn.innerHTML = `<div class="user-msg">${esc(text)}</div>`
    + `<div class="routing-bar thinking">classifying...</div>`
    + `<div class="llm-response thinking">waiting for LLM...</div>`;
  chat.appendChild(turn);
  chat.scrollTop = chat.scrollHeight;

  const routingBar = turn.querySelector('.routing-bar');
  const responseDiv = turn.querySelector('.llm-response');

  try {
    const classRes = await fetch('/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, messages: history})
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

  try {
    responseDiv.textContent = '';
    responseDiv.className = 'llm-response';

    const llmRes = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ model: "auto", messages: history, stream: false })
    });

    if (!llmRes.ok) {
      const err = await llmRes.text();
      responseDiv.className = 'llm-response error';
      responseDiv.textContent = `Error ${llmRes.status}: ${err}`;
    } else {
      const data = await llmRes.json();
      const content = data.choices && data.choices[0] && data.choices[0].message
        ? data.choices[0].message.content : JSON.stringify(data);
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

// ── Classifier test panel ──────────────────────────────────────────
const classifyInput = document.getElementById('classify-input');
const classifyResults = document.getElementById('classify-results');

classifyInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); classifyTest(); }
});

async function classifyTest() {
  const text = classifyInput.value.trim();
  if (!text) return;
  classifyInput.value = '';

  const row = document.createElement('div');
  row.className = 'classify-row';
  row.innerHTML = `<div class="text">${esc(text)}</div><div class="result thinking">classifying...</div>`;
  classifyResults.prepend(row);

  try {
    const res = await fetch('/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    const c = await res.json();
    const resultDiv = row.querySelector('.result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = `<span class="badge ${c.classification}">${c.classification}</span>`
      + `<span>${c.source}${c.rule_matched ? ': ' + c.rule_matched : ''}</span>`
      + `<span>conf: ${c.confidence}</span>`
      + `<span>${c.latency_ms}ms</span>`
      + `<div class="feedback-btns">`
      + `<button class="fb-btn correct" onclick="sendFeedback(this,'${esc(text).replace(/'/g,"\\'")}','${c.classification}','${c.classification}')">Correct</button>`
      + `<button class="fb-btn wrong" onclick="sendFeedback(this,'${esc(text).replace(/'/g,"\\'")}','${c.classification}','${c.classification==="PRIVATE"?"PUBLIC":"PRIVATE"}')">Wrong → ${c.classification==='PRIVATE'?'PUBLIC':'PRIVATE'}</button>`
      + `</div>`;
  } catch(e) {
    row.querySelector('.result').textContent = 'error: ' + e.message;
  }
}

async function sendFeedback(el, text, modelSaid, correctLabel) {
  if (el.classList.contains('sent')) return;
  const btns = el.parentElement;
  btns.querySelectorAll('.fb-btn').forEach(b => b.classList.add('sent'));
  el.textContent += ' (saved)';
  await fetch('/feedback', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({text, model_said: modelSaid, correct_label: correctLabel})
  });
}

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ── Attestation ──────────────────────────────────────────────────
const attestBanner = document.getElementById('attestation-banner');

function renderAttestCard(id, label, data) {
  if (!data) return '';
  const ok = data.valid === true;
  const status = ok ? 'pass' : 'fail';
  const checks = data.checks || {};
  const report = data.report || {};
  const errors = data.errors || [];
  const cpuType = report.cpu_type || data.attestation_type || '?';

  let summaryParts = [];
  if (ok) summaryParts.push('Verified');
  else summaryParts.push('Failed');
  if (cpuType && cpuType !== 'SECRET-VM') summaryParts.push(cpuType);
  if (checks.gpu_attestation_valid) summaryParts.push('+ GPU');
  const workload = report.workload;
  if (workload && workload.status === 'authentic_match') {
    summaryParts.push('v' + (workload.artifacts_ver || '?'));
  }

  let detailsHtml = '';

  // CPU section
  detailsHtml += '<div class="attest-section-label">CPU Attestation</div>';
  detailsHtml += checkLine(checks.cpu_quote_fetched, 'Quote fetched');
  detailsHtml += checkLine(checks.cpu_attestation_valid, 'Signature & cert chain valid (' + esc(cpuType) + ')');
  detailsHtml += checkLine(checks.tls_binding, 'TLS certificate binding');

  // GPU section
  if (checks.gpu_quote_fetched !== undefined) {
    detailsHtml += '<div class="attest-section-label">GPU Attestation</div>';
    if (checks.gpu_quote_fetched === false && !checks.gpu_attestation_valid) {
      detailsHtml += checkLine(null, 'No GPU on this machine');
    } else {
      detailsHtml += checkLine(checks.gpu_attestation_valid, 'NVIDIA attestation (JWT verified via NRAS)');
      detailsHtml += checkLine(checks.gpu_binding, 'GPU nonce binding to CPU quote');
    }
  }

  // Workload section
  if (workload) {
    detailsHtml += '<div class="attest-section-label">Workload</div>';
    detailsHtml += checkLine(checks.workload_verified, 'Authentic SecretVM image');
    if (workload.template_name) detailsHtml += checkLine(null, 'Template: ' + esc(workload.template_name));
    if (workload.artifacts_ver) detailsHtml += checkLine(null, 'Version: ' + esc(workload.artifacts_ver));
    if (workload.env) detailsHtml += checkLine(null, 'Env: ' + esc(workload.env));
  }

  // Code Provenance
  const prov = data.provenance || [];
  if (prov.length > 0) {
    detailsHtml += '<div class="attest-section-label">Code Provenance</div>';
    detailsHtml += '<div class="attest-prov">';
    prov.forEach(p => {
      let link = '';
      if (p.commit_url && p.commit) {
        link = '<a class="attest-prov-link" href="' + esc(p.commit_url) + '" target="_blank">' + esc(p.commit.substring(0, 12)) + '</a>';
      } else if (p.repo) {
        link = '<a class="attest-prov-link" href="' + esc(p.repo) + '" target="_blank">' + esc(p.repo.replace('https://github.com/', '')) + '</a>';
      } else {
        link = '<span style="color:#666;">unresolved</span>';
      }
      let conf = '';
      if (p.confidence) {
        conf = ' <span class="attest-prov-conf ' + esc(p.confidence) + '">' + esc(p.confidence) + '</span>';
      }
      detailsHtml += '<div class="attest-prov-row">'
        + '<span class="attest-prov-svc">' + esc(p.service) + '</span>'
        + link + conf
        + '</div>';
    });
    detailsHtml += '</div>';
  }

  // Errors
  if (errors.length > 0) {
    detailsHtml += '<div class="attest-section-label">Errors</div>';
    errors.forEach(e => { detailsHtml += '<div class="attest-check" style="color:#f87171;">' + esc(e) + '</div>'; });
  }

  return '<div class="attest-card" id="' + id + '">'
    + '<div class="attest-header" onclick="this.parentElement.classList.toggle(&quot;open&quot;)">'
    + '<div class="attest-status ' + status + '"></div>'
    + '<div class="attest-title">' + esc(label) + '</div>'
    + '<div class="attest-summary">' + esc(summaryParts.join(' / ')) + '</div>'
    + '<div class="attest-chevron">&#9654;</div>'
    + '</div>'
    + '<div class="attest-details">' + detailsHtml + '</div>'
    + '</div>';
}

function checkLine(val, text) {
  if (val === true) return '<div class="attest-check"><span class="icon" style="color:#4ade80;">&#10003;</span> ' + text + '</div>';
  if (val === false) return '<div class="attest-check"><span class="icon" style="color:#f87171;">&#10007;</span> ' + text + '</div>';
  return '<div class="attest-check"><span class="icon" style="color:#555;">&#8226;</span> ' + text + '</div>';
}

(async function loadAttestation() {
  attestBanner.innerHTML =
    '<div class="attest-card"><div class="attest-header">'
    + '<div class="attest-status loading"></div>'
    + '<div class="attest-title">Verifying TEE attestation...</div>'
    + '<div class="attest-summary">contacting Intel PCS, NVIDIA NRAS...</div>'
    + '</div></div>';
  try {
    const res = await fetch('/attestation');
    const data = await res.json();
    let html = '';
    if (data.secret_ai) html += renderAttestCard('attest-ai', 'Secret AI Machine', data.secret_ai);
    if (data.router) html += renderAttestCard('attest-router', 'SecretOrNot Router (this machine)', data.router);
    attestBanner.innerHTML = html || '<div style="color:#666;font-size:12px;padding:8px 0;">No attestation data available.</div>';
  } catch(e) {
    attestBanner.innerHTML = '<div style="color:#f87171;font-size:12px;padding:8px 0;">Attestation check failed: ' + esc(e.message) + '</div>';
  }
})();
</script>
</body>
</html>"""
