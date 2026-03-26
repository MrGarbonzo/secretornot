"""Layer 2: Model-based classifier.

Two backends, controlled by CLASSIFIER_BACKEND env var:

  - "gemma"      (default / POC):  Calls Gemma3-4B via Secret AI endpoint
  - "distilbert" (production):     Local ONNX DistilBERT on CPU, ~20-50ms

The module exposes a single async function `classify()` with a fixed signature.
The rest of the codebase imports only that function and never knows which
backend is running.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx

from config import (
    CLASSIFICATION_TIMEOUT_MS,
    CLASSIFIER_API_KEY,
    CLASSIFIER_BACKEND,
    CLASSIFIER_ENDPOINT,
    CLASSIFIER_MODEL,
    DISTILBERT_MODEL_PATH,
)
from models import Classification

# ═════════════════════════════════════════════════════════════════════════
# DistilBERT ONNX backend — loaded once at module startup if selected
# ═════════════════════════════════════════════════════════════════════════

_onnx_session = None
_onnx_tokenizer = None

if CLASSIFIER_BACKEND == "distilbert":
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer

    _model_path = Path(DISTILBERT_MODEL_PATH)
    _onnx_file = _model_path if _model_path.suffix == ".onnx" else _model_path / "model.onnx"
    _tokenizer_dir = _model_path.parent if _model_path.suffix == ".onnx" else _model_path

    # If tokenizer files aren't next to the ONNX model, fall back to the
    # checkpoint directory or the base model name.
    if (_tokenizer_dir / "tokenizer_config.json").exists():
        _onnx_tokenizer = AutoTokenizer.from_pretrained(str(_tokenizer_dir))
    elif (Path(__file__).parent.parent / "training" / "model_checkpoint" / "tokenizer_config.json").exists():
        _onnx_tokenizer = AutoTokenizer.from_pretrained(
            str(Path(__file__).parent.parent / "training" / "model_checkpoint")
        )
    else:
        _onnx_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    _onnx_session = ort.InferenceSession(str(_onnx_file), providers=["CPUExecutionProvider"])


# ═════════════════════════════════════════════════════════════════════════
# Gemma backend constants
# ═════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are a binary privacy classifier. "
    "Given a user prompt, decide whether it contains or requests PRIVATE information "
    "(PII, credentials, medical, financial, legal, proprietary) or is a PUBLIC general-knowledge question.\n\n"
    "Respond with ONLY a JSON object — no markdown, no explanation:\n"
    '{"classification": "PRIVATE" or "PUBLIC", "confidence": 0.0 to 1.0}\n\n'
    "If uncertain, lean toward PRIVATE."
)


# ═════════════════════════════════════════════════════════════════════════
# Public interface — identical signature regardless of backend
# ═════════════════════════════════════════════════════════════════════════

async def classify(text: str) -> dict:
    """Classify *text* as PRIVATE or PUBLIC.

    Returns ``{"classification": "PRIVATE"|"PUBLIC", "confidence": float}``.

    On any error or timeout → returns PRIVATE with confidence 0.0 (fail-closed).
    """
    if CLASSIFIER_BACKEND == "distilbert":
        return await _classify_distilbert(text)
    return await _classify_gemma(text)


# ═════════════════════════════════════════════════════════════════════════
# DistilBERT ONNX backend
# ═════════════════════════════════════════════════════════════════════════

async def _classify_distilbert(text: str) -> dict:
    """Run DistilBERT ONNX inference on CPU.  ~20-50ms."""
    try:
        # Run in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _distilbert_sync, text)
    except Exception:
        return _fail_private("distilbert inference error")


def _distilbert_sync(text: str) -> dict:
    """Synchronous DistilBERT inference — called from executor."""
    tokens = _onnx_tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=256,
    )

    logits = _onnx_session.run(
        ["logits"],
        {
            "input_ids": tokens["input_ids"].astype(np.int64),
            "attention_mask": tokens["attention_mask"].astype(np.int64),
        },
    )[0]

    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # label 0 = PUBLIC, label 1 = PRIVATE
    pred_class = int(np.argmax(probs, axis=-1)[0])
    confidence = float(probs[0][pred_class])

    if pred_class == 1:
        return {"classification": Classification.PRIVATE, "confidence": confidence}
    return {"classification": Classification.PUBLIC, "confidence": confidence}


# ═════════════════════════════════════════════════════════════════════════
# Gemma backend (POC) — unchanged from original
# ═════════════════════════════════════════════════════════════════════════

async def _classify_gemma(text: str) -> dict:
    """Call Gemma3-4B via Secret AI for classification."""
    timeout_s = CLASSIFICATION_TIMEOUT_MS / 1000.0

    payload = {
        "model": CLASSIFIER_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CLASSIFIER_API_KEY}",
    }

    # Normalise endpoint — Secret AI (Ollama) uses /api/chat, but also
    # accept standard OpenAI /v1/chat/completions if already present.
    base = CLASSIFIER_ENDPOINT.rstrip("/")
    if "/v1/chat/completions" in base or "/api/chat" in base:
        url = base
    else:
        url = f"{base}/api/chat"

    try:
        async with httpx.AsyncClient(verify=False) as client:
            resp = await asyncio.wait_for(
                client.post(url, json=payload, headers=headers, timeout=timeout_s),
                timeout=timeout_s,
            )
            resp.raise_for_status()
            body = resp.json()

            # Ollama format: body.message.content  |  OpenAI format: body.choices[0].message.content
            content: str | None = None
            if "message" in body and "content" in body["message"]:
                content = body["message"]["content"]
            elif "choices" in body and body["choices"]:
                content = body["choices"][0]["message"]["content"]

            if content is None:
                return _fail_private("no content in classifier response")

            return _parse_response(content)

    except (asyncio.TimeoutError, httpx.TimeoutException):
        return _fail_private("classifier timeout")
    except Exception:
        return _fail_private("classifier error")


def _parse_response(raw: str) -> dict:
    """Parse classifier JSON.  Fail toward PRIVATE on any parse error."""
    # Strip markdown fences if the model wraps its output
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last-ditch: look for the word PRIVATE or PUBLIC
        upper = raw.upper()
        if "PRIVATE" in upper:
            return {"classification": Classification.PRIVATE, "confidence": 0.5}
        if "PUBLIC" in upper:
            return {"classification": Classification.PUBLIC, "confidence": 0.5}
        return _fail_private("unparseable classifier output")

    classification = str(data.get("classification", "")).upper()
    confidence = float(data.get("confidence", 0.0))

    if classification == "PUBLIC":
        return {"classification": Classification.PUBLIC, "confidence": confidence}
    # Anything else (PRIVATE, missing, garbage) → PRIVATE
    return {"classification": Classification.PRIVATE, "confidence": confidence}


def _fail_private(reason: str) -> dict:
    """Return a fail-closed PRIVATE result."""
    return {"classification": Classification.PRIVATE, "confidence": 0.0}
