"""Layer 2: Model-based classifier.

Fine-tuned DistilBERT running locally via ONNX on CPU.  ~20-50ms per
classification.  The module exposes a single async function `classify()`
that the rest of the codebase imports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from config import DISTILBERT_MODEL_PATH
from models import Classification

# ═════════════════════════════════════════════════════════════════════════
# Load ONNX model + tokenizer once at startup
# ═════════════════════════════════════════════════════════════════════════

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
# Public interface
# ═════════════════════════════════════════════════════════════════════════

async def classify(text: str) -> dict:
    """Classify *text* as PRIVATE or PUBLIC.

    Returns ``{"classification": "PRIVATE"|"PUBLIC", "confidence": float}``.

    On any error → returns PRIVATE with confidence 0.0 (fail-closed).
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _distilbert_sync, text)
    except Exception:
        return {"classification": Classification.PRIVATE, "confidence": 0.0}


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
