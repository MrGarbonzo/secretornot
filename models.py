from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Classification(str, Enum):
    PRIVATE = "PRIVATE"
    PUBLIC = "PUBLIC"


class Destination(str, Enum):
    SECRET_AI = "secret_ai"
    PUBLIC_LLM = "public_llm"


class LayerTriggered(str, Enum):
    RULE = "rule"
    CLASSIFIER = "classifier"
    DEFAULT = "default"


class ClassificationResult(BaseModel):
    classification: Classification
    confidence: float = 1.0
    layer: LayerTriggered = LayerTriggered.DEFAULT
    rule_matched: str | None = None


class LatencyMs(BaseModel):
    classification: float = 0.0
    inference: float = 0.0
    total: float = 0.0


class AuditEntry(BaseModel):
    request_id: str
    timestamp: str
    classification: Classification
    layer_triggered: LayerTriggered
    rule_matched: str | None = None
    confidence: float = 1.0
    routed_to: Destination
    model_requested: str | None = None
    latency_ms: LatencyMs


# OpenAI-compatible request — we only validate what we need, pass everything
# else through untouched so we stay transparent.
class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: str | None = None
    tool_calls: Any = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    # Pass through everything else
    model_config = {"extra": "allow"}
