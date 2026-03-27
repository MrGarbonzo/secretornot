"""Routing decision engine.

Takes the outputs from Layer 1 (rule filter) and Layer 2 (model classifier),
applies the default-to-private policy, and returns a final routing destination.
"""

from __future__ import annotations

from classifier.rule_filter import check as rule_check
from classifier.model_classifier import classify as model_classify
from config import CONFIDENCE_THRESHOLD
from models import (
    ChatMessage,
    Classification,
    ClassificationResult,
    Destination,
    LayerTriggered,
)


async def decide(text: str) -> ClassificationResult:
    """Run both classification layers and return the routing decision.

    Layer 1 (rules) is synchronous and runs first.  If it matches,
    Layer 2 is never called.
    """

    # ── Layer 1: rule-based pre-filter ───────────────────────────────────
    rule_result = rule_check(text)

    if rule_result.matched:
        return ClassificationResult(
            classification=Classification.PRIVATE,
            confidence=1.0,
            layer=LayerTriggered.RULE,
            rule_matched=rule_result.rule_name,
        )

    # ── Layer 2: model classifier ────────────────────────────────────────
    model_result = await model_classify(text)
    classification = model_result["classification"]
    confidence = model_result["confidence"]

    # Apply confidence threshold — low confidence → PRIVATE
    if classification == Classification.PUBLIC and confidence < CONFIDENCE_THRESHOLD:
        return ClassificationResult(
            classification=Classification.PRIVATE,
            confidence=confidence,
            layer=LayerTriggered.CLASSIFIER,
        )

    return ClassificationResult(
        classification=classification,
        confidence=confidence,
        layer=LayerTriggered.CLASSIFIER,
    )


async def decide_conversation(messages: list[ChatMessage]) -> ClassificationResult:
    """Classify a full conversation by examining each message individually.

    Layer 1 (rules) runs on every message first — if any match, return PRIVATE.
    Layer 2 (model) runs on each user message, newest-first, short-circuiting
    on the first PRIVATE result.
    """

    # ── Layer 1: rules on every message (~0.1ms each) ─────────────────────
    for m in messages:
        if not m.content or not isinstance(m.content, str):
            continue
        rule_result = rule_check(m.content)
        if rule_result.matched:
            return ClassificationResult(
                classification=Classification.PRIVATE,
                confidence=1.0,
                layer=LayerTriggered.RULE,
                rule_matched=rule_result.rule_name,
            )

    # ── Layer 2: DistilBERT on user messages, newest-first ────────────────
    user_messages = [
        m for m in messages
        if m.role == "user" and m.content and isinstance(m.content, str)
    ]

    if not user_messages:
        return ClassificationResult(
            classification=Classification.PRIVATE,
            confidence=0.0,
            layer=LayerTriggered.DEFAULT,
        )

    min_confidence = 1.0
    for m in reversed(user_messages):
        model_result = await model_classify(m.content)
        classification = model_result["classification"]
        confidence = model_result["confidence"]

        if classification == Classification.PRIVATE:
            return ClassificationResult(
                classification=Classification.PRIVATE,
                confidence=confidence,
                layer=LayerTriggered.CLASSIFIER,
            )

        if confidence < CONFIDENCE_THRESHOLD:
            return ClassificationResult(
                classification=Classification.PRIVATE,
                confidence=confidence,
                layer=LayerTriggered.CLASSIFIER,
            )

        min_confidence = min(min_confidence, confidence)

    return ClassificationResult(
        classification=Classification.PUBLIC,
        confidence=min_confidence,
        layer=LayerTriggered.CLASSIFIER,
    )


def resolve_destination(result: ClassificationResult) -> Destination:
    """Map a classification result to a concrete destination."""
    if result.classification == Classification.PRIVATE:
        return Destination.SECRET_AI
    return Destination.PUBLIC_LLM
