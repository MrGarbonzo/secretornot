"""Metadata-only audit logger.

Writes one JSON line per routing decision.  NEVER logs prompt content.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from config import AUDIT_LOG_FILE
from models import (
    AuditEntry,
    ClassificationResult,
    Destination,
    LatencyMs,
)

_logger = logging.getLogger("secretornot.audit")

# File handler — append JSONL
_file_handler = logging.FileHandler(AUDIT_LOG_FILE, mode="a", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(message)s"))
_logger.addHandler(_file_handler)
_logger.setLevel(logging.INFO)

# Also emit to stdout for container environments
_stdout_handler = logging.StreamHandler()
_stdout_handler.setFormatter(logging.Formatter("[audit] %(message)s"))
_logger.addHandler(_stdout_handler)


def log_decision(
    *,
    request_id: str,
    result: ClassificationResult,
    destination: Destination,
    model_requested: str | None = None,
    classification_ms: float = 0.0,
    inference_ms: float = 0.0,
    total_ms: float = 0.0,
) -> None:
    """Write a single audit entry.  No prompt content — metadata only."""
    entry = AuditEntry(
        request_id=request_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        classification=result.classification,
        layer_triggered=result.layer,
        rule_matched=result.rule_matched,
        confidence=result.confidence,
        routed_to=destination,
        model_requested=model_requested,
        latency_ms=LatencyMs(
            classification=round(classification_ms, 2),
            inference=round(inference_ms, 2),
            total=round(total_ms, 2),
        ),
    )
    _logger.info(entry.model_dump_json())
