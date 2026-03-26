"""Layer 1: Rule-based pre-filter.

Fast regex/keyword matching.  If any rule fires → PRIVATE immediately,
Layer 2 is never called.  Runs in <1 ms on any input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuleMatch:
    matched: bool
    rule_name: str | None = None


# Each rule is (name, compiled_regex).  Order doesn't matter — we short-circuit
# on the first hit.
_RULES: list[tuple[str, re.Pattern[str]]] = []


def _add(name: str, pattern: str, flags: int = re.IGNORECASE) -> None:
    _RULES.append((name, re.compile(pattern, flags)))


# ── Explicit developer flag ─────────────────────────────────────────────
_add("explicit_private_tag", r"\[PRIVATE\]")

# ── PII patterns ────────────────────────────────────────────────────────
# SSN  (xxx-xx-xxxx)
_add("ssn", r"\b\d{3}-\d{2}-\d{4}\b")
# Credit card (13-19 digits, optionally separated by spaces/dashes)
_add("credit_card", r"\b(?:\d[ -]*?){13,19}\b")
# US phone  (xxx) xxx-xxxx  or  xxx-xxx-xxxx  or  +1xxxxxxxxxx
_add("phone_number", r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
# Email address
_add("email_address", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b")

# ── Medical / health ────────────────────────────────────────────────────
_add(
    "medical",
    r"\b(?:diagnosis|diagnosed|patient|prescription|symptoms?|treatment plan"
    r"|medical record|health record|HIPAA|lab results?|blood test"
    r"|radiology|prognosis|clinical trial)\b",
)

# ── Credentials / secrets ───────────────────────────────────────────────
_add(
    "credentials",
    r"\b(?:password|passwd|api[_-]?key|secret[_-]?key|private[_-]?key"
    r"|access[_-]?token|refresh[_-]?token|bearer token|client[_-]?secret"
    r"|ssh[_-]?key|pgp[_-]?key|encryption[_-]?key)\b",
)
# Hex/base64 strings that look like actual keys (32+ hex chars or long base64)
_add("literal_key", r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{36,}|xox[bps]-[A-Za-z0-9-]{10,})\b")

# ── Financial ───────────────────────────────────────────────────────────
_add(
    "financial",
    r"\b(?:account\s*(?:number|#|no)|routing\s*(?:number|#|no)"
    r"|salary|compensation|tax\s*return|w-?2|1099|bank\s*statement"
    r"|wire\s*transfer|iban|swift\s*code)\b",
)

# ── Legal ───────────────────────────────────────────────────────────────
_add(
    "legal",
    r"\b(?:confidential|attorney[- ]client|privileged|under\s+seal"
    r"|non-?disclosure|trade\s+secret|proprietary)\b",
)

# ── Personal identifiers ────────────────────────────────────────────────
_add(
    "personal_id",
    r"\b(?:passport\s*(?:number|#|no)|driver'?s?\s*licen[sc]e"
    r"|date\s*of\s*birth|social\s*security|national\s*id)\b",
)


def check(text: str) -> RuleMatch:
    """Run all rules against *text*.  Returns on first match."""
    for name, pattern in _RULES:
        if pattern.search(text):
            return RuleMatch(matched=True, rule_name=name)
    return RuleMatch(matched=False)
