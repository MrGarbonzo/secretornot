"""Generate labeled training examples for the DistilBERT privacy classifier.

Uses the Anthropic Python SDK to call Claude and generate diverse labeled
prompts covering all privacy categories plus ambiguous edge cases.

Usage:
    ANTHROPIC_API_KEY=sk-... python training/generate_data.py

Output:
    training/training_data.jsonl  — one JSON object per line:
    {"text": "...", "label": 0}   (0 = PUBLIC)
    {"text": "...", "label": 1}   (1 = PRIVATE)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import anthropic

OUTPUT_FILE = Path(__file__).parent / "training_data.jsonl"
BATCH_SIZE = 50
TOTAL_EXAMPLES = 4000
EXAMPLES_PER_CLASS = TOTAL_EXAMPLES // 2

# ── Prompt templates ─────────────────────────────────────────────────────

PRIVATE_CATEGORIES = [
    {
        "name": "PII — SSNs, credit cards, phone numbers, emails with names",
        "count": 200,
    },
    {
        "name": "Medical — personal health questions, symptoms, diagnoses, prescriptions, therapy",
        "count": 250,
    },
    {
        "name": "Credentials — passwords, API keys, tokens, SSH keys, secrets in code snippets",
        "count": 200,
    },
    {
        "name": "Financial — salaries, bank accounts, tax info, wire transfers, investment portfolios",
        "count": 200,
    },
    {
        "name": "Legal — attorney-client privileged, NDAs, contracts, settlement details",
        "count": 150,
    },
    {
        "name": "Personal identifiers — passport numbers, driver's license, DOB, national IDs",
        "count": 150,
    },
    {
        "name": "Implicit business sensitivity — internal strategy, unreleased product details, M&A, board discussions, performance reviews",
        "count": 250,
    },
    {
        "name": "HR and personnel — hiring decisions, termination reasons, employee complaints, disciplinary actions",
        "count": 150,
    },
    {
        "name": "Mixed prompts — starts with a general question but contains or references private data embedded naturally",
        "count": 200,
    },
    {
        "name": "Ambiguous edge cases — prompts where privacy is contextual: asking about 'my' health, 'our' company revenue, personal relationships, private conversations",
        "count": 250,
    },
]

PUBLIC_CATEGORIES = [
    {
        "name": "General knowledge — science, history, geography, math, definitions",
        "count": 300,
    },
    {
        "name": "Programming — code help, debugging, algorithms, language syntax, open-source libraries",
        "count": 300,
    },
    {
        "name": "Creative writing — stories, poems, essays on public topics",
        "count": 200,
    },
    {
        "name": "Education — homework help, study questions, exam prep, concept explanations",
        "count": 200,
    },
    {
        "name": "Professional skills — resume writing tips (generic), interview prep, career advice (no personal details)",
        "count": 150,
    },
    {
        "name": "Technology — how products work, comparisons, troubleshooting (generic, no credentials)",
        "count": 200,
    },
    {
        "name": "Casual conversation — jokes, recommendations, opinions, philosophical questions",
        "count": 200,
    },
    {
        "name": "Business (generic) — public market analysis, industry trends, published financial reports, generic strategy frameworks",
        "count": 200,
    },
    {
        "name": "Near-miss edge cases — prompts that mention medical/legal/financial topics in a general educational way without personal information",
        "count": 250,
    },
]


def _build_generation_prompt(category_name: str, label: str, count: int) -> str:
    return f"""Generate exactly {count} example prompts that a user might send to an LLM.

Category: {category_name}
Classification: {label}

Requirements:
- Each prompt should be realistic — something a real person would actually type
- Vary length: some short (1 sentence), some medium (2-3 sentences), some long (paragraph)
- Vary formality: casual, professional, technical
- Vary phrasing: questions, instructions, requests, statements
- Do NOT include any labels, numbers, or metadata — just the raw prompt text
- Do NOT repeat yourself — every example must be meaningfully different
- {"Include subtle, implicit sensitivity — not just keyword-obvious cases" if label == "PRIVATE" else "Make sure these are clearly general-knowledge / public — no personal data even implicitly"}

Output format: Return ONLY a JSON array of strings, one per example. No markdown fences, no explanation.
Example: ["prompt one here", "prompt two here", ...]"""


def generate_batch(
    client: anthropic.Anthropic,
    category_name: str,
    label: str,
    count: int,
) -> list[str]:
    """Call Claude to generate a batch of examples. Returns list of prompt strings."""
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        temperature=1.0,
        messages=[
            {
                "role": "user",
                "content": _build_generation_prompt(category_name, label, count),
            }
        ],
    )

    raw = resp.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        examples = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  WARNING: Failed to parse response for '{category_name}', retrying...")
        return []

    if not isinstance(examples, list):
        print(f"  WARNING: Expected list, got {type(examples).__name__}")
        return []

    return [str(e) for e in examples if isinstance(e, str) and len(e.strip()) > 10]


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    all_examples: list[dict] = []

    # ── Generate PRIVATE examples ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating PRIVATE examples")
    print(f"{'='*60}")

    for cat in PRIVATE_CATEGORIES:
        remaining = cat["count"]
        while remaining > 0:
            batch_size = min(remaining, BATCH_SIZE)
            print(f"  [{len([e for e in all_examples if e['label'] == 1]):>4d}/{EXAMPLES_PER_CLASS}] "
                  f"{cat['name'][:50]}... ({batch_size} requested)")

            examples = generate_batch(client, cat["name"], "PRIVATE", batch_size)
            for text in examples:
                all_examples.append({"text": text.strip(), "label": 1})
            remaining -= max(len(examples), 1)  # always make progress

    # ── Generate PUBLIC examples ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Generating PUBLIC examples")
    print(f"{'='*60}")

    for cat in PUBLIC_CATEGORIES:
        remaining = cat["count"]
        while remaining > 0:
            batch_size = min(remaining, BATCH_SIZE)
            print(f"  [{len([e for e in all_examples if e['label'] == 0]):>4d}/{EXAMPLES_PER_CLASS}] "
                  f"{cat['name'][:50]}... ({batch_size} requested)")

            examples = generate_batch(client, cat["name"], "PUBLIC", batch_size)
            for text in examples:
                all_examples.append({"text": text.strip(), "label": 0})
            remaining -= max(len(examples), 1)

    # ── Write output ─────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_examples:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    private_count = sum(1 for e in all_examples if e["label"] == 1)
    public_count = sum(1 for e in all_examples if e["label"] == 0)

    print(f"\n{'='*60}")
    print(f"Done! Wrote {len(all_examples)} examples to {OUTPUT_FILE}")
    print(f"  PRIVATE: {private_count}")
    print(f"  PUBLIC:  {public_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
