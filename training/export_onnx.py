"""Export fine-tuned DistilBERT to ONNX for CPU inference in TEE.

Usage:
    python training/export_onnx.py

Input:  training/model_checkpoint/
Output: training/model.onnx
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHECKPOINT_DIR = Path(__file__).parent / "model_checkpoint"
ONNX_PATH = Path(__file__).parent / "model.onnx"
MAX_LENGTH = 256

VALIDATION_SAMPLES = [
    "What is the capital of France?",
    "My SSN is 123-45-6789, can you help me file my taxes?",
    "Explain how photosynthesis works",
    "I've been having chest pains and my doctor said I might need surgery",
    "Write a Python function to sort a list",
    "Our company's Q3 revenue was $4.2M but don't share this externally",
    "What's the difference between TCP and UDP?",
    "My password is hunter2 and I can't log into my bank account",
    "Recommend a good book about machine learning",
    "The board voted to acquire TechCorp for $50M, this is confidential",
]


def main() -> None:
    if not CHECKPOINT_DIR.exists():
        print(f"ERROR: {CHECKPOINT_DIR} not found. Run train.py first.")
        sys.exit(1)

    print(f"Loading model from {CHECKPOINT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR))
    model.eval()

    # ── Export to ONNX ───────────────────────────────────────────────────
    print(f"Exporting to {ONNX_PATH}...")

    dummy = tokenizer(
        "test input",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )

    # ── Validate: PyTorch vs ONNX ────────────────────────────────────────
    print(f"\nValidating with {len(VALIDATION_SAMPLES)} samples...")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    max_diff = 0.0
    all_match = True

    for i, text in enumerate(VALIDATION_SAMPLES):
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # PyTorch inference
        with torch.no_grad():
            pt_logits = model(tokens["input_ids"], tokens["attention_mask"]).logits.numpy()

        # ONNX inference
        onnx_logits = session.run(
            ["logits"],
            {
                "input_ids": tokens["input_ids"].numpy(),
                "attention_mask": tokens["attention_mask"].numpy(),
            },
        )[0]

        diff = np.max(np.abs(pt_logits - onnx_logits))
        max_diff = max(max_diff, diff)

        pt_label = "PRIVATE" if np.argmax(pt_logits) == 1 else "PUBLIC"
        onnx_label = "PRIVATE" if np.argmax(onnx_logits) == 1 else "PUBLIC"
        match = pt_label == onnx_label

        if not match:
            all_match = False

        print(f"  [{i+1:>2d}] PT={pt_label:<8s} ONNX={onnx_label:<8s} diff={diff:.6f} {'OK' if match else 'MISMATCH'}")

    # ── Summary ──────────────────────────────────────────────────────────
    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"ONNX model exported: {ONNX_PATH}")
    print(f"  File size:    {size_mb:.1f} MB")
    print(f"  Max logit diff: {max_diff:.8f}")
    print(f"  All predictions match: {'YES' if all_match else 'NO'}")
    print(f"{'='*60}")

    if not all_match:
        print("WARNING: PyTorch and ONNX outputs diverged on some samples!")
        sys.exit(1)


if __name__ == "__main__":
    main()
