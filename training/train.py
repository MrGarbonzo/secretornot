"""Fine-tune distilbert-base-uncased as a binary privacy classifier.

Usage:
    python training/train.py

Input:  training/training_data.jsonl
Output: training/model_checkpoint/

Uses CUDA automatically if available (RTX 5070 Ti).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

DATA_FILE = Path(__file__).parent / "training_data.jsonl"
CHECKPOINT_DIR = Path(__file__).parent / "model_checkpoint"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


def load_data() -> Dataset:
    """Load training_data.jsonl into a HuggingFace Dataset."""
    records = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print(f"ERROR: No data found in {DATA_FILE}")
        sys.exit(1)

    from datasets import ClassLabel
    ds = Dataset.from_dict({
        "text": [r["text"] for r in records],
        "label": [int(r["label"]) for r in records],
    })
    # Cast label to ClassLabel so stratify_by_column works
    ds = ds.cast_column("label", ClassLabel(names=["PUBLIC", "PRIVATE"]))
    return ds


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, pos_label=1),
        "recall": recall_score(labels, preds, pos_label=1),
        "f1": f1_score(labels, preds, pos_label=1),
        "false_negative_rate": 1.0 - recall_score(labels, preds, pos_label=1),
    }


def main() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run generate_data.py first.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ────────────────────────────────────────────────────────
    dataset = load_data()
    print(f"Loaded {len(dataset)} examples")
    print(f"  Label distribution: {dict(zip(*np.unique(dataset['label'], return_counts=True)))}")

    # 80/20 split
    split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds)}  |  Eval: {len(eval_ds)}")

    # ── Tokenize ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    # ── Model ────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "PUBLIC", 1: "PRIVATE"},
        label2id={"PUBLIC": 0, "PRIVATE": 1},
    )

    # ── Training ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        fp16=device == "cuda",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\nTraining...")
    trainer.train()

    # ── Save best model + tokenizer ──────────────────────────────────────
    trainer.save_model(str(CHECKPOINT_DIR))
    tokenizer.save_pretrained(str(CHECKPOINT_DIR))
    print(f"\nModel saved to {CHECKPOINT_DIR}")

    # ── Final evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    metrics = trainer.evaluate()
    print(f"  Accuracy:            {metrics['eval_accuracy']:.4f}")
    print(f"  Precision (PRIVATE): {metrics['eval_precision']:.4f}")
    print(f"  Recall (PRIVATE):    {metrics['eval_recall']:.4f}")
    print(f"  F1 (PRIVATE):        {metrics['eval_f1']:.4f}")
    print(f"  False Negative Rate: {metrics['eval_false_negative_rate']:.4f}")

    # Targets
    acc_ok = metrics["eval_accuracy"] > 0.95
    fnr_ok = metrics["eval_false_negative_rate"] < 0.02
    print(f"\n  Target accuracy > 95%:        {'PASS' if acc_ok else 'FAIL'}")
    print(f"  Target FNR < 2%:              {'PASS' if fnr_ok else 'FAIL'}")

    # Full classification report
    preds = trainer.predict(eval_ds)
    pred_labels = np.argmax(preds.predictions, axis=-1)
    print(f"\n{classification_report(eval_ds['label'], pred_labels, target_names=['PUBLIC', 'PRIVATE'])}")


if __name__ == "__main__":
    main()
