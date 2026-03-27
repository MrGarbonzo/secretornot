"""Minimal chatbox demo — type a prompt, get PRIVATE or PUBLIC."""

import os
os.environ.setdefault("SECRET_AI_ENDPOINT", "https://unused")
os.environ.setdefault("SECRET_AI_API_KEY", "unused")
os.environ.setdefault("PUBLIC_LLM_ENDPOINT", "https://unused")
os.environ.setdefault("PUBLIC_LLM_API_KEY", "unused")
os.environ.setdefault("DISTILBERT_MODEL_PATH", "training/model.onnx")

import asyncio
from classifier.rule_filter import check as rule_check
from classifier.model_classifier import classify as model_classify
from config import CONFIDENCE_THRESHOLD

async def classify_prompt(text):
    # Layer 1: rules
    rule = rule_check(text)
    if rule.matched:
        return "PRIVATE", f"rule: {rule.rule_name}", 1.0

    # Layer 2: model
    result = await model_classify(text)
    classification = result["classification"].value
    confidence = result["confidence"]

    if classification == "PUBLIC" and confidence < CONFIDENCE_THRESHOLD:
        return "PRIVATE", "low confidence", confidence

    return classification, "model", confidence

def main():
    print("SecretOrNot Demo — type a prompt, get PRIVATE or PUBLIC")
    print("Type 'quit' to exit\n")

    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if text.strip().lower() in ("quit", "exit", "q"):
            break
        if not text.strip():
            continue

        classification, source, confidence = asyncio.run(classify_prompt(text))
        print(f"  {classification}  ({source}, confidence: {confidence:.2f})\n")

if __name__ == "__main__":
    main()
