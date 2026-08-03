#!/usr/bin/env python3
"""Optional FinBERT inference scaffold for the advanced model extension.

The submitted project uses the dictionary classifier as the fully reproducible
baseline. This script shows how the same sentence-level pipeline can be swapped
to a pretrained financial language model once a small reviewed hawkish/dovish
label file is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_NAME = "ProsusAI/finbert"


def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, truncation=True)


def map_finbert_sentiment(label: str) -> str:
    """Map FinBERT's generic sentiment labels to a policy-tone proxy.

    For a production-quality advanced model, replace this mapping by fine-tuning
    on hawkish/dovish/neutral labels. Positive financial sentiment is not always
    hawkish, so this file is intentionally framed as an extension scaffold.
    """
    normalized = label.lower()
    if normalized == "positive":
        return "hawkish"
    if normalized == "negative":
        return "dovish"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", default="data/processed/labeled_policy_sentences.csv")
    parser.add_argument("--output", default="data/processed/finbert_sentence_predictions.csv")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    sentences = pd.read_csv(args.sentences).head(args.limit)
    classifier = load_classifier()
    predictions = classifier(sentences["sentence"].tolist(), batch_size=16)
    sentences["finbert_raw_label"] = [p["label"] for p in predictions]
    sentences["finbert_score"] = [p["score"] for p in predictions]
    sentences["policy_tone_proxy"] = sentences["finbert_raw_label"].map(map_finbert_sentiment)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    sentences.to_csv(args.output, index=False)
    print(f"Wrote {len(sentences)} FinBERT predictions to {args.output}.")


if __name__ == "__main__":
    main()
