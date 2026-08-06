#!/usr/bin/env python3
"""Fine-tune FinBERT for FOMC hawkish/dovish/neutral sentence classification.

This script completes the advanced model path for the project without adding new
dependencies beyond torch, transformers, pandas, and scikit-learn.

Two label sources are supported:
1. Reviewed labels: pass --labels-csv with columns sentence,label.
2. Weak labels: if --labels-csv is omitted, labels are generated with the
   dictionary classifier from analyze_tone.py. This is useful for a reproducible
   course demo, but reviewed labels are preferred for final research claims.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_tone import score_sentence, split_sentences  # noqa: E402


MODEL_NAME = "ProsusAI/finbert"
LABELS = ["dovish", "neutral", "hawkish"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_reviewed_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sentence", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df[["sentence", "label"]].copy()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(LABEL_TO_ID)]
    df = df[df["sentence"].astype(str).str.split().str.len() >= 6]
    if df.empty:
        raise ValueError("No usable reviewed labels found.")
    return df.drop_duplicates("sentence")


def build_weak_labels(metadata_path: Path, max_per_class: int, seed: int) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    rows: list[dict[str, str]] = []
    for _, record in metadata.iterrows():
        text = Path(record["local_path"]).read_text(encoding="utf-8")
        for sentence in split_sentences(text):
            label = score_sentence(sentence)
            rows.append({"date": record["date"], "sentence": sentence, "label": label})

    df = pd.DataFrame(rows).drop_duplicates("sentence")
    sampled = []
    for label in LABELS:
        group = df[df["label"] == label]
        n = min(max_per_class, len(group))
        sampled.append(group.sample(n=n, random_state=seed))
    out = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=seed)
    return out


class SentenceDataset(Dataset):
    def __init__(self, sentences: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            sentences,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: value[idx] for key, value in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    macro_f1: float
    predictions: list[int]
    labels: list[int]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    losses: list[float] = []
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
            preds = outputs.logits.argmax(dim=-1).detach().cpu().tolist()
            gold = batch["labels"].detach().cpu().tolist()
            predictions.extend(preds)
            labels.extend(gold)
    return EvalResult(
        loss=float(np.mean(losses)) if losses else 0.0,
        accuracy=accuracy_score(labels, predictions),
        macro_f1=f1_score(labels, predictions, average="macro"),
        predictions=predictions,
        labels=labels,
    )


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    device = choose_device()
    model.to(device)

    train_dataset = SentenceDataset(
        train_df["sentence"].tolist(),
        train_df["label"].map(LABEL_TO_ID).tolist(),
        tokenizer,
        max_length,
    )
    val_dataset = SentenceDataset(
        val_df["sentence"].tolist(),
        val_df["label"].map(LABEL_TO_ID).tolist(),
        tokenizer,
        max_length,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            train_losses.append(float(outputs.loss.detach().cpu()))

        val_result = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_macro_f1": val_result.macro_f1,
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.3f} "
            f"val_macro_f1={row['val_macro_f1']:.3f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    final_eval = evaluate(model, val_loader, device)
    report = classification_report(
        [ID_TO_LABEL[i] for i in final_eval.labels],
        [ID_TO_LABEL[i] for i in final_eval.predictions],
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "model_name": MODEL_NAME,
        "device": str(device),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "history": history,
        "classification_report": report,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    return {"model": model, "tokenizer": tokenizer, "device": device, "metrics": metrics}


def predict_all_minutes(model, tokenizer, device: torch.device, metadata_path: Path, output_csv: Path, batch_size: int, max_length: int) -> None:
    rows: list[dict[str, str]] = []
    metadata = pd.read_csv(metadata_path)
    for _, record in metadata.iterrows():
        text = Path(record["local_path"]).read_text(encoding="utf-8")
        for sentence in split_sentences(text):
            rows.append({"date": record["date"], "sentence": sentence})

    all_df = pd.DataFrame(rows)
    predictions: list[str] = []
    confidences: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(all_df), batch_size):
            batch_sentences = all_df["sentence"].iloc[start : start + batch_size].tolist()
            encoded = tokenizer(
                batch_sentences,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            predictions.extend([ID_TO_LABEL[int(i)] for i in pred.detach().cpu()])
            confidences.extend([float(x) for x in conf.detach().cpu()])

    all_df["finbert_label"] = predictions
    all_df["finbert_confidence"] = confidences
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(output_csv, index=False)

    doc_scores = (
        all_df.assign(
            hawkish=lambda x: (x["finbert_label"] == "hawkish").astype(int),
            dovish=lambda x: (x["finbert_label"] == "dovish").astype(int),
        )
        .groupby("date")
        .agg(
            sentence_count=("sentence", "size"),
            hawkish_sentences=("hawkish", "sum"),
            dovish_sentences=("dovish", "sum"),
            avg_confidence=("finbert_confidence", "mean"),
        )
        .reset_index()
    )
    doc_scores["finbert_net_hawkish_score"] = (
        doc_scores["hawkish_sentences"] - doc_scores["dovish_sentences"]
    ) / doc_scores["sentence_count"]
    doc_scores.to_csv(output_csv.with_name("finbert_document_scores.csv"), index=False)
    print(f"Wrote sentence predictions to {output_csv}")
    print(f"Wrote document scores to {output_csv.with_name('finbert_document_scores.csv')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-csv", type=Path, default=None, help="Reviewed labels CSV with sentence,label columns.")
    parser.add_argument("--metadata", type=Path, default=PROJECT_ROOT / "data/processed/fomc_metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models/finbert_policy_tone")
    parser.add_argument("--predictions", type=Path, default=PROJECT_ROOT / "data/processed/finbert_sentence_predictions.csv")
    parser.add_argument("--max-per-class", type=int, default=800, help="Weak-label sample cap per class.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-predict-all", action="store_true")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    set_seed(args.seed)

    if args.labels_csv:
        labeled = load_reviewed_labels(args.labels_csv)
        label_source = f"reviewed labels from {args.labels_csv}"
    else:
        labeled = build_weak_labels(args.metadata, args.max_per_class, args.seed)
        label_source = "weak dictionary labels"

    print(f"Loaded {len(labeled)} training examples from {label_source}.")
    print(labeled["label"].value_counts().reindex(LABELS, fill_value=0).to_string())

    train_df, val_df = train_test_split(
        labeled,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=labeled["label"],
    )
    result = train_model(
        train_df=train_df,
        val_df=val_df,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )

    if not args.skip_predict_all:
        predict_all_minutes(
            model=result["model"],
            tokenizer=result["tokenizer"],
            device=result["device"],
            metadata_path=args.metadata,
            output_csv=args.predictions,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )


if __name__ == "__main__":
    main()
