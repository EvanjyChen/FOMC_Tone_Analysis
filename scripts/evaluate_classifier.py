#!/usr/bin/env python3
"""Evaluate classifiers against the completed human audit.

Usage: python3 scripts/evaluate_classifier.py
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from analyze_tone import score_sentence
from finbert_optional import LABELS, build_weak_labels
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def sentence_id(date: str, sentence: str) -> str:
    return hashlib.sha256(f"{date}\0{sentence}".encode()).hexdigest()[:16]


def require_columns(
    frame: pd.DataFrame, path: Path, columns: list[str]
) -> pd.DataFrame:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[columns]


def validate_audit(audit: pd.DataFrame, path: Path) -> pd.DataFrame:
    audit = require_columns(
        audit, path, ["audit_id", "date", "sentence", "manual_label"]
    ).copy()
    audit["manual_label"] = audit["manual_label"].astype(str).str.lower().str.strip()
    invalid = audit[~audit["manual_label"].isin(LABELS)]
    if not invalid.empty:
        raise ValueError(
            f"manual_label must be one of {sorted(LABELS)}; "
            f"invalid or blank rows: {invalid['audit_id'].tolist()}"
        )
    if audit["audit_id"].duplicated().any() or audit["sentence"].duplicated().any():
        raise ValueError("Audit rows must have unique audit_id and sentence values")
    expected_ids = [
        sentence_id(row.date, row.sentence) for row in audit.itertuples(index=False)
    ]
    if audit["audit_id"].tolist() != expected_ids:
        raise ValueError(
            "One or more audit_id values do not match their date and sentence"
        )
    return audit


def canonical_frame_sha256(frame: pd.DataFrame, columns: list[str]) -> str:
    records = (
        frame.loc[:, columns]
        .astype(str)
        .sort_values(columns, kind="stable")
        .to_dict(orient="records")
    )
    payload = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_audit_provenance(
    audit: pd.DataFrame, manifest_path: Path
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weak_config = manifest["weak_pool"]
    weak_pool = build_weak_labels(
        Path(weak_config["metadata_path"]),
        int(weak_config["max_per_class"]),
        int(weak_config["seed"]),
    )

    if len(weak_pool) != int(weak_config["rows"]):
        raise ValueError(
            f"Weak-label pool has {len(weak_pool)} rows; "
            f"manifest requires {weak_config['rows']}"
        )
    weak_hash = canonical_frame_sha256(weak_pool, ["date", "sentence", "label"])
    if weak_hash != weak_config["sha256"]:
        raise ValueError("Weak-label pool does not match the frozen audit manifest")

    audit_hash = canonical_frame_sha256(audit, ["audit_id", "date", "sentence"])
    if audit_hash != manifest["audit_input_sha256"]:
        raise ValueError("Audit inputs do not match the frozen audit manifest")

    overlap = set(audit["sentence"]).intersection(weak_pool["sentence"])
    if overlap:
        raise ValueError(
            f"Audit contains {len(overlap)} sentence(s) from the weak-label pool"
        )

    actual_counts = (
        audit["dictionary_label"].value_counts().reindex(LABELS, fill_value=0).to_dict()
    )
    expected_counts = {
        label: int(count)
        for label, count in manifest["expected_dictionary_counts"].items()
    }
    if actual_counts != expected_counts:
        raise ValueError(
            f"Dictionary-prediction strata are {actual_counts}; "
            f"manifest requires {expected_counts}"
        )

    expected_year_counts = {
        str(year): 1 for year in manifest["expected_years_per_class"]
    }
    for label in LABELS:
        actual_year_counts = (
            audit.loc[audit["dictionary_label"] == label, "date"]
            .str[:4]
            .value_counts()
            .to_dict()
        )
        if actual_year_counts != expected_year_counts:
            raise ValueError(
                f"{label} audit years are {actual_year_counts}; "
                f"manifest requires {expected_year_counts}"
            )

    return {
        "selection_status": manifest["selection_status"],
        "audit_input_hash_verified": True,
        "weak_pool_hash_verified": True,
        "weak_pool_rows": len(weak_pool),
        "weak_pool_overlap": 0,
        "dictionary_strata_verified": True,
        "temporal_spread_verified": True,
        "note": manifest["note"],
    }


def metric_rows(
    model: str, truth: pd.Series, predictions: pd.Series
) -> list[dict[str, object]]:
    accuracy = accuracy_score(truth, predictions)
    macro_f1 = f1_score(
        truth, predictions, labels=LABELS, average="macro", zero_division=0
    )
    rows: list[dict[str, object]] = [
        {
            "model": model,
            "class": "overall",
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "precision": None,
            "recall": None,
            "f1": None,
            "support": len(truth),
        }
    ]
    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
    rows.extend(
        {
            "model": model,
            "class": label,
            "accuracy": None,
            "macro_f1": None,
            "precision": p,
            "recall": r,
            "f1": f1_value,
            "support": int(n),
        }
        for label, p, r, f1_value, n in zip(
            LABELS, precision, recall, class_f1, support, strict=True
        )
    )
    return rows


def choose_examples(results: pd.DataFrame) -> pd.DataFrame:
    chosen: list[pd.Series] = []
    for label in LABELS:
        candidates = results[results["manual_label"] == label]
        agreed = candidates[
            (candidates["dictionary_label"] == label)
            & (candidates["finbert_label"] == label)
        ]
        chosen.append(
            (agreed if not agreed.empty else candidates).sort_values("audit_id").iloc[0]
        )
    disagreements = results[results["dictionary_label"] != results["finbert_label"]]
    if disagreements.empty:
        disagreements = results[
            (results["dictionary_label"] != results["manual_label"])
            | (results["finbert_label"] != results["manual_label"])
        ]
    if not disagreements.empty:
        chosen.append(
            disagreements.sort_values(["finbert_confidence", "audit_id"]).iloc[0]
        )
    examples = pd.DataFrame(chosen).drop_duplicates("audit_id")
    examples.insert(0, "example_type", [*LABELS, "model_disagreement"][: len(examples)])
    return examples


def save_confusion_matrix(results: pd.DataFrame, path: Path) -> None:
    plt.switch_backend("Agg")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (title, column) in zip(
        axes,
        [("Dictionary", "dictionary_label"), ("FinBERT", "finbert_label")],
        strict=True,
    ):
        ConfusionMatrixDisplay.from_predictions(
            results["manual_label"],
            results[column],
            labels=LABELS,
            display_labels=LABELS,
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(title)
    fig.suptitle("Classifier Performance on Independent Human Audit")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def evaluate_audit(args: argparse.Namespace) -> None:
    audit = validate_audit(
        pd.read_csv(args.audit, keep_default_na=False, dtype={"date": str}), args.audit
    )
    audit["dictionary_label"] = audit["sentence"].map(score_sentence)
    provenance = validate_audit_provenance(audit, args.audit_manifest)

    columns = ["date", "sentence", "finbert_label", "finbert_confidence"]
    finbert = require_columns(
        pd.read_csv(args.finbert_predictions, dtype={"date": str}),
        args.finbert_predictions,
        columns,
    ).drop_duplicates(["date", "sentence"])
    results = audit.merge(
        finbert, on=["date", "sentence"], how="left", validate="one_to_one"
    )
    if results[["finbert_label", "finbert_confidence"]].isna().any().any():
        missing_ids = results.loc[results["finbert_label"].isna(), "audit_id"].tolist()
        raise ValueError(f"Missing FinBERT predictions for audit rows: {missing_ids}")
    if not set(results["finbert_label"]).issubset(LABELS):
        raise ValueError("FinBERT predictions contain invalid labels")

    metrics = metric_rows(
        "dictionary", results["manual_label"], results["dictionary_label"]
    ) + metric_rows("finbert", results["manual_label"], results["finbert_label"])
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(args.metrics, index=False)
    choose_examples(results).to_csv(args.examples, index=False)
    save_confusion_matrix(results, args.confusion_matrix)

    metadata = {
        "audit_rows": len(results),
        "labels": LABELS,
        "provenance": provenance,
        "annotation": {
            "blind_columns": ["audit_id", "date", "sentence", "manual_label"],
            "rubric": {
                "hawkish": (
                    "expresses or supports tighter policy, restraint, or inflation "
                    "pressure requiring tightening"
                ),
                "dovish": (
                    "expresses or supports easier policy, accommodation, or weakness "
                    "requiring support"
                ),
                "neutral": (
                    "expresses neither direction, balances both directions, or lacks "
                    "enough context"
                ),
            },
        },
        "limitations": [
            f"The human audit contains only {len(results)} sentences.",
            "One annotator does not measure inter-rater agreement.",
            (
                "Stratification by dictionary prediction does not reproduce the "
                "corpus class distribution."
            ),
            (
                "FinBERT was trained with dictionary-generated weak labels; the "
                "audit is its independent evaluation."
            ),
            provenance["note"],
        ],
    }
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Verified audit provenance: "
        f"{len(audit)} frozen inputs, {provenance['weak_pool_overlap']} weak-pool "
        "overlap, balanced dictionary strata, and expected year coverage"
    )
    print(f"Wrote classifier metrics to {args.metrics}")
    print(f"Wrote traceable examples to {args.examples}")
    print(f"Wrote empirical confusion matrices to {args.confusion_matrix}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", type=Path, default="data/processed/manual_audit_labels.csv"
    )
    parser.add_argument(
        "--finbert-predictions",
        type=Path,
        default="data/processed/finbert_sentence_predictions.csv",
    )
    parser.add_argument(
        "--metrics", type=Path, default="data/processed/classifier_metrics.csv"
    )
    parser.add_argument(
        "--examples", type=Path, default="data/processed/classifier_examples.csv"
    )
    parser.add_argument(
        "--confusion-matrix",
        type=Path,
        default="figures/confusion_matrix_audit.png",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default="data/processed/classifier_evaluation_metadata.json",
    )
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default="data/processed/audit_selection_manifest.json",
    )
    return parser


def main() -> None:
    os.chdir(Path(__file__).resolve().parents[1])
    evaluate_audit(build_parser().parse_args())


if __name__ == "__main__":
    main()
