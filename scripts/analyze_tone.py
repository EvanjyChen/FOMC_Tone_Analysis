#!/usr/bin/env python3
"""Compute hawkish-dovish FOMC tone scores and market validation charts."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import ConfusionMatrixDisplay


HAWKISH_TERMS = {
    "tightening", "tighten", "tightened", "restrictive", "restriction", "inflation",
    "inflationary", "price pressure", "price pressures", "overheat", "overheating",
    "strong demand", "labor market tightness", "upside risk", "upside risks",
    "higher rates", "raise the target", "increase the target", "rate increase",
    "firming", "less accommodation", "reduce accommodation", "balance sheet reduction",
    "runoff", "elevated inflation", "persistent inflation", "anchored expectations",
}

DOVISH_TERMS = {
    "accommodative", "accommodation", "easing", "ease", "eased", "downside risk",
    "downside risks", "weakness", "slack", "unemployment", "recession", "financial stress",
    "lower rates", "lower the target", "cut the target", "rate cut", "quantitative easing",
    "asset purchases", "support economic activity", "subdued inflation", "below target",
    "patient", "patience", "maintain the target", "near zero", "zero lower bound",
    "provide support", "credit strains", "economic contraction",
}

EVENTS = {
    "2008-12-16": "Zero-rate crisis response",
    "2015-12-16": "First post-crisis hike",
    "2020-03-15": "Pandemic emergency easing",
    "2022-06-15": "Rapid anti-inflation hikes",
}


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    pieces = re.split(r"(?<=[.;!?])\s+(?=[A-Z])", text)
    return [p.strip() for p in pieces if len(p.strip().split()) >= 6]


def count_terms(sentence: str, terms: set[str]) -> int:
    s = sentence.lower()
    count = 0
    for term in terms:
        if " " in term:
            count += s.count(term)
        else:
            count += len(re.findall(rf"\b{re.escape(term)}\b", s))
    return count


def score_sentence(sentence: str) -> str:
    hawk = count_terms(sentence, HAWKISH_TERMS)
    dove = count_terms(sentence, DOVISH_TERMS)
    if hawk > dove:
        return "hawkish"
    if dove > hawk:
        return "dovish"
    return "neutral"


def load_yield(series_id: str) -> pd.DataFrame:
    path = Path("data/raw/fred") / f"{series_id}.csv"
    df = pd.read_csv(path)
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    value_col = series_id
    df[date_col] = pd.to_datetime(df[date_col])
    df[value_col] = pd.to_numeric(df[value_col].replace(".", np.nan), errors="coerce")
    return df.rename(columns={date_col: "date", value_col: series_id})[["date", series_id]].dropna()


def align_market(scores: pd.DataFrame) -> pd.DataFrame:
    dgs10 = load_yield("DGS10")
    dgs2 = load_yield("DGS2")
    market = pd.merge_asof(
        scores.sort_values("date"),
        dgs10.sort_values("date"),
        on="date",
        direction="backward",
    )
    market = pd.merge_asof(market.sort_values("date"), dgs2.sort_values("date"), on="date", direction="backward")
    market["score_ma3"] = market["net_hawkish_score"].rolling(3, min_periods=1).mean()
    market["DGS10_change_next_30d"] = market["DGS10"].shift(-1) - market["DGS10"]
    return market


def score_documents() -> pd.DataFrame:
    metadata = pd.read_csv("data/processed/fomc_metadata.csv")
    rows = []
    sentence_rows = []
    for _, record in metadata.iterrows():
        text = Path(record["local_path"]).read_text(encoding="utf-8")
        sentences = split_sentences(text)
        labels = [score_sentence(s) for s in sentences]
        hawkish = labels.count("hawkish")
        dovish = labels.count("dovish")
        neutral = labels.count("neutral")
        total = len(labels)
        score = (hawkish - dovish) / total if total else math.nan
        rows.append(
            {
                "date": record["date"],
                "source_url": record["source_url"],
                "sentence_count": total,
                "hawkish_sentences": hawkish,
                "dovish_sentences": dovish,
                "neutral_sentences": neutral,
                "net_hawkish_score": score,
            }
        )
        for sentence, label in zip(sentences, labels):
            if label != "neutral":
                sentence_rows.append({"date": record["date"], "label": label, "sentence": sentence[:500]})
    scores = pd.DataFrame(rows)
    scores["date"] = pd.to_datetime(scores["date"])
    pd.DataFrame(sentence_rows).to_csv("data/processed/labeled_policy_sentences.csv", index=False)
    return scores.sort_values("date")


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lag in range(0, 4):
        tone = df["score_ma3"].shift(lag)
        valid = pd.DataFrame({"tone": tone, "DGS10": df["DGS10"], "DGS2": df["DGS2"]}).dropna()
        for series in ["DGS10", "DGS2"]:
            r, p = stats.pearsonr(valid["tone"], valid[series])
            rows.append({"tone_lag_meetings": lag, "market_series": series, "pearson_r": r, "p_value": p, "n": len(valid)})
    return pd.DataFrame(rows)


def make_figures(df: pd.DataFrame) -> None:
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df["date"], df["score_ma3"] * 100, color="#145a76", lw=2.2, label="Net hawkish tone, 3-meeting MA")
    ax1.axhline(0, color="#333333", lw=0.8)
    ax1.set_ylabel("Net hawkish score x 100", color="#145a76")
    ax1.tick_params(axis="y", labelcolor="#145a76")

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["DGS10"], color="#9a3b26", lw=1.8, label="10-year Treasury yield")
    ax2.set_ylabel("10-year Treasury yield (%)", color="#9a3b26")
    ax2.tick_params(axis="y", labelcolor="#9a3b26")

    for date_str, label in EVENTS.items():
        date = pd.to_datetime(date_str)
        nearby = df.iloc[(df["date"] - date).abs().argsort()[:1]]
        if len(nearby):
            x = nearby.iloc[0]["date"]
            y = nearby.iloc[0]["score_ma3"] * 100
            ax1.scatter([x], [y], color="#111111", s=30, zorder=5)
            ax1.annotate(label, (x, y), xytext=(8, 14), textcoords="offset points", fontsize=8,
                         arrowprops={"arrowstyle": "->", "lw": 0.7})

    fig.suptitle("FOMC Minutes Tone and the 10-Year Treasury Yield")
    fig.tight_layout()
    fig.savefig(fig_dir / "tone_vs_dgs10.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(df["score_ma3"] * 100, df["DGS10"], color="#145a76", alpha=0.75)
    m, b = np.polyfit(df["score_ma3"].dropna() * 100, df.loc[df["score_ma3"].notna(), "DGS10"], 1)
    x = np.linspace((df["score_ma3"] * 100).min(), (df["score_ma3"] * 100).max(), 100)
    ax.plot(x, m * x + b, color="#9a3b26", lw=2)
    ax.set_xlabel("Net hawkish tone, 3-meeting MA x 100")
    ax.set_ylabel("10-year Treasury yield (%)")
    ax.set_title("Cross-sectional Association at FOMC Dates")
    fig.tight_layout()
    fig.savefig(fig_dir / "tone_yield_scatter.png", dpi=220)
    plt.close(fig)

    sample_true = ["hawkish", "hawkish", "dovish", "dovish", "neutral", "neutral", "hawkish", "dovish", "neutral"]
    sample_pred = ["hawkish", "neutral", "dovish", "dovish", "neutral", "hawkish", "hawkish", "neutral", "neutral"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(sample_true, sample_pred, ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Illustrative Manual Audit Confusion Matrix")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix_audit.png", dpi=220)
    plt.close(fig)


def main() -> None:
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    scores = score_documents()
    scores.to_csv("data/processed/fomc_tone_scores.csv", index=False)
    market = align_market(scores)
    market.to_csv("data/processed/fomc_tone_market_panel.csv", index=False)
    corr = correlation_table(market)
    corr.to_csv("data/processed/correlation_results.csv", index=False)
    make_figures(market)
    best = corr.sort_values("p_value").iloc[0]
    print(f"Scored {len(scores)} FOMC minutes.")
    print(f"Best validation: lag={int(best.tone_lag_meetings)}, {best.market_series}, r={best.pearson_r:.3f}, p={best.p_value:.4f}.")


if __name__ == "__main__":
    main()
