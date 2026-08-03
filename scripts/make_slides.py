#!/usr/bin/env python3
"""Generate presentation slides PDF and a short video script."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


SLIDES = Path("slides/FOMC_Hawkish_Dovish_Slides.pdf")
SCRIPT = Path("slides/video_script.md")


def slide_text(pdf: PdfPages, title: str, bullets: list[str]) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.06, 0.90, title, fontsize=28, fontweight="bold", va="top")
    y = 0.75
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=92)
        ax.text(0.08, y, "- " + wrapped[0], fontsize=18, va="top")
        y -= 0.08
        for line in wrapped[1:]:
            ax.text(0.105, y, line, fontsize=18, va="top")
            y -= 0.07
        y -= 0.03
    pdf.savefig(fig)
    plt.close(fig)


def slide_image(pdf: PdfPages, title: str, image_path: str, note: str) -> None:
    fig = plt.figure(figsize=(13.33, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.05, 0.93, title, fontsize=26, fontweight="bold", va="top")
    image = mpimg.imread(image_path)
    image_ax = fig.add_axes([0.06, 0.14, 0.70, 0.70])
    image_ax.imshow(image)
    image_ax.axis("off")
    ax.text(0.79, 0.72, "\n".join(textwrap.wrap(note, width=28)), fontsize=17, va="top", linespacing=1.35)
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    SLIDES.parent.mkdir(parents=True, exist_ok=True)
    corr = pd.read_csv("data/processed/correlation_results.csv")
    best = corr.sort_values("p_value").iloc[0]
    with PdfPages(SLIDES) as pdf:
        slide_text(pdf, "Predicting FOMC Policy Stance from Minutes", [
            "Question: Can public FOMC minutes be converted into a measurable hawkish-dovish policy stance index?",
            "Data: Federal Reserve minutes from 2008-2026 and FRED Treasury yields.",
            "Method: sentence-level NLP classification plus document-level aggregation.",
        ])
        slide_text(pdf, "Pipeline", [
            "Download official FOMC minutes and Treasury yield CSV files.",
            "Clean HTML text, split minutes into sentences, and classify each sentence.",
            "Aggregate net hawkish score and compare it with DGS10 and DGS2.",
        ])
        slide_text(pdf, "Sentence Classifier", [
            "Hawkish examples: inflation pressure, tightening, restrictive policy, rate increase.",
            "Dovish examples: accommodation, easing, downside risk, slack, asset purchases.",
            "Score = (hawkish sentences - dovish sentences) / total sentences.",
        ])
        slide_image(pdf, "Main Result", "figures/tone_vs_dgs10.png",
                    "The tone index shifts around crisis and tightening regimes, including 2008, 2020, and 2022.")
        slide_image(pdf, "Market Validation", "figures/tone_yield_scatter.png",
                    "The smoothed hawkish score is positively related to the 10-year Treasury yield.")
        slide_text(pdf, "Quantitative Evidence", [
            f"Strongest validation: {best.market_series}, tone lagged by {int(best.tone_lag_meetings)} meetings.",
            f"Pearson correlation r = {best.pearson_r:.3f}; p-value = {best.p_value:.2e}.",
            "The 2-year yield relationship is strongest, consistent with near-term policy sensitivity.",
        ])
        slide_text(pdf, "Conclusion", [
            "FOMC minutes contain measurable policy-tone information.",
            "The dictionary model is transparent and reproducible.",
            "FinBERT fine-tuning is the natural next step for context, negation, and stronger sentence labels.",
        ])

    SCRIPT.write_text(
        """# 5-7 Minute Video Script

1. Introduce the problem: FOMC minutes are long qualitative documents, but markets respond to monetary-policy tone.
2. Explain the data: public Federal Reserve minutes from 2008-2026 and FRED DGS10/DGS2 Treasury yields.
3. Walk through the pipeline: download, clean, sentence split, classify, aggregate, align with market data.
4. Explain the model: hawkish and dovish dictionary terms classify sentences; net hawkish score summarizes each meeting.
5. Present the main chart: point out 2008 crisis easing, 2020 pandemic easing, and 2022 anti-inflation tightening.
6. Present validation: positive correlation with Treasury yields, strongest for DGS2, which is more policy-sensitive.
7. Discuss limitations and FinBERT extension: dictionary is interpretable but misses context; fine-tuning FinBERT can improve sentence labels.
8. Conclude: the project demonstrates an end-to-end AI/NLP application using public data and market validation.
""",
        encoding="utf-8",
    )
    print(f"Wrote {SLIDES} and {SCRIPT}")


if __name__ == "__main__":
    main()
