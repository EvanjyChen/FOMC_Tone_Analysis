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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / "slides/FOMC_Hawkish_Dovish_Slides.pdf"
SCRIPT = ROOT / "slides/video_script.md"
SLIDE_SIZE = (13.33, 7.5)
NAVY = "#17324D"
BLUE = "#2D6A8A"
LIGHT_BLUE = "#EAF2F7"
TEXT = "#263238"
MUTED = "#60717B"


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        " ".join(text.split()),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _slide_frame(title: str, slide_number: int) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.972), 1, 0.028, color=NAVY))
    ax.text(
        0.06,
        0.935,
        "FOMC TONE ANALYSIS  /  CS 5100",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    ax.text(0.06, 0.865, title, color=NAVY, fontsize=27, fontweight="bold", va="top")
    ax.plot([0.06, 0.94], [0.79, 0.79], color=BLUE, linewidth=1.5)
    ax.text(0.06, 0.035, "Jinyu Chen  •  Dylan Ullrich", color=MUTED, fontsize=9)
    ax.text(0.94, 0.035, str(slide_number), color=MUTED, fontsize=9, ha="right")
    return fig, ax


def slide_title(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=SLIDE_SIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 0.59, 1, color=NAVY))
    ax.text(
        0.065,
        0.88,
        "FOMC TONE ANALYSIS",
        color="#B8D7E8",
        fontsize=11,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.065,
        0.77,
        "Predicting Monetary\nPolicy Stance from\nFOMC Minutes",
        color="white",
        fontsize=32,
        fontweight="bold",
        linespacing=1.12,
        va="top",
    )
    ax.text(0.065, 0.19, "Jinyu Chen  •  Dylan Ullrich", color="white", fontsize=14)
    ax.text(0.065, 0.13, "CS 5100  /  August 12, 2026", color="#B8D7E8", fontsize=11)

    ax.text(0.65, 0.86, "RESEARCH QUESTION", color=BLUE, fontsize=10, fontweight="bold")
    ax.text(
        0.65,
        0.80,
        "\n".join(
            _wrap(
                "Can public FOMC minutes be converted into a measurable hawkish-dovish policy stance index?",
                28,
            )
        ),
        color=NAVY,
        fontsize=18,
        fontweight="bold",
        linespacing=1.25,
        va="top",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.63, 0.27),
            0.30,
            0.30,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            facecolor=LIGHT_BLUE,
            edgecolor="none",
        )
    )
    ax.text(
        0.66, 0.51, "PROJECT CONTRIBUTIONS", color=BLUE, fontsize=10, fontweight="bold"
    )
    contributions = [
        "Reproducible NLP pipeline",
        "Independent human audit",
        "Exploratory market comparison",
    ]
    for index, contribution in enumerate(contributions):
        y = 0.445 - index * 0.072
        ax.scatter([0.665], [y + 0.006], s=26, color=BLUE)
        ax.text(0.69, y, contribution, color=TEXT, fontsize=13, va="center")
    ax.text(0.94, 0.035, "1", color=MUTED, fontsize=9, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def slide_text(
    pdf: PdfPages,
    title: str,
    bullets: list[str],
    slide_number: int,
    *,
    callout: str | None = None,
) -> None:
    fig, ax = _slide_frame(title, slide_number)
    y = 0.705
    for bullet in bullets:
        wrapped = _wrap(bullet, width=78)
        ax.scatter([0.078], [y - 0.008], s=38, color=BLUE)
        ax.text(
            0.105,
            y,
            "\n".join(wrapped),
            color=TEXT,
            fontsize=17,
            linespacing=1.3,
            va="top",
        )
        y -= 0.066 * len(wrapped) + 0.052

    if callout:
        ax.add_patch(
            FancyBboxPatch(
                (0.07, 0.105),
                0.86,
                0.105,
                boxstyle="round,pad=0.015,rounding_size=0.014",
                facecolor=LIGHT_BLUE,
                edgecolor="none",
            )
        )
        ax.text(
            0.10,
            0.158,
            callout,
            color=NAVY,
            fontsize=15,
            fontweight="bold",
            va="center",
        )
    pdf.savefig(fig)
    plt.close(fig)


def slide_image(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    note: str,
    slide_number: int,
    *,
    examples: list[str] | None = None,
) -> None:
    fig, ax = _slide_frame(title, slide_number)
    image = mpimg.imread(image_path)
    image_box = [0.065, 0.34, 0.87, 0.42] if examples else [0.065, 0.19, 0.87, 0.56]
    image_ax = fig.add_axes(image_box)
    image_ax.imshow(image)
    image_ax.axis("off")
    if examples:
        y = 0.305
        for example in examples:
            wrapped = _wrap(example, width=124)
            ax.text(
                0.08,
                y,
                "\n".join(wrapped),
                color=TEXT,
                fontsize=11,
                linespacing=1.25,
                va="top",
            )
            y -= 0.036 * len(wrapped) + 0.010
    ax.add_patch(
        FancyBboxPatch(
            (0.075, 0.075),
            0.85,
            0.075,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=LIGHT_BLUE,
            edgecolor="none",
        )
    )
    ax.text(
        0.5,
        0.112,
        "\n".join(_wrap(note, width=112)),
        color=TEXT,
        fontsize=12.5,
        ha="center",
        va="center",
        linespacing=1.2,
    )
    pdf.savefig(fig)
    plt.close(fig)


def slide_pipeline(pdf: PdfPages, slide_number: int) -> None:
    fig, ax = _slide_frame("Pipeline and Models", slide_number)
    stages = [
        ("01", "Data", "147 FOMC minutes\nDGS10 / DGS2 yields"),
        ("02", "Preprocess", "Clean HTML\nSplit into sentences"),
        ("03", "Models", "Dictionary baseline\nWeak-label FinBERT"),
        ("04", "Evidence", "30-sentence audit\nTreasury comparison"),
    ]
    colors = ["#DDECF5", "#E5F1EA", "#F5EBDD", "#ECE5F3"]
    box_width = 0.195
    for index, ((number, stage, details), color) in enumerate(zip(stages, colors)):
        x = 0.06 + index * 0.225
        box = FancyBboxPatch(
            (x, 0.36),
            box_width,
            0.32,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            facecolor=color,
            edgecolor="#B5C2CA",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.018, 0.635, number, color=BLUE, fontsize=10, fontweight="bold")
        ax.text(
            x + box_width / 2,
            0.565,
            stage,
            color=NAVY,
            fontsize=18,
            fontweight="bold",
            ha="center",
        )
        ax.text(
            x + box_width / 2,
            0.455,
            details,
            color=TEXT,
            fontsize=12.5,
            ha="center",
            va="center",
            linespacing=1.35,
        )
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_width + 0.006, 0.52),
                    (x + 0.218, 0.52),
                    arrowstyle="-|>",
                    mutation_scale=17,
                    linewidth=1.4,
                    color=BLUE,
                )
            )

    ax.add_patch(
        FancyBboxPatch(
            (0.13, 0.13),
            0.74,
            0.13,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor=LIGHT_BLUE,
            edgecolor="none",
        )
    )
    ax.text(
        0.5,
        0.215,
        "PRINCIPAL DICTIONARY-DERIVED TONE INDEX",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
        ha="center",
    )
    ax.text(
        0.5,
        0.16,
        "100 × (hawkish − dovish) / all sentences   →   three-meeting moving average",
        color=NAVY,
        fontsize=16,
        fontweight="bold",
        ha="center",
    )
    pdf.savefig(fig)
    plt.close(fig)


def slide_evaluation(
    pdf: PdfPages,
    audit_n: int,
    dictionary_correct: int,
    finbert_correct: int,
    overall: pd.DataFrame,
    slide_number: int,
) -> None:
    fig, ax = _slide_frame("Independent Classifier Evaluation", slide_number)
    cards = [
        (
            0.08,
            "DICTIONARY",
            overall.loc["dictionary", "accuracy"],
            dictionary_correct,
            overall.loc["dictionary", "macro_f1"],
            "#DDECF5",
        ),
        (
            0.51,
            "WEAKLY SUPERVISED FINBERT",
            overall.loc["finbert", "accuracy"],
            finbert_correct,
            overall.loc["finbert", "macro_f1"],
            "#ECE5F3",
        ),
    ]
    for x, label, accuracy, correct, macro_f1, color in cards:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.39),
                0.39,
                0.30,
                boxstyle="round,pad=0.018,rounding_size=0.018",
                facecolor=color,
                edgecolor="none",
            )
        )
        ax.text(x + 0.03, 0.64, label, color=BLUE, fontsize=10, fontweight="bold")
        ax.text(
            x + 0.03,
            0.52,
            f"{accuracy:.1%}",
            color=NAVY,
            fontsize=31,
            fontweight="bold",
        )
        ax.text(
            x + 0.20,
            0.535,
            f"{correct}/{audit_n} correct\nMacro-F1  {macro_f1:.3f}",
            color=TEXT,
            fontsize=13,
            linespacing=1.35,
            va="center",
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.08, 0.105),
            0.84,
            0.185,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor=LIGHT_BLUE,
            edgecolor="none",
        )
    )
    ax.text(
        0.105,
        0.25,
        "TAKEAWAY",
        color=BLUE,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        0.105,
        0.20,
        "One prediction separates the models—too little to establish meaningful superiority.",
        color=NAVY,
        fontsize=16,
        fontweight="bold",
    )
    ax.text(
        0.105,
        0.16,
        "Both models beat the 33.3% chance baseline for three balanced classes, but not by a wide margin.",
        color=TEXT,
        fontsize=11.5,
    )
    ax.text(
        0.105,
        0.125,
        "The small, single-annotator audit is preliminary; weak supervision may reproduce the dictionary's assumptions.",
        color=TEXT,
        fontsize=11.5,
    )
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    SLIDES.parent.mkdir(parents=True, exist_ok=True)
    corr = pd.read_csv(ROOT / "data/processed/correlation_results.csv")
    metrics = pd.read_csv(ROOT / "data/processed/classifier_metrics.csv")
    best = corr.loc[corr["pearson_r"].abs().idxmax()]
    examples = pd.read_csv(ROOT / "data/processed/classifier_examples.csv").set_index(
        "example_type"
    )

    def audit_example(example_type: str) -> str:
        row = examples.loc[example_type]
        sentence = " ".join(str(row["sentence"]).split())
        if len(sentence) > 150:
            sentence = "…" + sentence[-150:].split(" ", 1)[1]
        return f'{row["manual_label"].upper()} ({row["date"]}):  “{sentence}”'

    overall = metrics[metrics["class"] == "overall"].set_index("model")
    audit_n = int(overall.loc["dictionary", "support"])
    dictionary_correct = round(overall.loc["dictionary", "accuracy"] * audit_n)
    finbert_correct = round(overall.loc["finbert", "accuracy"] * audit_n)
    with PdfPages(
        SLIDES,
        metadata={
            "Title": "Predicting Monetary Policy Stance from FOMC Minutes",
            "Author": "Jinyu Chen and Dylan Ullrich",
            "Subject": "CS 5100 Final Project Presentation",
        },
    ) as pdf:
        slide_title(pdf)
        slide_pipeline(pdf, 2)
        slide_evaluation(
            pdf,
            audit_n,
            dictionary_correct,
            finbert_correct,
            overall,
            3,
        )
        slide_image(
            pdf,
            "Audit Confusion Matrices",
            ROOT / "figures/confusion_matrix_audit.png",
            "Independent human labels provide classifier ground truth. Both models were evaluated on "
            "the same 30 audited sentences.",
            4,
            examples=[audit_example("hawkish"), audit_example("dovish")],
        )
        slide_image(
            pdf,
            "Tone Across Policy Regimes",
            ROOT / "figures/tone_vs_dgs10.png",
            "The three-meeting-smoothed, dictionary-derived tone index shifts around crisis, "
            "normalization, pandemic, and tightening regimes. The red series is the 10-year yield, "
            "compared directly on the next slide.",
            5,
        )
        slide_image(
            pdf,
            "Tone and the 10-Year Treasury Yield",
            ROOT / "figures/tone_yield_scatter.png",
            "Tone and the 10-year yield move together across broad policy regimes. The association is "
            "descriptive; the full caveats follow on the next slide.",
            6,
        )
        slide_text(
            pdf,
            "Sensitivity Across Yields and Lags",
            [
                f"Two yields × four tone lags were screened. The largest observed correlation was "
                f"{best.market_series} at lag {int(best.tone_lag_meetings)}: "
                f"r = {best.pearson_r:.3f} (n = {int(best['n'])}).",
                "The maximum reflects specification selection across eight yield-and-lag combinations.",
                "Overlapping smoothing, persistence, and common policy-regime trends may inflate the "
                "association.",
            ],
            7,
            callout="Not classifier validation, a causal result, or a minutes-release event study.",
        )
        slide_text(
            pdf,
            "Conclusion",
            [
                "The project delivers an end-to-end, reproducible monetary-policy text-classification "
                "workflow.",
                f"On the 30-sentence audit, the dictionary made {dictionary_correct} correct predictions "
                f"versus FinBERT's {finbert_correct}; this cannot establish a meaningful difference.",
                "Independent labels are essential when a comparison model learns from weak labels "
                "generated by the baseline.",
            ],
            8,
            callout="Classifier accuracy, teacher-label agreement, and market association are different evidence.",
        )

    SCRIPT.write_text(
        f"""# 5–7 Minute Video Script

## Slide 1 — Predicting Monetary Policy Stance from FOMC Minutes

Hello. We are Jinyu Chen and Dylan Ullrich, and our CS 5100 project is Predicting Monetary Policy Stance from FOMC Minutes. Federal Open Market Committee minutes are lengthy qualitative records, but they contain important language about inflation, employment, financial conditions, and the direction of monetary policy. We ask whether public minutes can be converted into a reproducible hawkish-dovish policy stance index, following a literature that measures central-bank communication tone with dictionary and transformer methods; our report situates the project with full citations. The project contributes an end-to-end pipeline that processes official minutes, classifies sentence-level stance, aggregates the results by meeting, independently audits the classifiers, and explores how the index moves across policy regimes and alongside Treasury yields.

## Slide 2 — Pipeline and Models

The pipeline has four stages. First, the data are 147 official FOMC minutes from 2008 through June 2026 and FRED's DGS10 and DGS2 Treasury yield series. Second, we clean the minutes' HTML and split each document into sentences. Third, our transparent dictionary counts domain-specific hawkish and dovish terms. A sentence is hawkish or dovish when one count is larger and neutral when the counts tie. We also fine-tuned ProsusAI's FinBERT as a three-class comparison model using labels generated by that dictionary, so its training is weak supervision rather than independent human supervision. Fourth, the evidence includes a blind human audit and an exploratory Treasury comparison. Our principal tone index is dictionary-derived: 100 times hawkish sentences minus dovish sentences, divided by all sentences. Neutral sentences therefore remain in the denominator. We plot a three-meeting moving average to reduce meeting-to-meeting noise.

## Slide 3 — Independent Classifier Evaluation

To evaluate actual classifier performance, we created a blind human audit of {audit_n} sentences. The sample was stratified across ten dictionary-predicted hawkish, ten dovish, and ten neutral sentences, spread across documents and years, and excluded sentences used in FinBERT training. The dictionary made {dictionary_correct} correct predictions out of {audit_n}, or {overall.loc["dictionary", "accuracy"]:.1%} accuracy, with {overall.loc["dictionary", "macro_f1"]:.3f} macro-F1. FinBERT made {finbert_correct} correct predictions, or {overall.loc["finbert", "accuracy"]:.1%} accuracy, with {overall.loc["finbert", "macro_f1"]:.3f} macro-F1. Both models beat the 33.3 percent chance baseline for three balanced classes, but not by a wide margin. One prediction separates the models. That difference is too small to establish meaningful performance superiority. The weak-supervision lesson still matters: a complex model trained from dictionary labels may inherit the teacher's assumptions rather than improve agreement with human judgment. Because the audit is small and uses one annotator, these scores are preliminary.

## Slide 4 — Audit Confusion Matrices

These confusion matrices show where each model agreed and disagreed with the human labels on the same {audit_n} audited sentences. Neutral was the strongest class for both models, while directional hawkish and dovish language was more difficult. The audited sentences below the matrices show the kind of directional language the classifiers must detect. For example, "inflation firmed" and participants anticipating movement toward the two percent objective mark the hawkish sentence. The project saves traceable examples like these, so every displayed prediction maps to a specific audit row. This human audit is our classifier evaluation. Agreement with dictionary-generated training labels is only teacher-label agreement, and market correlation is separate evidence.

## Slide 5 — Tone Across Policy Regimes

This chart plots the dictionary-derived principal tone index after the three-meeting moving average shown in the pipeline. The series shifts around major policy regimes: crisis easing in 2008, normalization around 2015, pandemic emergency easing in 2020, and rapid anti-inflation tightening beginning in 2022. The red series on the chart is the 10-year Treasury yield, which the next slide compares directly. These patterns show how the pipeline converts long documents into a compact, interpretable time series. They do not prove that every sentence label is correct, which is why the independent human audit is necessary.

## Slide 6 — Tone and the 10-Year Treasury Yield

We align Treasury yields to each FOMC meeting date using the latest available observation on or before that date. The scatterplot shows a positive descriptive association between the smoothed, dictionary-derived tone index and the 10-year yield. This is consistent with both measures varying across broad monetary-policy regimes. The association is descriptive, and the next slide covers the caveats in full.

## Slide 7 — Sensitivity Across Yields and Lags

We examined two yield series, DGS10 and DGS2, at tone lags from zero through three meetings. The largest observed correlation was {best.pearson_r:.3f} for {best.market_series} with tone lagged by {int(best.tone_lag_meetings)} meetings, using n = {int(best["n"])} observations. This maximum was selected from eight yield-and-lag specifications, so it is exploratory rather than uniquely validated. The tone measure uses overlapping three-meeting smoothing, both tone and yields are persistent, and common policy-regime trends may move both series. Those features can inflate an association. The result is not causal, and because the minutes are published only after each meeting, it is not a minutes-release event study. It is also not classifier validation: sentence-label accuracy is evaluated only by the human audit.

## Slide 8 — Conclusion

In conclusion, the project delivers a reproducible NLP workflow using public data, a transparent dictionary baseline, a weakly supervised FinBERT comparison, independent human-audit metrics, and exploratory market visualizations. On the small audit, the dictionary made {dictionary_correct} correct predictions versus FinBERT's {finbert_correct}. A one-prediction difference in {audit_n} sentences cannot establish a meaningful performance difference. The broader lesson is that model complexity does not guarantee better independent performance when supervision comes from the baseline itself. Classifier accuracy, teacher-label agreement, and market association are different forms of evidence and must be reported separately. Future work should expand independently reviewed labels and use stronger release-date and time-series designs. Thank you.
""",
        encoding="utf-8",
    )
    print(f"Wrote {SLIDES} and {SCRIPT}")


if __name__ == "__main__":
    main()
