#!/usr/bin/env python3
"""Generate the final project report PDF with Matplotlib."""

from __future__ import annotations

import os
import textwrap
from collections.abc import Sequence
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


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/FOMC_Hawkish_Dovish_Final_Report.pdf"
PAGE_SIZE = (8.27, 11.69)
NAVY = "#17324D"
BLUE = "#2D6A8A"
LIGHT_BLUE = "#EAF2F7"
TEXT = "#263238"
MUTED = "#60717B"
Section = tuple[str, str | Sequence[str]]


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(
        " ".join(text.split()),
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _add_page_frame(title: str, page_number: int) -> tuple[plt.Figure, plt.Axes, float]:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.09,
        0.965,
        "FOMC TONE ANALYSIS  /  CS 5100",
        color=BLUE,
        fontsize=8.5,
        fontweight="bold",
        va="top",
    )
    title_lines = _wrap(title, width=52)
    ax.text(
        0.09,
        0.925,
        "\n".join(title_lines),
        color=NAVY,
        fontsize=20,
        fontweight="bold",
        linespacing=1.12,
        va="top",
    )
    rule_y = 0.925 - 0.045 * len(title_lines)
    ax.plot([0.09, 0.91], [rule_y, rule_y], color=BLUE, linewidth=1.4)
    ax.text(0.09, 0.045, "Jinyu Chen  •  Dylan Ullrich", color=MUTED, fontsize=8)
    ax.text(0.91, 0.045, str(page_number), color=MUTED, fontsize=8, ha="right")
    return fig, ax, rule_y - 0.035


def _draw_sections(
    ax: plt.Axes,
    sections: Sequence[Section],
    start_y: float,
    *,
    font_size: float = 10.2,
    wrap_width: int = 91,
    min_y: float = 0.075,
    spacing_scale: float = 1.0,
) -> float:
    y = start_y
    line_height = 0.0205 * (font_size / 10.2)

    for heading, content in sections:
        if heading:
            ax.text(
                0.09,
                y,
                heading.upper(),
                color=BLUE,
                fontsize=9,
                fontweight="bold",
                va="top",
            )
            y -= 0.026 * spacing_scale

        if isinstance(content, str):
            paragraphs = [part for part in content.split("\n\n") if part.strip()]
            for paragraph in paragraphs:
                lines = _wrap(paragraph, wrap_width)
                ax.text(
                    0.09,
                    y,
                    "\n".join(lines),
                    color=TEXT,
                    fontsize=font_size,
                    linespacing=1.35,
                    va="top",
                )
                y -= line_height * len(lines) + 0.014 * spacing_scale
        else:
            for item in content:
                lines = textwrap.wrap(
                    " ".join(str(item).split()),
                    width=wrap_width - 4,
                    initial_indent="•  ",
                    subsequent_indent="   ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                ax.text(
                    0.105,
                    y,
                    "\n".join(lines),
                    color=TEXT,
                    fontsize=font_size,
                    linespacing=1.32,
                    va="top",
                )
                y -= line_height * len(lines) + 0.009 * spacing_scale
        y -= 0.013 * spacing_scale

    if y < min_y:
        raise RuntimeError(f"Page content overflowed the footer area (y={y:.3f})")
    return y


def add_cover_page(pdf: PdfPages, abstract: str, page_number: int) -> None:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0.79), 1, 0.21, color=NAVY))
    ax.text(
        0.09,
        0.95,
        "FOMC TONE ANALYSIS",
        color="#B8D7E8",
        fontsize=9.5,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.09,
        0.895,
        "Predicting Monetary Policy Stance\nfrom FOMC Minutes",
        color="white",
        fontsize=24,
        fontweight="bold",
        linespacing=1.15,
        va="top",
    )
    ax.text(0.09, 0.715, "Jinyu Chen", color=NAVY, fontsize=12, fontweight="bold")
    ax.text(0.09, 0.685, "Dylan Ullrich", color=NAVY, fontsize=12, fontweight="bold")
    ax.text(0.55, 0.715, "CS 5100", color=TEXT, fontsize=11)
    ax.text(0.55, 0.685, "August 12, 2026", color=TEXT, fontsize=11)
    ax.plot([0.09, 0.91], [0.63, 0.63], color="#C8D6DE", linewidth=1)
    ax.text(
        0.09,
        0.58,
        "ABSTRACT",
        color=BLUE,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    abstract_lines = _wrap(abstract, width=78)
    ax.add_patch(plt.Rectangle((0.075, 0.30), 0.85, 0.23, color=LIGHT_BLUE, zorder=0))
    ax.text(
        0.105,
        0.49,
        "\n".join(abstract_lines),
        color=TEXT,
        fontsize=10.5,
        linespacing=1.45,
        va="top",
    )
    ax.text(0.09, 0.045, "Final Project Report", color=MUTED, fontsize=8)
    ax.text(0.91, 0.045, str(page_number), color=MUTED, fontsize=8, ha="right")
    pdf.savefig(fig)
    plt.close(fig)


def add_text_page(
    pdf: PdfPages,
    title: str,
    sections: Sequence[Section],
    page_number: int,
    *,
    font_size: float = 10.2,
) -> None:
    fig, ax, body_y = _add_page_frame(title, page_number)
    _draw_sections(ax, sections, body_y, font_size=font_size)
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(
    pdf: PdfPages,
    title: str,
    image_path: Path,
    sections: Sequence[Section],
    page_number: int,
    *,
    caption_font_size: float = 9.6,
    caption_wrap_width: int = 86,
    caption_spacing_scale: float = 1.0,
) -> None:
    fig, ax, body_y = _add_page_frame(title, page_number)
    image = mpimg.imread(image_path)
    image_ax = fig.add_axes([0.085, 0.40, 0.83, body_y - 0.41])
    image_ax.imshow(image)
    image_ax.axis("off")

    ax.add_patch(plt.Rectangle((0.075, 0.065), 0.85, 0.30, color=LIGHT_BLUE, zorder=0))
    _draw_sections(
        ax,
        sections,
        0.335,
        font_size=caption_font_size,
        wrap_width=caption_wrap_width,
        min_y=0.045,
        spacing_scale=caption_spacing_scale,
    )
    pdf.savefig(fig)
    plt.close(fig)


def result_summaries() -> tuple[str, str, list[str]]:
    panel = pd.read_csv(ROOT / "data/processed/fomc_tone_market_panel.csv")
    corr = pd.read_csv(ROOT / "data/processed/correlation_results.csv")
    metrics = pd.read_csv(ROOT / "data/processed/classifier_metrics.csv")
    examples = pd.read_csv(ROOT / "data/processed/classifier_examples.csv")

    best = corr.loc[corr["pearson_r"].abs().idxmax()]
    same_time_dgs10 = corr[
        (corr["tone_lag_meetings"] == 0) & (corr["market_series"] == "DGS10")
    ].iloc[0]
    market_summary = (
        f"The sample contains {len(panel)} FOMC minutes from {panel['date'].min()} to {panel['date'].max()}. "
        f"The contemporaneous descriptive correlation between the three-meeting-smoothed tone index and "
        f"DGS10 is r={same_time_dgs10.pearson_r:.3f}. Across the eight combinations formed from two Treasury-yield "
        f"series and four tone lags, the largest observed correlation was for {best.market_series} with tone "
        f"lagged by {int(best.tone_lag_meetings)} meetings (r={best.pearson_r:.3f}). This was an exploratory "
        "comparison rather than a prespecified or uniquely validated model specification."
    )

    overall = metrics[metrics["class"] == "overall"].set_index("model")
    audit_summary = (
        "On the independent 30-sentence human audit, the dictionary baseline achieved "
        f"{overall.loc['dictionary', 'accuracy']:.1%} accuracy and {overall.loc['dictionary', 'macro_f1']:.3f} "
        "macro-F1, while the weakly supervised FinBERT model achieved "
        f"{overall.loc['finbert', 'accuracy']:.1%} accuracy and {overall.loc['finbert', 'macro_f1']:.3f} "
        "macro-F1. The observed accuracy difference was one correctly classified sentence, and the audit is too "
        "small to establish a meaningful performance difference."
    )
    example_lines = []
    for row in examples.itertuples(index=False):
        sentence = (
            str(row.sentence).replace("\u00e2\u0080\u0094", " - ").replace("â", " - ")
        )
        sentence = textwrap.shorten(sentence, width=96, placeholder="...")
        example_lines.append(
            f'{row.example_type.replace("_", " ").title()}: "{sentence}" '
            f"(human {row.manual_label}; dictionary {row.dictionary_label}; FinBERT {row.finbert_label})"
        )
    return market_summary, audit_summary, example_lines


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    market_summary, audit_summary, example_summary = result_summaries()
    abstract = (
        "This project builds an AI/NLP pipeline that converts public Federal Reserve FOMC meeting "
        "minutes into a document-level hawkish-dovish policy stance index. It compares a transparent "
        "dictionary baseline with a FinBERT classifier trained from dictionary-generated weak labels. "
        "Both classifiers are evaluated against an independent 30-sentence human audit. The resulting "
        "tone index is also compared descriptively with U.S. Treasury yields across monetary-policy "
        "regimes. The market analysis is exploratory, not causal, and not a minutes-release event study."
    )

    with PdfPages(
        REPORT,
        metadata={
            "Title": "Predicting Monetary Policy Stance from FOMC Minutes",
            "Author": "Jinyu Chen and Dylan Ullrich",
            "Subject": "CS 5100 Final Project Report",
        },
    ) as pdf:
        add_cover_page(pdf, abstract, 1)
        add_text_page(
            pdf,
            "1. Introduction",
            [
                (
                    "Problem statement",
                    "Central-bank communication moves financial markets because minutes and statements "
                    "reveal policymakers' assessment of inflation, employment, financial conditions, and "
                    "future policy risks. The practical challenge is that FOMC minutes are long, qualitative "
                    "documents. A systematic NLP index can make policy tone comparable across meetings and "
                    "across decades.",
                ),
                (
                    "Research goals",
                    [
                        "Estimate whether each FOMC minute is relatively hawkish or dovish.",
                        "Aggregate sentence classifications into a comparable document-level tone index.",
                        "Evaluate the dictionary and FinBERT classifiers against independent human labels.",
                        "Examine whether the tone index comoves descriptively with Treasury yields.",
                    ],
                ),
                (
                    "Expected relationship",
                    "Hawkish language should generally align with higher expected policy rates and higher "
                    "short-to-intermediate yields, while dovish language should align with easing "
                    "expectations. The project contributes a reproducible pipeline using public data, "
                    "interpretable sentence-level labels, and an exploratory market association analysis.",
                ),
            ],
            2,
        )
        add_text_page(
            pdf,
            "2. Background",
            [
                (
                    "Related work",
                    [
                        "Loughran and McDonald (2011) show why domain-specific dictionaries improve "
                        "financial tone measurement over general sentiment dictionaries.",
                        "Lucca and Trebbi (2009) and Hansen and McMahon (2016) show that central-bank text "
                        "contains measurable signals relevant to asset prices and expectations.",
                        "BERT (Devlin et al., 2019) and FinBERT (Araci, 2019) motivate the transformer-model "
                        "comparison.",
                    ],
                ),
                (
                    "AI/ML concepts used",
                    [
                        "Dictionary classification assigns each sentence a hawkish, dovish, or neutral label "
                        "from monetary-policy term counts.",
                        "Document-level aggregation converts sentence labels into a continuous tone index.",
                        "FinBERT is fine-tuned on dictionary-generated labels, a form of weak supervision.",
                        "A blind 30-sentence human audit independently evaluates both classifiers.",
                        "A separate Treasury-yield analysis examines exploratory market association.",
                    ],
                ),
            ],
            3,
        )
        add_text_page(
            pdf,
            "3. Methodology",
            [
                (
                    "Tools and frameworks",
                    [
                        "Data and collection: Python, pandas, NumPy, BeautifulSoup, and requests/curl.",
                        "Modeling and evaluation: scikit-learn, SciPy, PyTorch, and Hugging Face Transformers.",
                        "Reporting and visualization: Matplotlib.",
                    ],
                ),
                (
                    "Data sources and preprocessing",
                    [
                        "147 FOMC minutes from 2008 through June 2026, downloaded from the Board of "
                        "Governors' public meeting pages.",
                        "FRED DGS10 and DGS2 Treasury series, aligned to the latest observation on or before "
                        "each meeting date; the corresponding minutes are released later.",
                        "HTML cleaned to readable text and split into sentences.",
                    ],
                ),
                (
                    "Classification and aggregation",
                    [
                        "Count hawkish terms such as inflation pressure, tightening, restrictive, and rate "
                        "increase; count dovish terms such as accommodation, easing, downside risk, slack, "
                        "and asset purchases.",
                        "Assign the direction with the larger count; classify ties as neutral.",
                        "Compute (hawkish sentences − dovish sentences) / total sentences, then apply a "
                        "three-meeting moving average.",
                        "Fine-tune FinBERT as a three-class comparison using dictionary-generated weak labels "
                        "and evaluate both classifiers against the same independent human audit.",
                    ],
                ),
            ],
            4,
            font_size=9.9,
        )
        add_image_page(
            pdf,
            "4. Results — Main Time Series",
            ROOT / "figures/tone_vs_dgs10.png",
            [
                (
                    "What the figure shows",
                    "The smoothed hawkish-dovish score rises during tightening cycles and shifts around "
                    "major policy regimes.",
                ),
                (
                    "Marked policy regimes",
                    [
                        "2008 zero-rate crisis response",
                        "2015 first post-crisis rate hike",
                        "2020 pandemic emergency easing",
                        "2022 rapid anti-inflation hiking cycle",
                    ],
                ),
            ],
            5,
        )
        add_image_page(
            pdf,
            "4. Results — Exploratory Market Association",
            ROOT / "figures/tone_yield_scatter.png",
            [
                (
                    "Interpretation",
                    "The scatterplot shows a positive descriptive association between the three-meeting-"
                    "smoothed tone index and the 10-year Treasury yield at FOMC meeting dates.",
                ),
                (
                    "Important qualification",
                    "Both measures may vary with broader monetary-policy regimes. The pattern does not "
                    "establish that the minutes caused changes in yields.",
                ),
            ],
            6,
        )
        add_image_page(
            pdf,
            "4. Results — Independent Classifier Evaluation",
            ROOT / "figures/confusion_matrix_audit.png",
            [
                ("Audit result", audit_summary),
                ("Traceable audit examples", example_summary),
            ],
            7,
            caption_font_size=8.1,
            caption_wrap_width=96,
            caption_spacing_scale=0.7,
        )
        add_text_page(
            pdf,
            "5. Discussion",
            [
                (
                    "Classifier evaluation",
                    f"{audit_summary} This illustrates a central limitation of weak supervision: a model "
                    "trained primarily from dictionary-generated labels may reproduce the teacher's "
                    "assumptions without gaining greater agreement with independent human judgments.",
                ),
                (
                    "Market interpretation",
                    f"{market_summary} The larger DGS2 association is consistent with the 2-year yield's "
                    "sensitivity to near-term monetary-policy conditions, but it may also reflect persistent "
                    "common trends and broad policy regimes. Market correlation does not validate "
                    "sentence-label accuracy.",
                ),
                (
                    "Limitations",
                    [
                        "Dictionary models can miss negation, conditional language, and context.",
                        "The human audit is small, stratified by dictionary prediction, and labeled by one "
                        "annotator.",
                        "Three-meeting smoothing creates overlapping observations; tone and yields are "
                        "serially persistent.",
                        "Pearson p-values do not adjust for dependence or selection across eight comparisons.",
                        "Meeting-date yields precede the later minutes releases, so this is neither causal "
                        "analysis nor a minutes-release event study.",
                    ],
                ),
                (
                    "Potential improvements",
                    [
                        "Expand the independently reviewed label set.",
                        "Evaluate statements separately from minutes.",
                        "Study release-date windows and futures surprises.",
                        "Use time-series methods that account for persistence and multiple comparisons.",
                    ],
                ),
            ],
            8,
            font_size=8.9,
        )
        add_text_page(
            pdf,
            "6. Conclusion",
            [
                (
                    "Achievements",
                    [
                        "Built a reproducible workflow for public-data acquisition, sentence-level policy-tone "
                        "classification, document aggregation, independent human-audit evaluation, and "
                        "exploratory market visualization.",
                        "Measured 53.3% dictionary accuracy and 50.0% weakly supervised FinBERT accuracy on "
                        "the 30-sentence audit; the sample is too small to establish a meaningful difference.",
                    ],
                ),
                (
                    "Lessons learned",
                    [
                        "Independent labels are necessary; agreement with dictionary-generated training labels "
                        "is not independent accuracy.",
                        "A pretrained model does not necessarily outperform a transparent baseline when its "
                        "supervision inherits the baseline's assumptions.",
                        "Market association is a separate descriptive analysis, not classifier ground truth.",
                    ],
                ),
                (
                    "Individual contributions — Jinyu Chen",
                    [
                        "Built the original data-collection and preprocessing pipeline, dictionary classifier, "
                        "document-level tone and Treasury-yield analysis, and FinBERT fine-tuning workflow.",
                        "Created the original figures, notebook, report, and presentation materials.",
                    ],
                ),
                (
                    "Individual contributions — Dylan Ullrich",
                    [
                        "Added the reproducible development environment and clarified the saved FinBERT "
                        "artifacts' weak-label provenance.",
                        "Created and manually labeled the 30-sentence audit; implemented classifier evaluation "
                        "and audit-provenance checks.",
                        "Corrected the market-analysis interpretation; filmed the video; and updated the "
                        "README, notebook, report, slides, script, figures, data, and final submission package.",
                    ],
                ),
            ],
            9,
            font_size=9.2,
        )
        add_text_page(
            pdf,
            "7. References",
            [
                (
                    "",
                    [
                        "Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language "
                        "Models. arXiv:1908.10063.",
                        "Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of "
                        "Deep Bidirectional Transformers for Language Understanding. NAACL.",
                        "Hansen, S. and McMahon, M. (2016). Shocking Language: Understanding the "
                        "Macroeconomic Effects of Central Bank Communication. Journal of International "
                        "Economics.",
                        "Loughran, T. and McDonald, B. (2011). When Is a Liability Not a Liability? Textual "
                        "Analysis, Dictionaries, and 10-Ks. Journal of Finance.",
                        "Lucca, D. O. and Trebbi, F. (2009). Measuring Central Bank Communication: An "
                        "Automated Approach with Application to FOMC Statements. NBER Working Paper.",
                        "Board of Governors of the Federal Reserve System. FOMC meeting calendars, statements, "
                        "and minutes. https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
                        "Federal Reserve Bank of St. Louis FRED. DGS10 and DGS2 Treasury Constant Maturity "
                        "Rates. https://fred.stlouisfed.org/",
                    ],
                )
            ],
            10,
        )
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
