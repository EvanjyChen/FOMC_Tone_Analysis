#!/usr/bin/env python3
"""Generate the final project report PDF with Matplotlib."""

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


REPORT = Path("reports/FOMC_Hawkish_Dovish_Final_Report.pdf")


def add_text_page(pdf: PdfPages, title: str, body: str, footer: str = "") -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.08, 0.94, title, fontsize=18, fontweight="bold", va="top")
    wrapped_lines: list[str] = []
    for paragraph in body.strip().split("\n\n"):
        wrapped_lines.extend(textwrap.wrap(paragraph, width=92))
        wrapped_lines.append("")
    ax.text(0.08, 0.88, "\n".join(wrapped_lines), fontsize=10.5, va="top", linespacing=1.35)
    if footer:
        ax.text(0.08, 0.05, footer, fontsize=8.5, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: str, caption: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.08, 0.94, title, fontsize=18, fontweight="bold", va="top")
    image = mpimg.imread(image_path)
    image_ax = fig.add_axes([0.08, 0.28, 0.84, 0.55])
    image_ax.imshow(image)
    image_ax.axis("off")
    ax.text(0.08, 0.20, "\n".join(textwrap.wrap(caption, width=95)), fontsize=10.5, va="top", linespacing=1.35)
    pdf.savefig(fig)
    plt.close(fig)


def metric_summary() -> tuple[str, str]:
    panel = pd.read_csv("data/processed/fomc_tone_market_panel.csv")
    corr = pd.read_csv("data/processed/correlation_results.csv")
    best = corr.sort_values("p_value").iloc[0]
    same_time_dgs10 = corr[(corr["tone_lag_meetings"] == 0) & (corr["market_series"] == "DGS10")].iloc[0]
    summary = (
        f"The sample contains {len(panel)} FOMC minutes from {panel['date'].min()} to {panel['date'].max()}. "
        f"The contemporaneous correlation between the smoothed net hawkish score and DGS10 is "
        f"r={same_time_dgs10.pearson_r:.3f} with p={same_time_dgs10.p_value:.2e}. "
        f"The strongest specification is {best.market_series} with tone lagged by "
        f"{int(best.tone_lag_meetings)} meetings: r={best.pearson_r:.3f}, p={best.p_value:.2e}."
    )
    table = corr.to_string(index=False, formatters={"pearson_r": "{:.3f}".format, "p_value": "{:.2e}".format})
    return summary, table


def main() -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    summary, corr_table = metric_summary()
    with PdfPages(REPORT) as pdf:
        add_text_page(
            pdf,
            "FOMC Minutes Hawkish-Dovish Tone Analysis",
            """Project Title: Predicting Monetary Policy Stance from FOMC Minutes with Pretrained-Language-Model-Inspired Tone Analysis

Student/Team: Your Name or Team Name
Course: CS5100 Final Project
Date: August 3, 2026

Abstract: This project builds an AI/NLP pipeline that converts public Federal Reserve FOMC meeting minutes into a document-level hawkish-dovish policy stance index. The core reproducible model is a sentence-level dictionary classifier based on monetary-policy hawkish and dovish terms. The project also documents a FinBERT extension path for a pretrained financial language model. The resulting tone index is validated against U.S. Treasury yields from FRED, with particular attention to crisis and tightening episodes including 2008, 2020, and 2022.""",
        )
        add_text_page(
            pdf,
            "1. Introduction",
            """Problem statement: Central-bank communication moves financial markets because minutes and statements reveal policymakers' assessment of inflation, employment, financial conditions, and future policy risks. The practical challenge is that FOMC minutes are long, qualitative documents. A systematic NLP index can make policy tone comparable across meetings and across decades.

Motivation and goals: The goal is to estimate whether each FOMC minute is relatively hawkish or dovish, then test whether that index comoves with Treasury yields. Hawkish language should generally align with higher expected policy rates and higher short-to-intermediate yields, while dovish language should align with easing expectations. The project contributes a reproducible pipeline using public data, interpretable sentence-level labels, and market validation.""",
        )
        add_text_page(
            pdf,
            "2. Background",
            """Related work: Financial text analysis often begins with specialized dictionaries because general sentiment dictionaries misclassify financial language. Loughran and McDonald (2011) show that domain-specific word lists improve financial tone measurement. Central-bank communication studies, including Lucca and Trebbi (2009) and Hansen and McMahon (2016), show that monetary-policy text contains measurable signals relevant to asset prices and expectations. Transformer models such as BERT (Devlin et al., 2019) and FinBERT (Araci, 2019) motivate the advanced model extension.

AI/ML concepts used: The implemented baseline is text classification at the sentence level. Each sentence is converted to a hawkish, dovish, or neutral label through a transparent rule-based classifier. Document-level aggregation then creates a continuous index. Evaluation uses correlation analysis and lag analysis against market variables. The optional advanced path uses transfer learning with FinBERT for sentence classification once reviewed labels are available.""",
        )
        add_text_page(
            pdf,
            "3. Methodology",
            """Tools and frameworks: Python, pandas, NumPy, BeautifulSoup, requests/curl, scikit-learn, SciPy, Matplotlib, PyTorch, and Hugging Face Transformers.

Data sources and preprocessing: FOMC minutes are downloaded from the Board of Governors of the Federal Reserve System public meeting calendar and historical pages. The main sample begins in 2008 and runs through the available 2026 meetings. Treasury yield data are downloaded from FRED: DGS10 for the 10-year Treasury Constant Maturity Rate and DGS2 for the 2-year rate. HTML pages are stripped to readable text, split into sentences, and aligned with the most recent available daily Treasury yield on or before the meeting date.

Algorithm: For each sentence, the classifier counts hawkish terms such as inflation pressure, tightening, restrictive, and rate increase, and dovish terms such as accommodation, easing, downside risk, slack, and asset purchases. If hawkish counts exceed dovish counts, the sentence is hawkish; if dovish counts exceed hawkish counts, it is dovish; otherwise neutral. The document score is (hawkish sentences - dovish sentences) / total sentences. A 3-meeting moving average reduces meeting-to-meeting noise.""",
        )
        add_image_page(
            pdf,
            "4. Results: Main Time Series",
            "figures/tone_vs_dgs10.png",
            "The smoothed hawkish-dovish score rises during tightening cycles and shifts around major policy regimes. The chart marks the zero-rate crisis response in 2008, the first post-crisis rate hike in 2015, the pandemic emergency easing in 2020, and the rapid anti-inflation hiking cycle in 2022.",
        )
        add_image_page(
            pdf,
            "4. Results: Tone-Yield Association",
            "figures/tone_yield_scatter.png",
            "The scatterplot shows a positive relationship between the FOMC tone index and the 10-year Treasury yield. This is consistent with the interpretation that hawkish communication is associated with higher expected policy rates and higher yields.",
        )
        add_text_page(
            pdf,
            "4. Results: Quantitative Validation",
            summary + "\n\nCorrelation table:\n" + corr_table,
        )
        add_text_page(
            pdf,
            "5. Discussion",
            """Interpretation: The results support the central hypothesis: a simple NLP tone index extracted from FOMC minutes contains information related to Treasury yields. The relationship is stronger for the 2-year yield than for the 10-year yield, which is economically intuitive because the 2-year rate is more sensitive to expected monetary policy over the near horizon.

Limitations: Dictionary models are transparent but imperfect. They may misclassify negation, conditional language, and sentences where inflation is mentioned in a dovish context. Meeting minutes are released after the meeting, so the empirical validation is a tone-market association rather than a pure real-time trading signal. The optional FinBERT model requires carefully reviewed hawkish/dovish labels because generic financial positive/negative sentiment is not identical to monetary-policy stance.

Potential improvements: Add a manually audited training set, fine-tune FinBERT for hawkish/dovish/neutral classification, evaluate statements separately from minutes, include federal funds futures surprises, and run event-window regressions around release dates.""",
        )
        add_text_page(
            pdf,
            "6. Conclusion",
            """Summary of achievements: The project delivers a complete AI/NLP workflow: public data acquisition, sentence-level policy-tone classification, document-level aggregation, visualization across major monetary-policy regimes, and statistical validation against Treasury yields. The code is structured so results can be reproduced with one command sequence.

Lessons learned: Domain-specific financial language requires domain-specific labels or dictionaries. A transparent baseline is valuable because it makes model behavior explainable, while pretrained language models offer a natural extension for handling context and negation.

Individual contributions: Replace this paragraph with each team member's role before final submission. Suggested categories are data collection, modeling, evaluation, report writing, and presentation/video production.""",
        )
        add_text_page(
            pdf,
            "7. References",
            """Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.

Hansen, S. and McMahon, M. (2016). Shocking Language: Understanding the Macroeconomic Effects of Central Bank Communication. Journal of International Economics.

Loughran, T. and McDonald, B. (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. Journal of Finance.

Lucca, D. O. and Trebbi, F. (2009). Measuring Central Bank Communication: An Automated Approach with Application to FOMC Statements. NBER Working Paper.

Board of Governors of the Federal Reserve System. FOMC meeting calendars, statements, and minutes. https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

Federal Reserve Bank of St. Louis FRED. DGS10 and DGS2 Treasury Constant Maturity Rates. https://fred.stlouisfed.org/""",
        )
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
