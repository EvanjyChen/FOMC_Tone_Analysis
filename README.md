# FOMC Hawkish-Dovish Tone Project

Course: CS5100 Final Project

This project estimates the policy stance embedded in Federal Open Market Committee (FOMC) meeting minutes. It downloads public FOMC minutes from the Federal Reserve, downloads Treasury yields from FRED, classifies sentences with a transparent hawkish-dovish dictionary, aggregates document-level tone scores, and compares tone with the 10-year Treasury yield.

## Project Structure

- `scripts/fetch_data.py`: downloads public FOMC minutes and FRED Treasury yield data.
- `scripts/analyze_tone.py`: sentence splitting, dictionary scoring, aggregation, evaluation, and figure generation.
- `scripts/finbert_optional.py`: optional FinBERT fine-tuning workflow for the advanced model section.
- `scripts/make_report.py`: generates the final PDF report from computed results.
- `scripts/make_slides.py`: generates presentation slides as a PDF.
- `data/raw/`: downloaded raw text/market data.
- `data/processed/`: processed scores and evaluation tables.
- `figures/`: charts used in the report and slides.
- `reports/`: final report PDF.
- `slides/`: presentation slide deck PDF and video script.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The current workspace already has the required libraries installed, so the commands below can also be run directly with `python3`.

## Reproduce Results

```bash
python3 scripts/fetch_data.py --start-year 2008 --end-year 2026
python3 scripts/analyze_tone.py
python3 scripts/make_report.py
python3 scripts/make_slides.py
```

## Data Sources

- FOMC minutes: Board of Governors of the Federal Reserve System, meeting calendars and historical materials.
- Market data: FRED 10-Year Treasury Constant Maturity Rate, series `DGS10`; optional 2-Year Treasury rate, series `DGS2`.

All primary data are public and redistribution-safe for an academic project. The raw text files are downloaded from official public pages, and the source URLs are preserved in `data/processed/fomc_metadata.csv`.

## Methods

The main reproducible model is a dictionary baseline. Each FOMC minute is split into sentences. A sentence is labeled hawkish if hawkish monetary-policy terms outnumber dovish terms, dovish if the opposite is true, and neutral otherwise. The document score is:

```text
net hawkish score = (hawkish sentence count - dovish sentence count) / total sentence count
```

The optional advanced path fine-tunes FinBERT (`ProsusAI/finbert`) as a hawkish/dovish/neutral sentence classifier. It supports reviewed labels through a CSV with `sentence,label` columns. If reviewed labels are not available, it can build a balanced weak-supervision training set from the dictionary classifier:

```bash
python3 scripts/finbert_optional.py --epochs 2 --batch-size 8
```

The script saves the fine-tuned model under `models/finbert_policy_tone/`, exports validation metrics, writes `data/processed/finbert_sentence_predictions.csv`, and aggregates `data/processed/finbert_document_scores.csv`.

## Deliverables

Generated files:

- `reports/FOMC_Hawkish_Dovish_Final_Report.pdf`
- `slides/FOMC_Hawkish_Dovish_Slides.pdf`
- `slides/video_script.md`
- `AI_Project_FOMC_Tone.zip`
