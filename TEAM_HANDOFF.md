# CS5100 Final Project Handoff

Project: Predicting Monetary Policy Stance from FOMC Minutes  
Topic: FOMC hawkish-dovish tone analysis using public Federal Reserve text and Treasury yields.

## Current Status

The project is mostly complete and ready for review/submission.

Main deliverable zip:
`AI_Project_FOMC_Tone.zip`

Important files:

- `reports/FOMC_Hawkish_Dovish_Final_Report.pdf`
- `notebooks/FOMC_Tone_Analysis.ipynb`
- `slides/FOMC_Hawkish_Dovish_Slides.pdf`
- `slides/video_script.md`
- `README.md`
- `scripts/fetch_data.py`
- `scripts/analyze_tone.py`
- `scripts/finbert_optional.py`

## What the Project Does

1. Downloads public FOMC Minutes from the Federal Reserve website.
2. Downloads Treasury yield data from FRED:
   - `DGS10`: 10-year Treasury yield
   - `DGS2`: 2-year Treasury yield
3. Splits FOMC minutes into sentences.
4. Classifies sentences as hawkish, dovish, or neutral using a monetary-policy dictionary.
5. Aggregates sentence labels into a document-level net hawkish score.
6. Compares the tone score with Treasury yields.
7. Generates figures, report PDF, slides PDF, and video script.

## Key Result

Sample size: 147 FOMC minutes from 2008-2026.

Strongest validation result:

- Market series: `DGS2`
- Tone lag: 2 meetings
- Pearson correlation: `r = 0.806`
- p-value: `< 0.001`

Main interpretation:
The hawkish-dovish tone score is positively related to Treasury yields, especially the 2-year yield, which is more sensitive to expected monetary policy.

## Main Figure

Core chart:
`figures/tone_vs_dgs10.png`

It shows:

- Blue line: net hawkish FOMC tone score
- Red line: 10-year Treasury yield
- Annotated events:
  - 2008 zero-rate crisis response
  - 2015 first post-crisis hike
  - 2020 pandemic emergency easing
  - 2022 rapid anti-inflation hikes

## How to Reproduce

From the project root:

```bash
python3 scripts/fetch_data.py --start-year 2008 --end-year 2026
python3 scripts/analyze_tone.py
python3 scripts/make_report.py
python3 scripts/make_slides.py
```

If dependencies are missing:

```bash
pip install -r requirements.txt
```

## Important Note Before Submission

The report still has placeholder text that must be updated:

1. Replace `Your Name or Team Name` with actual team/member names.
2. Update the "Individual contributions" paragraph in the conclusion.

Suggested contribution split:

- Data collection and preprocessing
- NLP model / dictionary classifier
- Evaluation and market validation
- Report writing
- Slides and video presentation

## FinBERT Note

There is an optional advanced model file:
`scripts/finbert_optional.py`

Current project uses the dictionary model as the main reproducible method. `scripts/finbert_optional.py` now supports full FinBERT fine-tuning with either reviewed labels or weak dictionary-generated labels. In presentation, phrase this carefully if only weak labels are used:

"We implemented a transparent dictionary-based sentence classifier as the reproducible baseline, and added a FinBERT fine-tuning workflow for the advanced model path. In the current reproducible version, FinBERT can be trained with weak labels generated from the dictionary; manually reviewed labels would be preferred for final research claims."

## Remaining To-Do

- [ ] Replace names in report.
- [ ] Replace individual contribution section.
- [ ] Review slides for team names.
- [ ] Record 5-7 minute video using `slides/video_script.md`.
- [ ] Submit `AI_Project_FOMC_Tone.zip`.

## Suggested Presentation Flow

1. Problem: FOMC minutes are long but contain policy signals.
2. Data: Fed minutes + FRED Treasury yields.
3. Method: sentence classification into hawkish/dovish/neutral.
4. Score: net hawkish score per meeting.
5. Main chart: tone score vs 10-year yield.
6. Validation: correlations with DGS10 and DGS2.
7. Limitations: dictionary misses context/negation.
8. Future work: fine-tuned FinBERT and futures-market validation.
