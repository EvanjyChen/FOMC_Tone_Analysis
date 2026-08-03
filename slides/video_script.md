# 5-7 Minute Video Script

1. Introduce the problem: FOMC minutes are long qualitative documents, but markets respond to monetary-policy tone.
2. Explain the data: public Federal Reserve minutes from 2008-2026 and FRED DGS10/DGS2 Treasury yields.
3. Walk through the pipeline: download, clean, sentence split, classify, aggregate, align with market data.
4. Explain the model: hawkish and dovish dictionary terms classify sentences; net hawkish score summarizes each meeting.
5. Present the main chart: point out 2008 crisis easing, 2020 pandemic easing, and 2022 anti-inflation tightening.
6. Present validation: positive correlation with Treasury yields, strongest for DGS2, which is more policy-sensitive.
7. Discuss limitations and FinBERT extension: dictionary is interpretable but misses context; fine-tuning FinBERT can improve sentence labels.
8. Conclude: the project demonstrates an end-to-end AI/NLP application using public data and market validation.
