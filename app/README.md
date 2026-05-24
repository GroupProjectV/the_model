---
title: Issue Salience Electoral Predictor
emoji: "\U0001F5F3️"
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: false
---

# Issue Salience Electoral Predictor

Interactive demo for analyzing political tweets and predicting U.S. state-level election outcomes based on issue salience profiles.

## Features

- **Live Tweet Analysis** — Enter any tweet to see its detected political topic, sentiment, and partisan stance
- **Interactive State Maps** — Explore choropleth maps of topic distributions, sentiment, partisan lean, and election predictions
- **Topic Explorer** — Drill into which political issues dominate in each state
- **Model Performance** — View predictive model accuracy, feature importance, and bias audit results

## How to Run Locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Generating Artifacts

Run the full notebook (`bert_scratch_kl.ipynb`) first — the final section exports all model artifacts to `app/artifacts/`.
