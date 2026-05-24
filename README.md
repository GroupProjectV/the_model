# Issue Salience and Partisan Electoral Outcomes

**Research Question:** To what extent do state-level issue salience profiles — defined by the dominant policy concerns driving voter behavior — predict partisan electoral outcomes across U.S. states in presidential elections?

**Theoretical Framework:** Issue Ownership Theory (Petrocik, 1996)

## Overview

This project analyzes ~2.1 million tweets from the 2024 U.S. election cycle to determine whether the political issues people discuss on Twitter/X can predict which party wins each state. We combine three NLP methods to extract different signals from the text, then use those signals as features in a predictive model.

## Methods

| Method | Purpose | Tool |
|--------|---------|------|
| **BERTopic** | Identify political issues in tweets (topic modeling) | `all-MiniLM-L6-v2` + UMAP + HDBSCAN |
| **VADER** | Measure emotional tone toward each issue | Lexicon-based sentiment analysis |
| **Zero-Shot Classification** | Detect partisan stance (Republican / Democrat / Neutral) | `facebook/bart-large-mnli` |
| **Predictive Modeling** | Predict state-level election outcomes from issue salience | Logistic Regression, Random Forest, Gradient Boosting |
| **Bias Audit** | Assess geographic, engagement, temporal, and fairness biases | Statistical analysis |

## Pipeline

```
Raw Tweets (HuggingFace)
    │
    ▼
Text Cleaning & Stratified Sampling (75K tweets)
    │
    ├──► BERTopic ──► State-level topic distributions
    │
    ├──► VADER ──► State-level sentiment scores
    │
    └──► Zero-Shot LLM ──► State-level partisan lean
            │
            ▼
    Feature Matrix (50 states × ~25 features)
      + 2024 Election Results
            │
            ▼
    Predictive Model (LOOCV)
            │
            ▼
    Bias Audit & Comparison
```

## Dataset

Hosted on HuggingFace: [`Diogo2110/sem4data`](https://huggingface.co/datasets/Diogo2110/sem4data)

- ~2.1 million tweets collected during the 2024 U.S. election cycle
- 75 columns including tweet text, user metrics, engagement metrics, state geolocation, and timestamps
- Loaded directly in the notebook via the `datasets` library

## Setup

### Requirements

```
pip install datasets bertopic sentence-transformers vaderSentiment transformers torch scikit-learn plotly gensim umap-learn hdbscan tqdm kaleido
```

### Running

Open `bert_scratch_kl.ipynb` and run all cells sequentially. The notebook handles data loading, model training, evaluation, and visualization.

**Runtime estimates:**
- Data loading: ~5–10 min (downloads from HuggingFace)
- BERTopic embedding + clustering: ~3–5 min (GPU recommended)
- VADER sentiment: ~2 min
- Zero-shot stance classification: ~30–60 min on CPU, ~10 min on GPU
- Predictive modeling + bias audit: < 1 min

## Notebook Structure

| Section | Cells | Description |
|---------|-------|-------------|
| 0. Setup | 1–3 | Dependencies, imports, configuration |
| 1. Data Loading | 4–10 | Load from HuggingFace, clean, filter, stratified sample |
| 2. BERTopic | 11–24 | Topic modeling, outlier reduction, evaluation, state distributions |
| 3. VADER | 25–32 | Sentiment analysis by topic, state, and cross-tabulation |
| 4. Zero-Shot LLM | 33–38 | Partisan stance detection, state-level partisan lean |
| 5. Feature Engineering | 39–43 | Election data, state feature matrix, correlation analysis |
| 6. Predictive Modeling | 44–50 | Model comparison (LOOCV), feature importance, prediction map |
| 7. Bias Audit | 51–61 | Geographic, engagement, temporal, fairness analysis |
| 8. Summary | 62–64 | Results table and limitations |

## Key Improvements Over Baseline

| Metric | Baseline | Improved |
|--------|----------|----------|
| Outlier rate | 51.4% | Target < 25% |
| Coherence (c_v) | 0.4609 | Maintained or improved |
| Methods | 1 (BERTopic only) | 3 (BERTopic + VADER + LLM) |
| Prediction | None | Logistic Regression / RF / GBM |
| Bias audit | None | Geographic, engagement, temporal, fairness |

## Project Context

This is part of the **Group Project V: Unequal Machines Grand Challenge** — a computational social science course examining algorithmic bias and social inequality. The project has two phases:

1. **Mindless Machine Phase** — Build the best predictive model possible using standard methods
2. **Auditor Phase** — Critically examine the model for biases using critical theory (feminism, critical political economy, or postcolonialism) and implement bias mitigation strategies
