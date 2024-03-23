# Extreme Weather Prediction — Tornado Detection

A machine learning system that predicts tornado occurrence from historical and
meteorological data, delivered as a Streamlit application.

Built over a six-month remote internship at **Tekkolab** (Canada) and submitted
as my BSc final-year dissertation, defended June 2024 (**16/20**).

![Tornado Alley](assets/Tornado%20Alley.svg.png)

## The problem

Tornado prediction is a hard, high-stakes classification problem with a brutal
class imbalance: tornadoes are rare events against a vast background of ordinary
weather. The goal was to predict occurrence from meteorological conditions, and
to identify where and under what conditions tornadoes are most likely.

## The data

**Base dataset:** 68,693 tornado records across 27 columns, spanning 1950–2022,
covering the United States.

Exploratory analysis identified **Tornado Alley** — concentrated around Texas —
as both the region of highest tornado frequency and the region with the most
complete documentation. That became the analytical focus, since modelling on
sparse or unreliable records elsewhere would have added noise rather than signal.

**Weather data** came from the NOAA API, joined to tornado events by location
and date.

## Engineering the data pipeline

The NOAA ingestion was the hardest engineering problem in the project.

**Parallel collection.** Serial requests against the NOAA API were far too slow
for the volume needed. I built a concurrent collection pipeline issuing
simultaneous requests load-balanced across multiple API keys, with
retry-on-failure handling for the API's frequent transient errors.

**Feature profiling.** NOAA exposes 1,567 distinct data types. I catalogued all
of them into a reference, then filtered by category and applied a 60%
missing-value threshold, keeping only features with enough coverage to be usable.

**Station matching, decided on evidence.** Two strategies were available for
attaching weather observations to tornado events: nearest-station matching using
the Haversine formula, or direct coordinate retrieval. Rather than assume, I
benchmarked both, measured a 33% match rate between them, and selected the more
accurate method on the evidence.

## Analysis and modelling

**Trend analysis.** A linear regression on annual tornado counts quantified an
upward trend of **12.4 additional recorded events per year** (r² = 0.62). Worth
reading carefully — improved detection and reporting over seven decades is a
plausible contributor, not only a genuine change in tornado frequency.

**Feature selection.** Pearson correlation analysis with a 0.8 threshold to drop
collinear features.

**Missing data.** Rather than dropping or mean-filling, I used a Random Forest
Regressor for model-based imputation of missing values such as magnitude and
loss. Wind direction needed special handling — as a circular variable, 359° and
1° are adjacent, so naive interpolation is wrong. I converted degrees to
radians, applied a sinusoidal transform, interpolated, and converted back.

**Feature engineering.** 20 new features including temporal attributes and
one-hot encoded categoricals.

**Class balancing.** Generated synthetic non-tornado events across matched
coordinate and date ranges, verifying zero overlap with real tornado events so
no negative sample could accidentally be a positive one.

**Model.** An `MLPClassifier` neural network with systematic hyperparameter
tuning.

## Results

| Metric | Value |
|---|---|
| Test accuracy | **96.6%** |
| 10-fold cross-validated accuracy | **96.1%** (std 0.0099) |

The low standard deviation across folds is the number that matters — it says the
result is stable rather than a lucky split.

## The application

A Streamlit app presenting the exploratory findings and model predictions for
non-technical stakeholders.

```
home.py                        entry point and project overview
pages/data_visualization.py    EDA, Tornado Alley mapping, trend analysis
pages/model_training.py        model training, tuning and evaluation
data/                          raw records and engineered feature sets
assets/                        reference imagery
```

## Running it

Requires Python 3.10+.

```bash
git clone https://github.com/AyoubDjellaoui/Extreme-Weather-Prediction.git
cd Extreme-Weather-Prediction
pip install -r requirements.txt
streamlit run home.py
```

## Limitations

The model is trained on US records with a Tornado Alley focus; it should not be
assumed to transfer to other regions or climates. The 1950–2022 record also
reflects seven decades of changing detection technology and reporting practice,
which affects trend interpretation more than it affects classification.

## Report

The full dissertation is included as [`Dissertation in french.pdf`](Dissertation%20in%20french.pdf) — written in French, as submitted.

## Tools

Python · scikit-learn · pandas · NumPy · NOAA API · Streamlit · Plotly · Folium ·
statsmodels · Google Colab

## Licence

MIT — see [LICENSE](LICENSE).
