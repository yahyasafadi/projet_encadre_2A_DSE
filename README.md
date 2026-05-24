# compute_indicators

This script computes common technical indicators (RSI, SMA, EMA ratio, MACD histogram, ATR, intraday range, hour) for the crude oil CSV `crude_oil_wti_daily_last_5_years.csv`, matches each trading day with the nearest daily news summary, computes a VADER sentiment score, and writes `crude_wti_features.csv`.

Usage:

1. Create a virtualenv and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the script:

```powershell
python compute_indicators.py
```

Notes:
- The original `wti_features_ready.csv` includes additional features (USD momentum, Brent spread, sentiment, correlation) which require external data; this script reproduces the price-based indicators only.
- The price pipeline now reindexes the series to a continuous daily calendar and forward-fills missing prices so the model sees a gap-free timeline.
- News articles are grouped by day into one daily summary, then aligned to each price day by nearest date so the feature table stays one row per day.
