# Real Market Data Engineering Pipeline

Production-grade market data engineering for quantitative trading research on India VIX minute bars.

This project turns raw index-level OHLCV data into a clean, validated, feature-rich research dataset with honest statistical diagnostics and a leakage-safe diagnostic backtest. The objective is not to claim a tradable strategy from one dataset; it is to demonstrate the market-data discipline required before alpha research or strategy development.

## What This Project Does

- Parses raw timestamps deterministically using the source ISO format (`YYYY-MM-DD HH:MM:SS`).
- Preserves raw OHLC prices in cleaned data and repairs invalid bars with bounded forward/backward filling.
- Flags bad ticks using robust Hampel filters on both price level and log returns.
- Enforces post-clean OHLC invariants (`high >= open/close/low`, `low <= open/close/high`).
- Runs stationarity and time-series diagnostics: ADF, KPSS, Ljung-Box, and ARCH LM.
- Applies HP-filter decomposition, linear/rolling detrending support, and rolling normalization.
- Builds a compact feature matrix with 78 engineered features and 10 forward target columns.
- Skips volume-derived features when volume is structurally non-informative, which is the case here because India VIX volume is all zero.
- Runs a one-bar-delayed, cost-aware diagnostic backtest to avoid same-bar close leakage.
- Generates validation JSON, plots, and a Markdown research report.

## Current Results

Latest regenerated run:

- Rows processed: `978,081`
- Date range: `2015-01-09 09:15:00` to `2025-08-06 17:56:00`
- Raw OHLC-invalid rows: `2,547`
- Bad ticks flagged: `10,434` (`1.07%`)
- Post-clean OHLC-invalid rows: `0`
- Feature matrix: `978,081 x 98`
- Engineered features: `78`
- Forward target columns: `10`
- Validation: `30 passed`, `0 failed`, `1 warning`

Stationarity:

- `close_price`: ADF says stationary, KPSS rejects stationarity, so the level series is treated as inconclusive.
- `log_returns`: ADF and KPSS both indicate stationarity on the tested sample.
- Returns still show serial dependence by Ljung-Box, which is expected for noisy intraday financial series and motivates careful modeling.

Diagnostic backtest:

- Total return: `-30.45%`
- Annualized return: `-3.45%`
- Annualized volatility: `7.10%`
- Sharpe ratio: `-0.486`
- Max drawdown: `-35.84%`
- Trades: `308`
- Gross return sum: `+12.52%`
- Cost drag: `46.20%`

Interpretation: the baseline signal has some gross directional information, but it does not survive the configured 10 bps commission plus 5 bps slippage assumptions. That is an honest research result, not a failure of the data pipeline.

## Project Structure

```text
.
|-- data/
|   |-- raw/                     # Raw CSV data
|   |-- cleaned/                 # Loaded, cleaned, stationary parquet files
|   `-- features/                # Engineered feature matrix
|-- reports/
|   |-- plots/                   # Diagnostic plots
|   |-- stationarity_tests.json
|   |-- pipeline_summary.json
|   |-- validation_results.json
|   `-- research_report.md
|-- src/
|   |-- config.py                # Central configuration
|   |-- load_data.py             # Raw ingestion and timestamp parsing
|   |-- cleaning_v2.py           # Robust OHLC cleaning
|   |-- stationarity_tests.py    # ADF, KPSS, Ljung-Box, ARCH LM
|   |-- drift_removal.py         # HP filter and detrending
|   |-- feature_engineering.py   # Feature and target generation
|   |-- backtest_engine.py       # Leakage-safe diagnostic backtest
|   |-- visualizations.py        # Sample-aware diagnostic plots
|   |-- validate_pipeline.py     # Output validation
|   |-- research_report.py       # Markdown report generator
|   `-- pipeline.py              # Master orchestrator
`-- README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python src/pipeline.py
python src/validate_pipeline.py
python src/research_report.py
```

The full pipeline processes nearly one million rows. If the statistical tests have already been generated and you are iterating only on features/backtests, run the relevant module directly to avoid unnecessary recomputation.

## Pipeline Stages

### 1. Data Ingestion

`src/load_data.py`

The loader parses timestamps using the raw ISO format. This is a critical control: using `dayfirst=True` on this dataset drops every date with day greater than 12 and swaps month/day on ambiguous dates.

Output: `data/cleaned/INDIA_VIX_loaded.parquet`

### 2. Cleaning

`src/cleaning_v2.py`

Cleaning uses robust Hampel filters and OHLC invariant checks. Raw prices are preserved in `open_raw`, `high_raw`, `low_raw`, and `close_raw`; cleaned prices are used for downstream returns and features.

Output: `data/cleaned/INDIA_VIX_cleaned.parquet`

### 3. Stationarity Diagnostics

`src/stationarity_tests.py`

The test battery includes ADF, KPSS, Ljung-Box, and ARCH LM. Very long series are capped to the most recent configured sample for expensive tests, and JSON output is written atomically to avoid corrupt partial reports.

Output: `reports/stationarity_tests.json`

### 4. Drift Removal

`src/drift_removal.py`

Adds HP trend/cycle decomposition, detrended price, and rolling-normalized price.

Output: `data/cleaned/INDIA_VIX_stationary.parquet`

### 5. Feature Engineering

`src/feature_engineering.py`

Feature families include returns, rolling statistics, OHLC volatility estimators, momentum/mean reversion, rolling autocorrelation, effective spread proxy, seasonality, regimes, and forward targets.

Output: `data/features/INDIA_VIX_features.parquet`

### 6. Diagnostic Backtest

`src/backtest_engine.py`

The backtest delays signals by one bar and charges turnover-based commission plus slippage. It is a diagnostic sanity check for whether simple features survive frictions, not a live trading claim.

Output: `reports/backtest_results.parquet`

## Important Caveats

- India VIX is an index, not a directly tradeable instrument. Live deployment would require mapping to a tradable proxy and validating execution assumptions.
- The provided momentum backtest is intentionally a baseline. It should not be interpreted as an optimized strategy.
- Volume is structurally zero in this dataset, so volume and VWAP features are skipped by design.
- Raw data contains OHLC inconsistencies; the cleaned dataset repairs them, and the validator keeps the raw warning visible.
