# Real Market Data Engineering - Research Report

## Executive Assessment

This run rebuilds the India VIX minute-data pipeline with corrected ISO timestamp parsing, bounded OHLC repair, leakage-safe forward targets, and a one-bar-delayed diagnostic backtest.

## Data Integrity

- Rows processed: 978081
- Date range: 2015-01-09 09:15:00 to 2025-08-06 17:56:00
- Raw OHLC-invalid rows: 2547
- Bad ticks flagged: 10434 (1.07%)
- Post-clean OHLC-invalid rows: 0
- Volume informative: False

## Stationarity

- close_price: INCONCLUSIVE (tests disagree - may need differencing)
- close_price observations tested: 250000 of 978081
- close_price ADF p-value: 0.000092
- close_price KPSS p-value: 0.010000
- log_returns: STATIONARY (both tests agree)
- log_returns observations tested: 250000 of 978080
- log_returns ADF p-value: 0.000000
- log_returns KPSS p-value: 0.100000

## Feature Set

- Feature columns created: 78
- Forward target columns created: 10
- Feature matrix shape: (978081, 98)
- Feature matrix missing-value share: 0.39%

## Diagnostic Backtest

- Total return: -30.45%
- Annualized return: -3.45%
- Annualized volatility: 7.10%
- Sharpe ratio: -0.486
- Max drawdown: -35.84%
- Trades: 308
- Total turnover: 308.00
- Total cost drag: 46.20%

Note: the backtest is a diagnostic data-engineering sanity check, not an investable strategy claim. India VIX is an index, so live deployment would require a mapped tradeable instrument and exchange-specific execution assumptions.

## Validation

- Tests passed: 30
- Tests failed: 0
- Warnings: 1
