# HFT Microstructure Alpha Signals

Research-grade limit order book microstructure project built around the real FI-2010 Nasdaq Nordic benchmark, with separate Binance validation and an explicitly isolated synthetic execution stress test.

The primary experiment predicts 10-event mid-price direction from 394,337 real ten-level LOB observations across five stocks and ten trading days. Days 1-7 train the model, day 8 is validation, and days 9-10 remain untouched holdout. Separate Binance benchmarks validate current-market feature computation and trade-flow behavior; a legacy synthetic simulator remains only for controlled execution-friction experiments.

## Executive Snapshot

| Metric | Result |
|---|---:|
| Primary data | FI-2010, 394,337 real LOB observations |
| Coverage | 5 stocks, 10 trading days, 10 book levels |
| Input features | 57 raw-book and derived state features |
| Holdout | Days 9-10, 84,109 observations |
| Majority balanced accuracy / macro-F1 | 0.333 / 0.272 |
| Class-balanced XGBoost balanced accuracy / macro-F1 | 0.468 / 0.419 |
| Market-order PnL | -$58.47 |
| Limit + confidence filter PnL | -$3.30 |
| EV-filtered PnL | +$0.41 on 1 trade |
| Negative-EV signal filter rate | 94.6% |
| Passive fill rate | 20.0% |
| Real-market validation | Live Binance BTCUSDT top-20 snapshots |

## Core Insight

The model can classify short-horizon price moves well, but most signals are not economically tradable after costs. The EV filter removes 94.6% of candidate signals and leaves only one executed trade in the committed report. That is not a profitability claim; it is evidence that realistic microstructure frictions dominate naive predictive accuracy.

## What This Project Demonstrates

- Order Flow Imbalance, queue imbalance, microprice, spread dynamics, and event-time feature engineering.
- Time-respecting model validation for short-horizon prediction.
- Logistic regression baseline versus XGBoost.
- Alpha decay across 1, 5, 10, and 20 tick horizons.
- Regime analysis across volatility, liquidity, and time-of-day segments.
- Event-driven backtesting with market orders, passive limit orders, and EV-based execution.
- Honest execution diagnostics explaining why most high-accuracy signals should be rejected.

## Quick Start

```bash
pip install -r requirements.txt
python verify_installation.py
python run_pipeline.py
python scripts/collect_live_binance_lob.py --snapshots 1000
```

The pipeline runs:

1. Checksum-verified FI-2010 download and extraction.
2. Leakage-controlled parsing of 40 ten-level price/volume variables.
3. Microprice, queue-imbalance, spread, depth, and book-slope features.
4. Chronological days 1-7/day 8/days 9-10 train-validation-holdout split.
5. Majority baseline and class-balanced XGBoost comparison.
6. JSON and Markdown evidence reports with confusion matrices and limitations.

Run `python run_pipeline.py --data-source synthetic` only to reproduce the legacy controlled execution study.

The live collector writes timestamped snapshots and a JSON summary under `reports/`. It is feature-validation evidence only; the predictive backtest remains explicitly synthetic.

Run `python scripts/run_real_binance_benchmark.py` for a chronological 60/20/20 train-validation-holdout benchmark on official Binance Vision BTCUSDT aggregate trades. The script exports holdout predictions, ROC-AUC, balanced accuracy, active-signal rate, and a deliberately conservative one-basis-point cost check.

### Committed live-market validation

The latest public-data run collected `1,000` BTCUSDT top-20 snapshots and `276` aggregate-trade messages in `108.64` seconds through Binance's market-data-only WebSocket endpoint. It observed a mean depth imbalance of `0.3723` (standard deviation `0.3360`) and writes the raw snapshot features to `reports/live_binance_lob_snapshots.csv` with the reproducible summary in `reports/live_binance_lob_summary.json`.

This evidence validates current-market parsing and feature computation. FI-2010 is now the primary predictive benchmark; neither experiment is presented as realized trading performance.

## Project Structure

```text
.
|-- config/
|-- notebooks/
|-- reports/
|   |-- summary_report.md
|   `-- figures/
|-- src/
|   |-- analysis/
|   |-- backtest/
|   |-- data/
|   |-- features/
|   |-- labels/
|   |-- models/
|   `-- utils/
|-- run_pipeline.py
|-- verify_installation.py
`-- requirements.txt
```

## Feature Families

| Family | Examples |
|---|---|
| Order Flow | OFI, signed volume, trade direction |
| Queue State | Bid/ask depth ratios, queue imbalance |
| Fair Value | Microprice, mid-price changes |
| Spread | Absolute spread, relative spread, spread widening |
| Event-Time | Tick returns, rolling volatility, event intensity |

## Execution Results

| Strategy | PnL | Sharpe | Trades | Interpretation |
|---|---:|---:|---:|---|
| Market Orders | -$58.47 | -105.02 | 116 | Spread crossing destroys edge |
| Limit + Filtering | -$3.30 | -2.17 | 53 | Passive execution helps but remains negative |
| EV-Based | +$0.41 | +0.23 | 1 | Too selective to claim robust profitability |

## Why Accuracy Does Not Monetize

1. Expected moves are small relative to spread and fees.
2. Passive orders fill only when the market comes to the quote.
3. Filled passive orders are exposed to adverse selection.
4. Queue priority and latency are missing from a simple research simulator.
5. Synthetic LOB data does not include hidden liquidity, toxic flow, or real venue mechanics.

## What Not To Claim

- Do not claim this is profitable HFT.
- Do not claim the +$0.41 EV-filtered result is meaningful performance; it is one trade.
- Do not describe FI-2010 as current-market or proprietary data.
- Do not claim production readiness.
- Do not claim high accuracy implies tradable edge.

## Resume Bullets

- Built 57 LOB state features on 394,337 real FI-2010 observations spanning five Nasdaq Nordic stocks, ten book levels, and ten trading days.
- Ran chronological days 1-7/day 8/days 9-10 evaluation; class-balanced XGBoost reached 0.468 holdout balanced accuracy versus 0.333 majority baseline.
- Added event-driven execution simulation showing market-order PnL -$58.47 and limit-filtered PnL -$3.30, demonstrating that spread costs and adverse selection can erase high predictive accuracy.
- Implemented EV-based execution filter that rejected 94.6% of candidate signals as uneconomic after costs; final result (+$0.41 on 1 trade) is framed as signal rejection discipline, not profitability.

## Interview Defense

### Why is this still a strong project if PnL is weak?

Because it demonstrates the most important HFT lesson: prediction and execution are different problems. A model can forecast direction while still failing after spread, queue, and adverse-selection costs.

### What would be needed for a stronger live strategy?

Real LOB data, calibrated fill probabilities, queue-position modeling, latency assumptions, venue-specific fees/rebates, and a strategy such as market making or cross-asset arbitrage that can monetize microstructure signals more directly.

### Why use synthetic data?

It makes the full pipeline reproducible and shareable. The limitation is that real LOBs contain hidden liquidity, participant behavior, and venue mechanics not captured here.

## References

- Cont, R., Kukanov, A., and Stoikov, S. (2014). The price impact of order book events.
- Cartea, A., Jaimungal, S., and Penalva, J. (2015). Algorithmic and High-Frequency Trading.
- Lehalle, C. A., and Laruelle, S. (2018). Market Microstructure in Practice.
