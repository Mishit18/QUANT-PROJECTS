# HFT Microstructure Alpha Signals

Research-grade limit order book microstructure project showing a central trading lesson: high directional accuracy does not automatically become profitable execution.

The project predicts short-horizon price moves from synthetic full-depth LOB data, then tests whether those predictions survive spread costs, queue dynamics, passive fill risk, and adverse selection.

## Executive Snapshot

| Metric | Result |
|---|---:|
| Data | Synthetic LOB, 100,000 events |
| Feature count | 59 microstructure features |
| Horizons | 1, 5, 10, 20 ticks |
| Logistic baseline accuracy | 62.5% |
| XGBoost 5-tick accuracy | 90.8% |
| 5-tick hit rate | 95.1% |
| Market-order PnL | -$58.47 |
| Limit + confidence filter PnL | -$3.30 |
| EV-filtered PnL | +$0.41 on 1 trade |
| Negative-EV signal filter rate | 94.6% |
| Passive fill rate | 20.0% |

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
```

The pipeline runs:

1. Synthetic LOB data generation.
2. Feature engineering.
3. Label creation for 1, 5, 10, and 20 tick horizons.
4. Logistic and XGBoost model training.
5. Alpha decay and regime analysis.
6. Event-driven execution tests.
7. Report and figure generation.

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
- Do not claim real LOB data.
- Do not claim production readiness.
- Do not claim high accuracy implies tradable edge.

## Resume Bullets

- Built a 59-feature synthetic LOB microstructure pipeline using OFI, queue imbalance, microprice, spread dynamics, and event-time features; XGBoost reached 90.8% 5-tick directional accuracy versus 62.5% logistic baseline.
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
