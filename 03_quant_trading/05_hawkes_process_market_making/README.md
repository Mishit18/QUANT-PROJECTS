# Hawkes Process Limit Order Book and Market Making

Research implementation of a multidimensional Hawkes-process order-flow simulator, event-driven limit order book, statistical diagnostics, MLE parameter estimation, and an inventory-aware market-making backtest.

This project is best positioned for quant research, quant trading, market microstructure, and HFT-adjacent screening. It now includes a real passive-fill linkage from the limit order book to the market-making agent, so the backtest updates inventory and cash when the agent's resting quotes are hit.

## What This Demonstrates

- Six-event Hawkes order-flow model: limit buy, limit sell, market buy, market sell, cancel buy, cancel sell.
- Exponential-kernel multidimensional Hawkes simulation using Ogata-style thinning.
- Stationarity validation through spectral radius of the excitation matrix.
- Time-rescaling residual diagnostics and Kolmogorov-Smirnov tests.
- Limit order book with price-time priority, market orders, cancellations, and passive order IDs.
- Inventory-aware market maker with reservation-price style quote skew.
- Event-driven backtest with real agent fill detection.
- Verified command-line demos and reproducible result summaries.
- Empirical clustering priors from 185,782 official Binance Vision trades, including interarrival dispersion and signed-flow dependence.

## Verified Full Simulation

Command:

```bash
$env:MPLBACKEND='Agg'
python experiments/run_simulation.py
```

Key output from the verified run:

| Metric | Value |
| --- | ---: |
| Simulated events | 3,805 |
| Simulation time | 999.13 |
| Hawkes spectral radius | 0.5732 |
| Stationary process | True |
| Market-maker trades | 96 |
| Final PnL | 8.30 |
| Max drawdown | 0.55 |
| Max drawdown pct | 16.67% |
| Spread capture | 1.72 ticks |
| Trade rate | 0.096 trades/sec |
| Final inventory | -40 |
| Raw Sharpe-like ratio | 0.346 |

The Sharpe-like ratio is intentionally not annualized because this is a simulator-level event process, not a production live-trading return stream.

## Hawkes Diagnostics

| Event Type | Count | Share | KS p-value |
| --- | ---: | ---: | ---: |
| Limit buy | 948 | 24.9% | 0.0656 |
| Limit sell | 1,011 | 26.6% | 0.0506 |
| Market buy | 506 | 13.3% | 0.0306 |
| Market sell | 479 | 12.6% | 0.0956 |
| Cancel buy | 423 | 11.1% | 0.0122 |
| Cancel sell | 438 | 11.5% | 0.0287 |

Interpretation: the process is stationary under the configured excitation matrix. Residual diagnostics are mixed: some event types are not rejected at the 5% level, while market/cancel event residuals show misspecification. This is good interview material because it shows model validation rather than blind curve-fitting.

## Mathematical Formulation

For event type `i`, the Hawkes intensity is:

```text
lambda_i(t) = mu_i + sum_j integral phi_ij(t - s) dN_j(s)
```

Using exponential kernels:

```text
phi_ij(t) = alpha_ij * beta_ij * exp(-beta_ij * t), t > 0
```

The process is stationary when:

```text
spectral_radius(alpha) < 1
```

The market maker uses an inventory-aware reservation price:

```text
r(t) = mid(t) - gamma * inventory(t) * sigma^2
```

## Run Locally

```bash
pip install -r requirements.txt
python experiments/demo_features.py
python experiments/run_simulation.py
```

For headless environments:

```powershell
$env:MPLBACKEND='Agg'
python experiments/run_simulation.py
```

## Project Structure

```text
05_hawkes_process_market_making/
|-- hawkes/
|   |-- kernels.py
|   |-- simulation.py
|   |-- estimation.py
|   `-- diagnostics.py
|-- lob/
|   |-- order.py
|   |-- limit_order_book.py
|   `-- event_processor.py
|-- agents/
|   |-- inventory_control.py
|   `-- market_maker.py
|-- backtest/
|   |-- engine.py
|   `-- metrics.py
|-- experiments/
|   |-- demo_features.py
|   `-- run_simulation.py
|-- results/
|   |-- event_distribution.csv
|   |-- ks_diagnostics.csv
|   `-- simulation_summary.csv
|-- docs/
|   `-- ATS_SCREENING_PACK.md
|-- configs/
|   `-- default.yaml
`-- report/
    `-- report_outline.md
```

## Screening Positioning

Best fit:

- Quant Research
- Quant Trader
- Market Microstructure Research
- HFT Research
- Execution Research

Do not overclaim:

- This is a simulator, not a live market-making system.
- The data is synthetic Hawkes-generated order flow.
- Real aggregate trades calibrate arrival and clustering priors but cannot identify the six-event LOB kernel because they omit limit orders and cancellations.
- MLE results are demonstration-level and should be strengthened before claiming production calibration.
- Some KS diagnostics reject the configured model, which should be described as validation evidence, not hidden.

## Next Upgrade

- Calibrate to LOBSTER/NASDAQ ITCH data.
- Add power-law or sum-of-exponentials kernels.
- Add size-dependent excitation and marked Hawkes processes.
- Add queue-position modeling and latency.
- Add adverse-selection losses against future mid-price moves.
