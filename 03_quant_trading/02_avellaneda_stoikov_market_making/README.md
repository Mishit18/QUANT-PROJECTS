# Avellaneda-Stoikov Market Making

Interview-grade implementation of the Avellaneda-Stoikov optimal market-making model with Monte Carlo validation, self-financing PnL accounting, multi-agent competition, queue-position analysis, and explicit failure-mode documentation.

## Executive Snapshot

| Area | Result |
|---|---:|
| Core model | Avellaneda-Stoikov inventory-aware optimal quoting |
| Validation style | Monte Carlo distributions, not single-path cherry-picking |
| PnL identity | Self-financing accounting verified |
| PnL attribution | Spread capture, inventory PnL, adverse selection, residual |
| Default baseline Sharpe | 0.025 mean across 500 Monte Carlo paths |
| Default baseline PnL | $4.07 +/- $1.62 across 500 Monte Carlo paths |
| Microprice effect | Reported as an experimental microstructure comparison, not a live edge |
| Queue effect | Fill probability decays from front-of-queue to deeper queue positions |
| Main failure regimes | High volatility, low arrival intensity, toxic flow, excessive competition |

## What This Project Demonstrates

- Derivation and implementation of reservation price and optimal bid/ask quotes.
- Inventory-risk control through asymmetric quoting.
- PnL decomposition into spread capture, inventory PnL, adverse selection, and residual attribution.
- Monte Carlo experiment design for market-making strategies.
- Multi-agent competition and spread compression.
- Queue-position and microprice extensions for more realistic microstructure discussion.
- Failure-mode analysis that explains when the model should not be trusted.
- Real-market priors from 185,782 official Binance Vision trades for volatility, trade intensity, and size calibration, with simulated-fill boundaries stated explicitly.

## Quick Start

```bash
pip install -r requirements.txt
python main.py baseline
```

Run all experiments:

```bash
python main.py all
```

Run individual experiments:

```bash
python main.py baseline
python main.py pnl_decomposition
python main.py competition
python main.py regime_sweep
python main.py microprice
python main.py queue
```

Generated artifacts are written to:

- `results/figures/`
- `results/tables/`

## Core Model

The model quotes around a reservation price that shifts with inventory:

```text
r(t, S, q) = S - q * gamma * sigma^2 * (T - t)
```

Intuition:

- Long inventory lowers the reservation price, encouraging selling.
- Short inventory raises the reservation price, encouraging buying.
- Higher volatility and risk aversion increase the inventory penalty.

The optimal spread balances spread capture against fill probability:

```text
delta* = (1 / gamma) * log(1 + gamma / kappa) + inventory_adjustment
```

This project uses the model as a controlled research framework, not as a live trading claim.

## Experiment Modules

| Experiment | Purpose |
|---|---|
| `baseline` | Monte Carlo baseline for single-agent market making |
| `pnl_decomposition` | Verifies self-financing PnL accounting and attribution |
| `competition` | Simulates multi-agent quoting and spread compression |
| `regime_sweep` | Tests volatility and arrival-rate failure regions |
| `microprice` | Compares mid-price versus microprice reference quoting |
| `queue` | Studies queue position and fill-probability decay |

## PnL Accounting

The project enforces the self-financing identity:

```text
Total PnL = Cash + Inventory * MidPrice - Initial Wealth
```

PnL is decomposed into:

- Spread capture: earned bid/ask spread.
- Inventory PnL: mark-to-market change from holding inventory.
- Adverse selection: cost from being filled before unfavorable price moves.
- Residual: attribution remainder that should be inspected rather than ignored.

This makes the project harder to overstate: the PnL source must be explained rather than treated as a black box. The current committed table marks the self-financing check as valid, but the attribution residual is not zero, so use the decomposition as a diagnostic rather than claiming perfect attribution.

## Key Findings

### 1. Realistic Market-Making Sharpe Is Modest

The default baseline run produces mean PnL of $4.07 +/- $1.62 and mean Sharpe of 0.025 +/- 0.005 across 500 Monte Carlo paths. That is intentionally modest: market making earns spread but pays for inventory risk, adverse selection, and competition. Very high Sharpe from a simple simulator would usually suggest cherry-picking or missing costs.

### 2. Microprice Helps But Does Not Remove Adverse Selection

Using microprice instead of mid-price improves quote placement when order-book imbalance contains information. Treat this as a microstructure experiment, not as a live profitability claim; toxic flow remains a core failure mode.

### 3. Queue Position Matters

Front-of-queue quotes have materially higher fill probability than deeper queue positions. This explains why theoretical quotes are not enough; execution priority changes realized PnL.

### 4. Competition Compresses Profits

When multiple market makers quote the same process, spreads compress and mean PnL moves toward zero while inventory risk remains. This is the correct economic intuition for competitive market making.

### 5. The Model Has Clear Failure Regimes

| Regime | Failure Mode |
|---|---|
| High volatility | Inventory variance overwhelms spread capture |
| Low arrival intensity | Too few fills; PnL dominated by isolated events |
| Toxic directional flow | Adverse selection dominates quoted spread |
| Many competitors | Spread compression erodes edge |

## What Not To Claim

- Do not claim this is a production trading system.
- Do not claim the strategy is live-tradeable.
- Do not claim exchange-validated fill probabilities; real trades calibrate market priors while fills and PnL remain simulated.
- Do not report a single lucky path as performance.
- Do not hide that results are simulated.

## Resume Bullets

- Derived and implemented Avellaneda-Stoikov inventory-aware optimal quoting with reservation-price adjustment, asymmetric spreads, and Monte Carlo validation across market regimes.
- Built PnL attribution diagnostics decomposing market-making returns into spread capture, inventory PnL, adverse selection, and residual attribution.
- Simulated multi-agent competition, queue-position effects, and microprice-based quoting; identified failure regimes where high volatility, low fills, toxic flow, or spread compression make the model unreliable.

## Interview Defense

### Why does inventory affect quotes?

Inventory creates directional exposure. A long market maker wants to reduce inventory, so the reservation price shifts downward and quotes become more aggressive on the ask side. A short market maker does the opposite.

### Why is the Sharpe not extremely high?

Because the model earns spread without directional alpha. The default Monte Carlo baseline has mean Sharpe 0.025, which is modest but more credible than an inflated single-path result.

### What is the most important limitation?

The model assumes a simplified price process and fill-intensity function. Real markets have latency, hidden liquidity, queue dynamics, jumps, regime changes, and informed flow.

### How would you improve it?

Calibrate fill intensities to real LOB data, add latency and queue priority, impose hard position limits, detect toxic flow, and extend from single-asset quoting to portfolio-aware market making.

## Repository Structure

```text
.
|-- main.py
|-- config/parameters.yaml
|-- experiments/
|   |-- baseline_experiment.py
|   |-- pnl_decomposition_experiment.py
|   |-- competition_experiment.py
|   |-- regime_sweep_experiment.py
|   |-- microprice_experiment.py
|   `-- queue_experiment.py
|-- src/
|   |-- agents/
|   |-- analysis/
|   |-- market/
|   |-- models/
|   |-- simulation/
|   `-- utils/
|-- results/
|   |-- figures/
|   `-- tables/
|-- requirements.txt
`-- report.pdf
```

## Status

This is a research and interview project for quant trader / market-making roles. It is intentionally honest about limitations and is not intended for production trading.
