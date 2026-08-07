# Results and Limitations

## Executive Summary

This project demonstrates a methodologically sound Kalman filter plus HMM regime framework. The latest reproduced run does **not** beat buy-and-hold SPY. That is an important result, not a failure to hide: the framework is useful for state estimation, regime diagnostics, and risk-aware signal construction, but the current active strategy should not be marketed as discovered alpha.

Latest production run:

- Run date: 2026-04-11
- Data source: cached real Yahoo Finance snapshot, `SPY_QQQ_TLT_20260411.csv`
- Primary return stream: SPY
- Return window: 2021-04-13 to 2026-04-10
- Observations: 1,255 daily returns
- Transaction cost: 5 bps per unit turnover
- Strategy: enhanced regime-aware signal with TSMOM enabled, regime gating disabled

## Strategy Results

| Strategy | Sharpe | Annual Return | Volatility | Max Drawdown | Avg Turnover |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | 0.67 | 10.42% | 17.05% | 26.22% | 0.08% |
| Kalman Trend | -0.71 | -12.68% | 17.07% | 53.59% | 48.84% |
| Regime-Aware | -0.35 | -3.18% | 8.18% | 26.03% | 27.69% |

## Regime Diagnostics

The HMM relabels regimes by ascending fitted emission volatility. Regime statistics use posterior-dominant smoothed states for diagnostics; trading signals use filtered probabilities.

| Regime | Interpretation | Frequency | Daily Mean | Daily Volatility | Sharpe |
|---|---|---:|---:|---:|---:|
| 0 | Low volatility / constructive | 65.42% | 0.0914% | 0.7104% | 2.04 |
| 1 | Medium volatility | 33.63% | 0.0045% | 1.3079% | 0.05 |
| 2 | Crisis / high volatility | 0.96% | -1.6658% | 4.7755% | -5.54 |

The high-volatility regime is rare in this sample. The pipeline correctly warns when a regime has low frequency; this should be treated as a model-risk signal, not as a green light for deployment.

## What Improved

Recent hardening changed the project in ways that matter for quant credibility:

- Trading signals now use filtered HMM probabilities by default, avoiding smoothed-probability lookahead.
- HMM labels are volatility-ordered after fitting, so regime `0` means low volatility and regime `K-1` means highest volatility.
- Gaussian emission covariance floors reduce outlier-memorizing regime collapse.
- Regime diagnostics use posterior-dominant labels rather than an over-persistent Viterbi path.
- Real-data loading is cache-first and reproducible, with live download retry as fallback.
- Plot generation is headless-safe and does not emit backend warnings in production runs.
- Result CSVs now write clean scalar regime statistics.

## Interpretation

Buy-and-hold wins on absolute return because SPY had a positive equity risk premium over the sample. The active strategy reduces volatility versus naive Kalman trend but still loses money after turnover and costs. This suggests the current signal stack is more useful as a research framework than as a production alpha model.

The naive Kalman trend strategy is especially weak: it has high turnover, negative return, and a 53.59% max drawdown. The regime-aware layer improves that substantially, but not enough to beat passive exposure.

## Why Negative Active Returns Are Acceptable Here

The project is built to demonstrate:

- Correct Kalman filter implementation with numerical safeguards.
- HMM regime inference and diagnostics.
- Causal signal construction.
- Transaction-cost-aware backtesting.
- Honest reporting of failure modes.

It does **not** claim:

- A new alpha discovery.
- Market-beating performance.
- A production trading system.
- Regulatory or operational readiness.

## Key Limitations

### Full-sample model fitting

The default pipeline fits the Kalman filter and HMM on the full sample before backtesting. Signal probabilities are filtered, and positions are lagged, but parameters are still estimated using the full sample. A true investment-grade study needs walk-forward re-estimation.

### Small universe

The default run uses SPY as the traded primary stream and loads QQQ/TLT for the data bundle. This is too narrow for robust cross-asset conclusions.

### Gaussian HMM assumptions

Financial returns are fat-tailed and heteroskedastic. Gaussian emissions are convenient and interpretable, but they can still overreact to outliers. Covariance floors help; they do not solve the distributional mismatch.

### Rare crisis regime

The high-volatility regime appears in less than 1% of posterior-dominant observations in the current sample. Any regime-conditioned rule depending on that state has high estimation uncertainty.

### Transaction costs

The enhanced strategy turnover is 27.69% average daily turnover. At 5 bps, costs materially drag returns. Any live version would need a stronger cost model including spreads, slippage, market impact, and capacity.

### No production execution stack

The repo does not include OMS integration, broker routing, real-time risk checks, trade reconciliation, model monitoring, compliance logging, or failover.

## Required Next Steps For Institutional Research

1. Add walk-forward HMM/Kalman re-estimation.
2. Evaluate multiple assets and portfolio-level construction, not only SPY.
3. Add robust emissions such as Student-t HMM or a switching volatility model.
4. Run crisis-window stress tests, including 2008, 2020, and 2022 where data availability permits.
5. Add parameter sensitivity reports for regime count, covariance floor, transaction cost, and volatility target.
6. Attribute PnL by signal component: Kalman trend, TSMOM, regime sizing, volatility targeting, and costs.
7. Add bootstrap confidence intervals and deflated Sharpe / multiple-testing controls before making any performance claim.

## Bottom Line

The framework is much stronger as a quant engineering and modeling project after the causality, regime-labeling, HMM regularization, and reporting fixes. The active trading results remain negative in the latest sample. A senior quant should present that plainly and use it to guide the next research iteration.
