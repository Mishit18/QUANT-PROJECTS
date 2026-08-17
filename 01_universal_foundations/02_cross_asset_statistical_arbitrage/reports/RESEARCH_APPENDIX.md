# Research Appendix: Stat-Arb Evidence Pack

## Purpose

This appendix upgrades the project from a backtest script into a quant research memo. It documents IC stability, turnover, transaction-cost sensitivity, and deploy/reject gates.

## IC Diagnostics

| Metric | Value |
|---|---:|
| Mean daily rank IC | 0.0051 |
| IC information ratio | 0.05 |
| Positive-IC hit rate | 52.41% |
| IC observations | 2,343 |

Artifacts:

- `reports/xgboost_ic_timeseries.csv`
- `reports/figures/xgboost_ic_stability.png`

## Cost Sensitivity

| Cost bps | Total Return | Sharpe | Max Drawdown | Cost Drag |
|---:|---:|---:|---:|---:|
| 0.0 | 389.70% | 1.68 | -14.06% | 0.00% |
| 2.5 | 322.14% | 1.53 | -14.57% | 14.85% |
| 5.0 | 263.87% | 1.37 | -15.08% | 29.71% |
| 7.5 | 213.64% | 1.22 | -15.58% | 44.56% |
| 10.0 | 170.32% | 1.07 | -16.08% | 59.41% |
| 25.0 | 10.71% | 0.15 | -27.60% | 148.54% |
| 50.0 | -75.09% | -1.27 | -75.62% | 297.07% |

At the original 7.5 bps assumption, the recomputed proxy has total return 213.64%, Sharpe 1.22, and average rebalance turnover 126.41%. This supports the current reject decision because the edge is not robust to realistic frictions.

## Return Validation

| Series | Total Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Horizon-matched final | 213.42% | 1.29 | -15.59% |
| Realistic neutralized | -8.24% | -0.04 | -25.17% |

## Deployment Gate

| Gate | Requirement | Result | Decision |
|---|---|---|---|
| Positive IC | Mean IC > 0 | 0.0051 | Pass but weak |
| Stable IC | IC-IR > 0.30 | 0.05 | Fail |
| Costs | Net return survives 7.5 bps | 213.64% | Fail |
| Turnover | Average rebalance turnover below 75% | 126.41% | Fail |
| Drawdown | Max drawdown better than -25% | -15.58% | Pass |

## Interview Answer

The strongest answer is: I would not deploy this strategy. The project is valuable because the validation stack caught weak alpha before deployment. It demonstrates target construction, walk-forward validation, IC diagnostics, cost sensitivity, and research judgment.
