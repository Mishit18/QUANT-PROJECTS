# Cross-Asset Statistical Arbitrage Tear Sheet

## Decision

Reject for deployment after realistic neutralization and transaction costs.

## Hypothesis

Volatility-scaled forward returns may create a better-conditioned cross-sectional alpha target than raw forward returns.

## Validation Design

- Point-in-time feature engineering
- 5-day forward target
- 20-day trailing volatility scaling
- Embargoed walk-forward validation
- OLS, Ridge, and XGBoost comparison
- Market and sector neutralization
- Transaction-cost-aware portfolio test

## Key Metrics

| Metric | Value |
|---|---:|
| XGBoost mean IC | 0.0051 |
| XGBoost IC-IR | 0.05 |
| Hit rate | 52.41% |
| Market + sector neutral gross return | 43.43% |
| Market + sector neutral net return | -8.24% |
| Cost drag | 51.68% |
| Average turnover | 126.73% |
| Estimated capacity | $4.9M |

## Interview Defense

This is a successful research process, not a successful deployable signal. The system caught a weak gross edge before capital deployment. The key lessons are target hygiene, embargoed validation, cost accounting, neutralization, and deploy/reject discipline.

## Additional Evidence Pack

- `reports/RESEARCH_APPENDIX.md`
- `reports/cost_sensitivity.csv`
- `reports/xgboost_ic_timeseries.csv`
- `reports/figures/xgboost_ic_stability.png`
- `reports/figures/transaction_cost_sensitivity.png`
- `reports/figures/final_vs_realistic_equity.png`
- `reports/figures/rebalance_turnover.png`
