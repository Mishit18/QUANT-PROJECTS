# Tech Resume Screening Pack

## Resume Positioning

- Built cross-asset ML alpha validation pipeline with point-in-time volatility-scaled targets, embargoed walk-forward validation, and OLS/Ridge/XGBoost model comparison.
- Added IC/IC-IR, hit-rate, t-stat, p-value, market/sector neutralization, turnover, capacity, and transaction-cost diagnostics.
- Implemented deploy/reject gates that rejected the signal after neutralized net return fell to -8.24% under 51.68% cost drag.
- Verified 5/5 pytest tests covering target construction, transaction-cost accounting, deploy gates, and backtest alignment.

## Verified Evidence

- Test suite: 5/5 passed in 6.42s.
- XGBoost mean IC: 0.0051.
- XGBoost IC-IR: 0.05.
- Hit rate: 52.41%.
- Market + sector neutral gross return: 43.43%.
- Market + sector neutral net return: -8.24%.
- Cost drag: 51.68%.
- Average turnover: 126.73%.
- Estimated capacity: $4.9M.

## Interview Defense

This is a strong project because it does not overclaim alpha. It shows how a candidate thinks like a professional research engineer: build leakage-safe labels, validate out of sample, neutralize risk, charge costs, evaluate stability, and reject weak models before deployment.

## Honest Scope

The current result uses synthetic data and is not a live trading strategy. The value is the validation framework and deploy/reject discipline, not a profitable alpha claim.

## Resume-Safe Bullets

- Built cross-asset ML alpha validation pipeline with point-in-time targets, embargoed walk-forward testing, OLS/Ridge/XGBoost comparison, and IC/IC-IR diagnostics.
- Added market/sector neutralization, turnover, capacity, and transaction-cost gates; rejected signal after net return fell to -8.24% under 51.68% cost drag.
- Verified target construction, backtest alignment, transaction-cost accounting, and deploy/reject gates through 5 passing pytest tests.
