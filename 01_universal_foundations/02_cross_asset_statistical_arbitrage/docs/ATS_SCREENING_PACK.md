# ATS Screening Pack: Cross-Asset Statistical Arbitrage

## Best-Fit Resume Profiles

- Quant Research
- Quant Trader
- Systematic Trading Research
- Data Scientist, Markets
- Portfolio Research

## Recruiter Summary

Built a cross-sectional statistical arbitrage research framework with point-in-time volatility-scaled targets, embargoed walk-forward validation, OLS/Ridge/XGBoost modeling, IC stability diagnostics, market/sector neutralization, horizon-matched execution, transaction-cost modeling, and honest rejection of a weak synthetic alpha.

## ATS Keyword Coverage

Cross-sectional alpha, statistical arbitrage, factor modeling, volatility-scaled targets, point-in-time features, forward returns, embargoed walk-forward validation, overlapping labels, information coefficient, IC-IR, hit rate, t-statistic, p-value, risk neutralization, market beta neutralization, sector neutralization, XGBoost, Ridge regression, OLS, transaction costs, turnover, slippage, capacity, horizon-matched execution, backtesting, Python, pandas, NumPy, scikit-learn, pytest.

## Quant Research Resume Bullets

- Built cross-sectional alpha research framework with corrected point-in-time volatility-scaled 5-day targets, embargoed walk-forward validation, and OLS/Ridge/XGBoost model comparison.
- Diagnosed weak synthetic alpha with XGBoost mean IC 0.0051, IC-IR 0.05, 52.41% hit rate, and factor-neutralized IC retention checks.
- Demonstrated that neutralized gross returns do not survive realistic trading costs: market+sector neutral Sharpe 0.44 before costs falls to -0.04 after 44.67% total cost drag.
- Added focused pytest coverage for target construction, target coverage, transaction-cost union accounting, and next-period backtest alignment.

## Quant Trader Resume Bullets

- Implemented cost-aware alpha-to-PnL validation with turnover, slippage, transaction costs, capacity estimate, and realistic neutralized portfolio construction.
- Compared daily versus horizon-matched execution, showing 5-day rebalancing improves synthetic diagnostic Sharpe from 0.06 to 1.29 while keeping the deployability verdict rejected.
- Added market and sector neutralization diagnostics to distinguish raw signal behavior from risk-factor exposure.
- Froze and documented the synthetic signal as rejected for deployment despite positive horizon-matched diagnostic performance.

## Strong Interview Defense

The strongest explanation is:

1. Target construction must be point-in-time; otherwise forward-return labels can contaminate conditioning.
2. Overlapping 5-day labels require an embargo in walk-forward validation.
3. Small IC can be statistically significant but still economically useless after costs.
4. Horizon matching can improve alpha-to-PnL translation, but it is not enough if realistic neutralized/costed validation fails.
5. Rejecting a signal is a successful research outcome when the acceptance gates are explicit.

## Claims To Avoid

- Do not claim deployable alpha.
- Do not claim real-market validation; current results use synthetic data.
- Do not present horizon-matched diagnostic as the final approval test.
- Do not hide the realistic neutralized/costed failure.

## Upgrade Path For 100/100 Screening

- Replace synthetic OHLCV with survivorship-aware real market data.
- Add borrow/funding/liquidity constraints.
- Add FDR/multiple-testing control across candidate signals.
- Add beta, sector, style, and PCA risk attribution in the final costed portfolio.
- Add a formal research memo with accept/reject gates and experiment lineage.
