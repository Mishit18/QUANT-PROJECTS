# ATS Screening Pack: Market Impact Models

## Best-Fit Resume Profiles

- Quant Research
- Quant Trader
- Execution Research
- Systematic Trading Research
- Market Microstructure Analytics

## Recruiter Summary

Implemented and compared Kyle's Lambda, Obizhaeva-Wang, and Bouchaud market-impact models across synthetic liquidity regimes, with statistical calibration, confidence intervals, constraint validation, parameter-stability analysis, cross-regime validation, and failure-mode reporting.

## ATS Keyword Coverage

Market impact, optimal execution, transaction cost analysis, TCA, market microstructure, Kyle's Lambda, Obizhaeva-Wang, Bouchaud propagator, permanent impact, transient impact, order flow, liquidity regimes, regime switching, statistical calibration, OLS, confidence intervals, p-values, R-squared, constrained optimization, power-law decay, long memory, Hurst exponent, cross-regime validation, parameter stability, model risk, failure analysis, Python, NumPy, pandas, SciPy, Matplotlib.

## Quant Research Resume Bullets

- Implemented comparative market-impact research framework covering Kyle's Lambda, Obizhaeva-Wang permanent/transient impact, and Bouchaud propagator models across low-, medium-, and high-liquidity regimes.
- Calibrated impact parameters with confidence intervals, p-values, constraint checks, long-memory diagnostics, and cross-regime validation to expose regime dependence and model instability.
- Found statistically significant positive Kyle impact coefficients across regimes while documenting low explanatory power, highlighting the difference between significant impact and noisy short-horizon return prediction.
- Produced failure-mode analysis covering large-order non-linearity, transient-decay assumptions, power-law overfitting risk, and need for regime-specific recalibration.

## Quant Trader Resume Bullets

- Built market-impact and execution-cost modeling pipeline for transaction-cost analysis, estimating permanent/transient impact and long-memory decay under controlled liquidity regimes.
- Compared model selection rules for simple impact slope estimation, transient recovery, and long-horizon execution where power-law memory may matter.
- Validated that transient impact dominates in the synthetic setup, with Obizhaeva-Wang permanent fraction ranging from 3.86% to 10.52% and all model constraints satisfied.
- Framed results for execution decision-making by separating robust calibration signals from weak predictive fit and synthetic-data limitations.

## Strong Interview Defense

Use this project to show that you understand both quant theory and model risk. The strongest explanation is:

1. Kyle is useful as a first-pass impact slope, but low R-squared means returns are still dominated by noise.
2. Obizhaeva-Wang adds economic structure by separating permanent and transient components.
3. Bouchaud is more flexible for long-memory impact, but flexibility increases calibration risk.
4. Cross-regime validation is essential because pooled impact parameters can hide liquidity-specific behavior.
5. A model can pass constraints and still be commercially weak if predictive fit or execution realism is poor.

## Claims To Avoid

- Do not claim live trading profitability.
- Do not claim production execution readiness.
- Do not claim calibration on real exchange order-book data unless that data is added.
- Do not oversell the Bouchaud model; the long-memory diagnostic is useful, but the fitted R-squared is weak in this synthetic setup.

## Upgrade Path For 100/100 Screening

- Add real TAQ/order-book data or a public LOB dataset.
- Add Almgren-Chriss optimal execution benchmark and compare cost/risk frontier.
- Add walk-forward calibration with rolling liquidity regimes.
- Add participation-rate and order-size stress tests.
- Add implementation shortfall simulation with realistic spread, slippage, and fill assumptions.
