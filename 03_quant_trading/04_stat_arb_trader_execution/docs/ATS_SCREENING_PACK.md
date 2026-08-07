# ATS Screening Pack: Statistical Arbitrage Research Framework

## Best-Fit Resume Profiles

- Quant Research
- Quant Trader
- Systematic Trading Research
- Statistical Arbitrage Research
- Portfolio Research

## Recruiter Summary

Built an end-to-end statistical arbitrage research framework with Engle-Granger cointegration screening, Kalman hedge-ratio estimation, OU spread modeling, HMM regime gating, transaction-cost modeling, equal-risk portfolio construction, rolling-window diagnostics, Monte Carlo risk analysis, and transparent negative-result reporting.

## ATS Keyword Coverage

Statistical arbitrage, pairs trading, cointegration, Engle-Granger, Ornstein-Uhlenbeck, OU process, half-life, Kalman filter, dynamic hedge ratio, HMM, regime detection, z-score signals, transaction costs, slippage, equal-risk allocation, portfolio construction, rolling-window backtest, Monte Carlo risk, Sharpe ratio, Sortino ratio, max drawdown, Python, pandas, NumPy, statsmodels, scikit-learn, hmmlearn.

## Quant Research Resume Bullets

- Built statistical arbitrage research framework screening 780 equity pairs with Engle-Granger cointegration tests, half-life classification, Kalman hedge ratios, and OU spread validation.
- Selected 5 portfolio pairs from 27 statistically valid candidates, then applied HMM regime gating, transaction costs, slippage, equal-risk allocation, and rolling-window diagnostics.
- Reported negative verified result transparently: net Sharpe -0.678, max drawdown -32.03%, 77.4% Monte Carlo loss probability, showing disciplined rejection of weak signals.
- Fixed pandas compatibility issue in regime-gating pipeline and made runtime failures return nonzero exit codes for reproducible research.

## Quant Trader Resume Bullets

- Implemented cost-aware pairs-trading backtest with asymmetric z-score entry/exit, stop-loss logic, time-to-reversion exits, and OU collapse detection.
- Built portfolio-level statistical arbitrage engine using equal-risk allocation rather than cherry-picking pairs by historical performance.
- Added defensive HMM regime gating to block new positions during volatile regimes where OU stationarity assumptions are weaker.
- Produced pair-level attribution and diagnostics to identify where statistical signal quality fails after execution costs.

## Strong Interview Defense

The strongest explanation is:

1. Cointegration alone is not tradability.
2. Kalman filtering can expose instability in a relationship that looked valid statically.
3. OU half-life and R-squared are more important than a lucky Sharpe.
4. Regime filters should enforce model validity, not create fitted trading signals.
5. A negative result is valuable if the research process is disciplined and reproducible.

## Claims To Avoid

- Do not claim the strategy is profitable.
- Do not claim production readiness.
- Do not claim the backtest meets target Sharpe or drawdown.
- Do not hide the verified negative result; it is the strongest evidence of research honesty.

## Upgrade Path For 100/100 Screening

- Add walk-forward universe selection without look-ahead bias.
- Add borrow/funding costs and liquidity filters.
- Add false-discovery-rate control for multiple cointegration tests.
- Add sector-neutral and beta-neutral portfolio constraints.
- Add robust covariance estimation for pair allocation.
