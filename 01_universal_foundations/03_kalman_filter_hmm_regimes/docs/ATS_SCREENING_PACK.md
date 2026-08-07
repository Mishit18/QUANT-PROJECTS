# ATS Screening Pack

## Best-Fit Roles

- Quant Research Intern
- Quant Trader Intern
- Systematic Strategies Intern
- Portfolio and Risk Research Intern
- Machine Learning for Finance Intern

## Resume Positioning

This project should be framed as a production-style regime research and risk
diagnostics project, not as a profitable trading strategy. Its strength is that
it combines state-space modelling, unsupervised regime discovery, transaction
costs, benchmark comparison, and honest failure analysis on real market data.

## Strong Resume Bullets

- Built a state-space and HMM regime framework for SPY using Kalman latent trend
  estimation, volatility-ordered Gaussian regimes, filtered regime probabilities,
  TSMOM overlays, position clipping, and transaction-cost-aware backtesting.
- Evaluated SPY/QQQ/TLT data from 2021-04-13 to 2026-04-10 with 5 bps turnover
  costs; active regime strategy reduced annualized volatility to 8.18% versus
  17.05% for buy-and-hold but underperformed on Sharpe (-0.35 versus 0.67).
- Added production research hygiene: cache-first Yahoo Finance data loading,
  covariance floors, regime-collapse warnings, structured logs, headless plot
  generation, result CSV exports, and a 17-test validation suite.
- Documented model limitations including HMM non-convergence after 50 EM
  iterations, a sparse crisis regime with 0.96% frequency, strategy
  underperformance, and the need for walk-forward validation before deployment.

## ATS Keywords

Kalman Filter, Hidden Markov Model, HMM, State-Space Model, Regime Detection,
Market Regimes, Time-Series Momentum, TSMOM, Statistical Arbitrage, Systematic
Trading, Quant Research, Portfolio Construction, Transaction Costs, Backtesting,
Risk Management, Volatility Regimes, Filtering, EM Algorithm, Gaussian
Mixtures, Covariance Regularization, Python, NumPy, pandas, scikit-learn,
matplotlib, pytest, yfinance.

## Interview Defense

### What Problem Does It Solve?

The project tests whether latent trend and volatility regimes can improve SPY
timing after realistic turnover costs. It is useful as a research framework
because it separates signal generation, regime inference, execution assumptions,
and benchmark comparison.

### What Worked?

- The pipeline ran end to end on real cached market data.
- The active strategy lowered realized volatility and max drawdown versus the
  failed raw Kalman trend baseline.
- The project produces reproducible plots, CSVs, logs, tests, and warnings.

### What Did Not Work?

- Buy-and-hold remained superior on Sharpe and annualized return.
- The HMM did not converge within the configured 50 EM iterations.
- The crisis regime was too sparse for reliable inference.
- The active signal incurred high turnover relative to its return edge.

### Why Is This Still Valuable?

In quant research, rejecting a weak strategy with evidence is valuable. The
project shows that the candidate can avoid overclaiming, account for costs,
compare against a benchmark, and convert a research idea into a defensible
production-style experiment.

## Claims To Avoid

- Do not claim the project finds profitable alpha.
- Do not claim the strategy beats SPY buy-and-hold.
- Do not describe the HMM regimes as stable production labels.
- Do not claim live-trading readiness.
- Do not hide the HMM convergence warning or sparse crisis regime.

## Upgrade Path To 100/100

- Add walk-forward retraining with expanding and rolling windows.
- Enforce strictly filtered-only regime probabilities at every signal timestamp.
- Compare Gaussian HMM against Student-t, Markov-switching regression, and
  volatility-threshold baselines.
- Add multi-asset allocation across SPY, QQQ, TLT, GLD, and sector ETFs.
- Add turnover-aware optimization that explicitly penalizes signal changes.
- Report confidence intervals for Sharpe, drawdown, and turnover-adjusted return.
- Add an experiment registry with config hashes, data hashes, and result hashes.
- Add a final "research decision memo" that states reject, iterate, or deploy
  based on pre-defined acceptance criteria.
