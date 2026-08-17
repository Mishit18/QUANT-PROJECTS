# ATS Screening Pack

## Best-Fit Roles

- Quant Research Intern
- Quantitative Research Analyst Intern
- Systematic Strategies Intern
- Data Scientist, Financial Data
- ML Research Intern, Time Series or Finance

## Strong Resume Bullets

- Built a reproducible factor research pipeline for 49 large-cap US equities,
  extracting 10 PCA eigen-portfolio factors that explained 67.12% of return
  variance across 2,514 trading days from 2015-01-05 to 2024-12-30.
- Compared statistical PCA factors against momentum, value, size, quality, and
  low-volatility factors using factor returns, Sharpe ratios, drawdowns,
  R-squared, residual diagnostics, and volatility/market/crisis regime analysis.
- Implemented methodologically correct PCA factor construction by estimating
  covariance on standardized returns and computing factor returns from raw
  returns, preserving economic scale for risk-premia analysis.
- Produced research-ready CSV, JSON, and PNG artifacts including eigen-portfolio
  weights, factor loadings, PCA variance explained, factor alphas/betas,
  residual diagnostics, regime statistics, and risk metrics.
- Added bootstrap Sharpe confidence intervals, eigenportfolio concentration
  checks, and factor decision gates to separate "research further" candidates
  from non-deployable in-sample results.
- Demonstrated honest quant research interpretation: PCA factors achieved 67.0%
  mean regression R-squared but did not guarantee positive premia, while the
  quality factor produced the strongest in-sample Sharpe in the classical set.

## ATS Keywords

PCA, Principal Component Analysis, Eigen-Portfolios, Factor Modeling, Risk
Premia, Cross-Sectional Returns, Time-Series Regression, Fama-MacBeth,
Long-Short Factors, Momentum, Value, Size, Quality, Low Volatility, Regime
Analysis, Volatility Regimes, Crisis Regimes, Risk Model, Covariance Matrix,
Eigenvectors, Factor Loadings, R-Squared, Residual Diagnostics, Sharpe Ratio,
Drawdown, Python, pandas, NumPy, scikit-learn, statsmodels, yfinance,
matplotlib, seaborn, pytest.

## Interview Defense

### What Is The Core Research Question?

The project asks whether equity return structure is better explained by
statistical PCA factors or by economically motivated long-short factors. The
answer is deliberately nuanced: PCA explains covariance well, while classical
factors are easier to interpret as possible premia.

### Why Standardize For PCA But Use Raw Returns For Factor Returns?

Standardization prevents high-volatility assets from dominating covariance
estimation. Applying the resulting eigenvectors to raw returns preserves the
economic scale of the factor return series, which makes Sharpe, drawdown, and
risk-premia estimates meaningful.

### What Worked?

- The end-to-end pipeline runs reproducibly in one command.
- PCA factors explain a large share of cross-sectional return variance.
- Classical factors provide a clear economic comparison set.
- The project exports enough diagnostics to defend the methodology in an
  interview.

### What Did Not Work?

- PCA factors are not reliable alpha factors by themselves.
- Value and low-volatility proxies performed poorly in this sample.
- The quality factor result is strong but should be treated as in-sample
  evidence requiring walk-forward validation.

## Claims To Avoid

- Do not claim PCA discovered profitable alpha.
- Do not claim the quality factor will remain profitable out of sample.
- Do not present price-based factor proxies as equivalent to Compustat/FactSet
  fundamental factors.
- Do not call the system production trading infrastructure.

## Upgrade Path To 100/100

- Add walk-forward factor construction and out-of-sample evaluation.
- Add transaction costs, turnover, borrow constraints, and capacity analysis.
- Replace price-only proxies with real fundamentals for value, quality, and
  profitability.
- Add Fama-French 3/5-factor benchmark comparison.
- Add factor-neutral portfolio optimization and risk attribution.
- Add bootstrap confidence intervals for Sharpe, factor premia, and drawdown.
- Add a Streamlit or notebook-free HTML report for quick reviewer inspection.
