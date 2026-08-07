# Research Report: PCA Eigen-Portfolios and Classical Equity Factors

## Executive Summary

This project builds a reproducible factor research pipeline for a 49-stock
large-cap US equity universe. It compares two factor-modeling approaches:

- Statistical PCA eigen-portfolios for risk decomposition.
- Classical long-short factors for economic premia research.

The core finding is that PCA factors explain return covariance much better than
classical factors, while classical factors are easier to interpret as potential
risk premia. PCA should therefore be positioned as a risk-modeling tool, not as
standalone alpha discovery.

## Verified Run

Latest local run:

- Period: 2015-01-05 to 2024-12-30
- Trading days: 2,514
- Assets after filtering: 49
- PCA components: 10
- PCA variance explained: 67.12%
- PCA regression mean R-squared: about 67.0%
- Classical factor regression mean R-squared: about 19.7%
- Regime types: volatility, market, crisis

## Research Design

### Data Pipeline

The pipeline downloads a large-cap equity universe, filters incomplete series,
computes returns and excess returns, and exports diagnostics. BRK.B is dropped
by the data provider in the verified run, leaving 49 assets with sufficient
history.

### PCA Model

The PCA model is intentionally built in two stages:

```text
R_std = standardize(R_raw)
Sigma = cov(R_std)
Sigma = V Lambda V'
F = R_raw @ V
```

Standardized returns are used only to estimate covariance structure. The final
factor return series use raw returns so that Sharpe ratios, drawdowns, and
premia remain economically meaningful.

### Classical Factors

The project constructs five price-based long-short factors:

- Momentum
- Value
- Size
- Quality
- Low volatility

These are useful as economic benchmarks against the statistical PCA factors.

### Regression And Regime Analysis

The pipeline estimates factor betas, alphas, R-squared values, residual
diagnostics, and regime-conditional statistics. Regimes are defined using
volatility, market trend, and crisis conditions.

## Results

### PCA Factors

| Factor | Annual Return | Volatility | Sharpe |
|---|---:|---:|---:|
| PC1 | -98.99% | 107.57% | -0.92 |
| PC2 | -31.90% | 53.92% | -0.59 |
| PC3 | -18.06% | 36.55% | -0.49 |
| PC6 | 15.58% | 25.11% | 0.62 |
| PC10 | 26.29% | 36.75% | 0.72 |

PCA components are valuable because they explain covariance structure. Their
mixed Sharpe ratios are expected and should not be treated as a failure.

### Classical Factors

| Factor | Annual Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Quality | 60.16% | 13.53% | 4.45 | -9.49% |
| Size | 3.18% | 9.24% | 0.34 | -20.75% |
| Momentum | 1.75% | 14.14% | 0.12 | -29.83% |
| LowVol | -6.55% | 17.78% | -0.37 | -59.02% |
| Value | -41.36% | 14.48% | -2.86 | -98.61% |

The quality proxy dominates this sample. The result is useful for discussion,
but it should be presented as in-sample evidence that needs walk-forward
validation.

## Interpretation

PCA and classical factors answer different questions.

| Question | Better Tool |
|---|---|
| What explains covariance? | PCA |
| What are dominant risk directions? | PCA |
| What has economic interpretation? | Classical factors |
| What might earn premia? | Classical factors |
| What helps build a hybrid risk model? | Both |

This distinction is the heart of the project. PCA explains risk; it does not
guarantee alpha.

## Limitations

- The equity universe is limited to 49 surviving large-cap names.
- Classical factors use price-based proxies rather than full fundamentals.
- Results are in-sample and should not be overclaimed.
- Transaction costs, turnover, capacity, and borrow constraints are simplified.
- PCA loadings are static; a production model would require walk-forward
  refitting and stability checks.

## Upgrade Path

- Add walk-forward out-of-sample testing.
- Add transaction-cost-aware portfolio construction.
- Add Fama-French 3-factor and 5-factor comparisons.
- Replace price-only proxies with fundamentals.
- Add confidence intervals for Sharpe and factor premia.
- Use PCA on residuals after removing known factors.
- Add a formal research decision memo with accept/reject criteria.

## Conclusion

This is a strong quant research screening project because it is methodologically
careful, reproducible, and honest. The best interview framing is that it shows
the difference between risk decomposition and alpha research, then builds the
pipeline needed to evaluate both.
