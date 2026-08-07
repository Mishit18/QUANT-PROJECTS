# PCA Factor Interpretation

## Core Lesson

PCA factors are statistical risk factors. They are useful because they explain
covariance structure, not because they automatically earn positive returns.
That distinction is the most important interview point in this project.

## The Common Mistake

A naive PCA implementation often does this:

1. Standardize returns.
2. Fit PCA on standardized returns.
3. Compute factor returns from standardized returns.

That makes factor returns lose economic scale. Since standardized returns are
centered, factor means can be close to zero by construction, and Sharpe ratios
can become meaningless.

## The Corrected Approach

This project separates covariance estimation from economic return computation:

1. Standardize returns only to estimate the correlation/covariance structure.
2. Extract PCA eigenvectors from that standardized matrix.
3. Treat eigenvectors as eigen-portfolio weights.
4. Apply those weights to raw returns to compute factor return series.

```text
R_std = standardize(R_raw)
Sigma = cov(R_std)
Sigma = V Lambda V'
F = R_raw @ V
```

This preserves the useful statistical property of PCA while keeping factor
returns interpretable in return, volatility, Sharpe, and drawdown terms.

## Three-Lens Evaluation

### Statistical Importance

Questions:

- How much variance does each component explain?
- Are the components stable over rolling windows?
- Does the factor capture broad market or sector covariance?

Success metric: variance explained and stability.

### Explanatory Power

Questions:

- How well do factors explain asset returns in time-series regressions?
- What is the mean R-squared across assets?
- Do PCA factors explain more variation than classical factors?

Success metric: R-squared and residual diagnostics.

### Economic Relevance

Questions:

- Does the factor earn a positive premium?
- What is the Sharpe ratio?
- How large is the drawdown?
- Is performance regime dependent?

Success metric: return, Sharpe, drawdown, and robustness.

## Verified Findings

- Ten PCA components explain 67.12% of return variance.
- PCA factor regressions reach about 67.0% mean R-squared across assets.
- Classical factor regressions reach about 19.7% mean R-squared.
- PCA factors are therefore stronger for risk decomposition.
- The quality factor is the strongest classical factor in this sample.
- PCA should not be presented as direct alpha discovery.

## PCA vs Classical Factors

| Dimension | PCA Factors | Classical Factors |
|---|---|---|
| Construction | Data-driven covariance | Economic or empirical anomaly |
| Orthogonality | Yes | No |
| Main use | Risk modeling | Premia and alpha research |
| Interpretation | Statistical | Economic |
| Success metric | Variance explained, R-squared | Sharpe, drawdown, robustness |
| Expected returns | Mixed | Positive by hypothesis |

## Interview Answers

### Why Did You Use PCA?

To identify dominant directions of cross-sectional risk and build an
orthogonal factor representation of the equity universe.

### Why Not Trade PCA Factors Directly?

Because PCA maximizes explained variance, not expected return. A PCA factor can
be important for risk and still have zero or negative expected return.

### Why Compare Against Classical Factors?

Classical factors provide an economic benchmark. The comparison shows the
difference between explaining return variance and earning risk premia.

### How Would You Improve It?

- Add walk-forward out-of-sample evaluation.
- Add transaction costs and turnover-aware portfolio construction.
- Replace price-only proxies with fundamentals.
- Compare against Fama-French 3-factor and 5-factor models.
- Add confidence intervals for Sharpe and factor premia.
- Build a hybrid model using PCA residuals after removing known factors.

## Final Takeaway

The strongest claim is not that PCA found alpha. The strongest claim is that
the project implements a defensible research workflow that separates risk
decomposition, explanatory power, and economic premia.
