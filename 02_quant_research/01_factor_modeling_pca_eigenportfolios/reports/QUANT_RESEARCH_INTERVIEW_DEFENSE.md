# Quant Research Interview Defense

## 30-Second Pitch

I built a factor research pipeline that separates PCA risk decomposition from economically motivated factor premia. The project shows that PCA can explain covariance structure well, but explained variance is not the same thing as expected return. I compare PCA factors against classical momentum, value, size, quality, and low-volatility factors using regressions, Sharpe, drawdown, residual diagnostics, and regime analysis.

## Key Numbers

| Metric | Value |
|---|---:|
| Equity universe | 49 large-cap stocks |
| Sample period | 2015-01-05 to 2024-12-30 |
| Trading days | 2,514 |
| PCA components | 10 |
| PCA variance explained | 67.12% |
| PCA regression mean R-squared | 67.0% |
| Classical factor regression mean R-squared | 19.7% |
| Best in-sample classical factor | Quality |
| Quality Sharpe | 4.45 |

## Main Research Lesson

PCA factors are strong for risk decomposition and covariance modeling. They are not automatically alpha factors because high variance explained does not imply positive expected return. Classical factors are more interpretable but can still be sample-specific and need out-of-sample validation.

## Likely Interview Questions

**Why standardize returns before PCA?**

Without standardization, high-volatility stocks dominate the covariance matrix. Standardizing lets PCA find common structure. The project then maps eigenvectors back to raw returns so factor returns have economic scale.

**Why did PC1 have poor returns despite explaining risk?**

PC1 captures a large common risk mode. A risk mode can have negative realized premium over a sample. PCA is unsupervised, so it optimizes variance explained, not expected return.

**Can you trade these factors?**

Not directly from this evidence. A production strategy would require out-of-sample factor construction, transaction costs, turnover constraints, capacity analysis, and robust risk controls.

**Why include regime analysis?**

A factor can look strong unconditionally but fail during high-volatility, bear-market, or crisis regimes. Regime analysis helps identify fragility before deployment.

## Resume-Safe Bullet

Built factor research pipeline on 49 large-cap equities comparing PCA eigen-portfolios against momentum, value, size, quality, and low-volatility factors; 10 PCs explained 67.12% variance and about 67% mean R-squared, while results showed PCA is stronger for risk decomposition than standalone alpha discovery.
