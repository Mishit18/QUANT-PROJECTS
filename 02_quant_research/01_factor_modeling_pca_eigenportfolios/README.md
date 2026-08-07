# Systematic Factor Modeling with PCA Eigen-Portfolios

A quant research project for decomposing equity returns into statistical PCA
factors and economically motivated long-short factors. The project is designed
to show institutional-style factor research: clean data handling, eigen-portfolio
construction, risk-premia estimation, regime analysis, risk controls, plots,
and reproducible outputs.

The main research point is explicit: PCA factors are excellent for risk
decomposition and covariance structure, while classical factors are better
suited for economic interpretation and possible alpha research.

## Why It Matters

- Separates statistical risk factors from economically priced factors.
- Runs PCA on standardized returns, then applies eigenvectors to raw returns so
  factor returns retain meaningful scale.
- Compares PCA factors against momentum, value, size, quality, and low-vol
  factors using Sharpe, drawdown, regressions, and regime behavior.
- Produces CSV/JSON/PNG artifacts suitable for resume screening and interview
  discussion.
- Keeps conclusions honest: high explained variance does not imply positive
  expected return.

## Quick Start

```bash
pip install -r requirements.txt
python tests/test_installation.py
python analysis/run_full_pipeline.py
```

Validated run time in this workspace: about 35 seconds.

## Verified Results

Latest reproduced pipeline run:

- Data: 49 large-cap US equities after filtering
- Period: 2015-01-05 to 2024-12-30
- Trading days: 2,514
- PCA components: 10
- PCA variance explained: 67.12%
- PCA regression mean R-squared: 67.0%
- Classical factor regression mean R-squared: 19.7%
- Regime types analyzed: volatility, market, crisis

### PCA Factor Performance

| Factor | Annual Return | Volatility | Sharpe |
|---|---:|---:|---:|
| PC1 | -98.99% | 107.57% | -0.92 |
| PC2 | -31.90% | 53.92% | -0.59 |
| PC6 | 15.58% | 25.11% | 0.62 |
| PC10 | 26.29% | 36.75% | 0.72 |

Interpretation: the first PCs explain risk, but they are not guaranteed to earn
premia. This is expected in a PCA risk model.

### Classical Factor Performance

| Factor | Annual Return | Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|
| Quality | 60.16% | 13.53% | 4.45 | -9.49% |
| Size | 3.18% | 9.24% | 0.34 | -20.75% |
| Momentum | 1.75% | 14.14% | 0.12 | -29.83% |
| LowVol | -6.55% | 17.78% | -0.37 | -59.02% |
| Value | -41.36% | 14.48% | -2.86 | -98.61% |

Interpretation: the quality proxy dominates in this sample, while value performs
poorly. This should be presented as sample evidence, not as a universal claim.

## Project Structure

```text
analysis/run_full_pipeline.py        End-to-end research pipeline
src/data_pipeline.py                 Data download, cleaning, diagnostics
src/pca_model.py                     PCA extraction and eigen-portfolios
src/factor_construction.py           Momentum, value, size, quality, low-vol
src/regression.py                    Factor betas, alphas, R-squared outputs
src/regime_analysis.py               Volatility, market, and crisis regimes
src/portfolio_controls.py            Vol targeting and risk metrics
src/visualization.py                 Publication-style charts
tests/test_installation.py           Dependency, config, and import smoke tests
reports/RESEARCH_REPORT.md           Longer research write-up
reports/PCA_FACTOR_INTERPRETATION.md Methodology explanation
reports/ATS_SCREENING_PACK.md        Resume bullets and interview defense
results/                             CSV/JSON outputs
plots/                               Generated figures
```

## Methodology

1. Download and clean the equity universe.
2. Compute daily returns, excess returns, diagnostics, and market series.
3. Standardize returns for PCA covariance estimation.
4. Convert eigenvectors into eigen-portfolios.
5. Apply eigen-portfolio weights to raw returns to compute economic factor
   returns.
6. Construct classical long-short factors.
7. Run time-series regressions, risk-premia estimates, and residual diagnostics.
8. Analyze factor behavior across volatility, market, and crisis regimes.
9. Export results and plots.

## Interview Positioning

Use this project for Quant Research, Data Science, and ML-for-finance screening.
The strongest story is not "I found alpha with PCA." The strongest story is:

"I built a reproducible factor research pipeline that distinguishes risk
decomposition from priced premia, verifies factor behavior with regressions and
regime analysis, and reports negative or weak results honestly."

## Limitations

- Uses a limited large-cap equity universe rather than a full institutional
  cross-section.
- Classical factors use price-based proxies, not full fundamental datasets.
- Factor definitions are static and should be walk-forward validated.
- Transaction costs and capacity constraints are simplified.
- The quality result is strong in-sample and should not be overclaimed.

## Documentation

- [RESEARCH_REPORT.md](reports/RESEARCH_REPORT.md)
- [PCA_FACTOR_INTERPRETATION.md](reports/PCA_FACTOR_INTERPRETATION.md)
- [ATS_SCREENING_PACK.md](reports/ATS_SCREENING_PACK.md)

## Verification

```bash
python -m pytest -q
python -m compileall -q .
python analysis/run_full_pipeline.py
```

Latest local verification: 4 tests passed and the full pipeline completed.

## Disclaimer

This project is for research, education, and interview demonstration. It is not
investment advice and is not a live-trading system.
