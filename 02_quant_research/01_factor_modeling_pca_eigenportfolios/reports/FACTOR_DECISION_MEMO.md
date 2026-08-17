# Factor Research Decision Memo

## Research Question

Do PCA eigen-portfolios or classical factors produce enough evidence to be treated as deployable alpha factors?

## Answer

No factor should be presented as production-ready alpha from this evidence alone. The project is strongest as a risk-decomposition and factor-research workflow. Factors with positive bootstrap lower bounds can be researched further, but still require walk-forward construction, turnover, costs, borrow constraints, and capacity checks.

## Top Factors By Sharpe

| Factor | Sharpe | 5% CI | 95% CI | Max Drawdown | Hit Rate | Decision |
|---|---:|---:|---:|---:|---:|---|
| Quality | 4.45 | 3.93 | 4.97 | -9.49% | 60.70% | research further |
| PC10 | 0.72 | 0.21 | 1.26 | -60.40% | 51.51% | do not deploy |
| PC6 | 0.62 | 0.09 | 1.14 | -44.25% | 51.95% | do not deploy |
| PC8 | 0.46 | -0.06 | 0.96 | -34.79% | 51.59% | do not deploy |
| Size | 0.34 | -0.17 | 0.87 | -20.09% | 48.25% | do not deploy |

## Concentration Checks

| Factor | Top-5 Weight Share | Max Abs Weight | Long Count | Short Count |
|---|---:|---:|---:|---:|
| PC1 | 11.17% | 2.23% | 0 | 49 |
| PC2 | 14.99% | 3.00% | 27 | 22 |
| PC3 | 17.04% | 3.41% | 20 | 29 |
| PC4 | 19.07% | 3.81% | 25 | 24 |
| PC5 | 18.56% | 3.71% | 25 | 24 |
| PC6 | 18.40% | 3.68% | 32 | 17 |
| PC7 | 18.46% | 3.69% | 25 | 24 |
| PC8 | 20.03% | 4.01% | 24 | 25 |
| PC9 | 17.05% | 3.41% | 20 | 29 |
| PC10 | 20.92% | 4.18% | 24 | 25 |

## Interview Framing

PCA explains covariance, not alpha. A strong PCA component can be useful for hedging, stress testing, and risk attribution even if its realized return is negative. Classical factors are more interpretable, but the strong in-sample quality result must not be overclaimed without walk-forward validation.

## Added Artifacts

- `results/factor_bootstrap_confidence_intervals.csv`
- `results/eigenportfolio_concentration.csv`
- `results/factor_decision_gates.csv`
- `plots/bootstrap_sharpe_intervals.png`
