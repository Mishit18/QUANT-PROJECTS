# Market Impact Models: Kyle, Obizhaeva-Wang, and Bouchaud

Comparative quant-research implementation of three market impact models across controlled liquidity regimes. The project calibrates Kyle's Lambda, Obizhaeva-Wang permanent/transient impact, and a Bouchaud-style propagator, then reports parameter stability, constraint checks, and cross-regime validation.

This is positioned as a research and execution-modeling project for quant research and quant trading resumes. It is intentionally honest about where the models work, where they fail, and what assumptions are synthetic.

## Why This Project Matters

Market impact is central to optimal execution, alpha decay, transaction-cost analysis, and capacity estimation. A good quant candidate should be able to move between the theory, the calibration code, and the practical question: "When does this model stop being reliable?"

This project demonstrates:

- Statistical calibration of impact models under different liquidity regimes.
- Constraint-aware estimation for permanent/transient impact.
- Long-memory impact modeling through a propagator kernel.
- Cross-regime validation to expose parameter instability.
- Research reporting with failure modes instead of only best-case metrics.

## Implemented Models

### Kyle's Lambda

```text
Delta P_t = alpha + lambda * Q_t + epsilon_t
```

Kyle's Lambda estimates linear adverse selection from signed order flow to price changes.

The implementation reports lambda, standard error, confidence interval, t-statistic, p-value, R-squared, adjusted R-squared, and sample size for each regime.

### Obizhaeva-Wang

```text
I(t) = gamma * I_0 + (1 - gamma) * I_0 * exp(-rho * t)
```

The Obizhaeva-Wang module decomposes impact into permanent and transient components. It validates the constraints `gamma in [0, 1]`, `rho > 0`, and finite half-life.

### Bouchaud Propagator

```text
G(tau) = A / (tau + tau_0)^beta
Delta P(t) = integral G(t - s) * Q(s) ds
```

The Bouchaud-style model estimates a power-law memory kernel and validates beta constraints, amplitude positivity, kernel stability, and long-memory diagnostics.

## Data Design

The pipeline generates synthetic market data using controlled liquidity regimes:

- Low, medium, and high liquidity regimes.
- 3,000 observations per regime.
- Long-memory order-flow component through fractional Brownian style generation.
- Known regime-dependent impact strengths.
- Reproducible random seed.

Because the data is synthetic, the claims are about modeling, calibration discipline, and failure analysis. The project does not claim live trading profitability or exchange-grade market impact estimation.

## Key Results

### Kyle Calibration

| Regime | Lambda | 95% CI | p-value | R-squared |
| --- | ---: | ---: | ---: | ---: |
| Low | 0.4254 | [0.2835, 0.5672] | 4.56e-09 | 0.0114 |
| Medium | 0.1492 | [0.0575, 0.2410] | 0.0014 | 0.0034 |
| High | 0.0716 | [0.0065, 0.1366] | 0.0310 | 0.0016 |

Interpretation: the impact coefficient is positive and statistically significant in all regimes, but the low R-squared values show that realized short-horizon returns remain extremely noisy.

### Obizhaeva-Wang Calibration

| Regime | Permanent Fraction | Transient Fraction | Half-Life | Constraints |
| --- | ---: | ---: | ---: | --- |
| Low | 10.52% | 89.48% | 0.156 | Pass |
| Medium | 5.81% | 94.19% | 0.151 | Pass |
| High | 3.86% | 96.14% | 0.203 | Pass |

Interpretation: transient impact dominates across regimes, while the permanent fraction rises as liquidity worsens.

### Bouchaud Calibration

| Regime | Amplitude | Beta | Hurst Estimate | Constraints |
| --- | ---: | ---: | ---: | --- |
| Low | 0.0156 | 0.30 | 0.981 | Pass |
| Medium | 0.0282 | 0.80 | 0.999 | Pass |
| High | 0.0078 | 0.80 | 0.989 | Pass |

Interpretation: the project detects long-memory behavior, but Bouchaud fit quality should be read cautiously because the calibrated R-squared values are low. This is useful for interview discussion: detecting memory is not the same as proving a strong predictive impact model.

## Model Selection View

- Use Kyle for simple impact slope estimation and fast regime comparison.
- Use Obizhaeva-Wang when transient decay and recovery speed matter.
- Use Bouchaud only when execution horizon and long-memory diagnostics justify the extra complexity.
- Recalibrate by regime; pooled parameters are misleading in this setup.

## Outputs

Tables:

- `results/tables/kyle_calibration.csv`
- `results/tables/ow_calibration.csv`
- `results/tables/bouchaud_calibration.csv`
- `results/tables/cross_regime_validation.csv`
- `results/tables/parameter_stability.csv`
- `results/tables/efficient_frontier.csv`

Figures:

- `results/figures/kyle_diagnostics.png`
- `results/figures/ow_diagnostics.png`
- `results/figures/bouchaud_diagnostics.png`
- `results/figures/data_generation_diagnostics.png`
- `results/figures/efficient_frontier.png`

Reports:

- `report/market_impact_analysis.md`
- `report/failure_analysis.md`
- `docs/ATS_SCREENING_PACK.md`

## Run Locally

```bash
pip install -r requirements.txt
python main.py
```

The script regenerates calibration tables and prints a concise summary of model parameters, stability, and constraint validation.

## Repository Structure

```text
03_market_impact_models/
|-- src/
|   |-- data_generation.py
|   |-- kyle_model.py
|   |-- obizhaeva_wang.py
|   `-- bouchaud_model.py
|-- results/
|   |-- tables/
|   `-- figures/
|-- report/
|   |-- market_impact_analysis.md
|   `-- failure_analysis.md
|-- docs/
|   `-- ATS_SCREENING_PACK.md
|-- main.py
|-- requirements.txt
`-- README.md
```

## Resume Positioning

Best fit:

- Quant Research
- Quant Trading
- Execution Research
- Systematic Trading Research
- Market Microstructure Analytics

Do not position this as:

- A production execution engine.
- A profitable live-trading strategy.
- A real exchange order-book calibration unless real market data is added.

## References

- Kyle, A. S. (1985). Continuous Auctions and Insider Trading.
- Obizhaeva, A. A., and Wang, J. (2013). Optimal Trading Strategy and Supply/Demand Dynamics.
- Bouchaud, J. P., Gefen, Y., Potters, M., and Wyart, M. (2004). Fluctuations and Response in Financial Markets.
- Gatheral, J. (2010). No-Dynamic-Arbitrage and Market Impact.
- Almgren, R., and Chriss, N. (2001). Optimal Execution of Portfolio Transactions.
