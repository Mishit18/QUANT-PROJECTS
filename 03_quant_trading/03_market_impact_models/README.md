# Market Impact Models: Comparative Analysis

Rigorous implementation and validation of three market impact models: Kyle's Lambda, Obizhaeva-Wang, and Bouchaud Propagator.

## Overview

This project provides a research-grade analysis of market impact across three theoretical frameworks. Each model is calibrated with statistical rigor, validated across liquidity regimes, and subjected to explicit failure mode analysis.

**Models Implemented:**
- Kyle's Lambda (1985): Linear adverse selection
- Obizhaeva-Wang (2013): Permanent/transient decomposition
- Bouchaud Propagator (2004): Power-law memory kernel

## Mathematical Framework

### Kyle's Lambda

```
ΔP_t = α + λ × Q_t + ε_t
```

Linear impact from adverse selection. Assumes informed traders reveal information through order flow.

**Parameters:**
- λ: Adverse selection coefficient (must be > 0)
- α: Intercept (drift term)
- ε: i.i.d. Gaussian noise

**Valid when:** Small-to-medium orders, liquid markets, short horizons

### Obizhaeva-Wang

```
I(t) = γ × I₀ + (1-γ) × I₀ × exp(-ρt)
```

Decomposes impact into permanent (information) and transient (liquidity) components.

**Parameters:**
- γ ∈ [0,1]: Permanent fraction
- ρ > 0: Decay rate (resilience)
- Half-life = ln(2)/ρ

**Valid when:** Transient impact dominates, exponential decay holds, medium orders

### Bouchaud Propagator

```
G(τ) = A / (τ + τ₀)^β
ΔP(t) = ∫₀ᵗ G(t-s) × Q(s) ds
```

Power-law kernel captures long-memory effects beyond exponential decay.

**Parameters:**
- β ∈ [0.3, 0.8]: Power-law exponent
- A > 0: Amplitude
- τ₀ = 1.0: Regularization cutoff

**Valid when:** Long memory detected (Hurst > 0.7), slow decay (β < 0.5), thin markets

## Data Generation

Synthetic data uses fractional Brownian motion (fBm) with controlled long-memory properties:

- Order flow: 30% informed (H = 0.7) + 70% noise (H = 0.5)
- Price dynamics: ΔP = λ × Q + σ × ε
- Three liquidity regimes: low (λ = 0.002), medium (λ = 0.001), high (λ = 0.0005)
- 3,000 periods per regime

This ensures λ > 0 by construction and provides controlled autocorrelation structure for model validation.

## Installation

```bash
pip install -r requirements.txt
```

Requirements: numpy, pandas, scipy, matplotlib

## Usage

Run complete analysis:

```bash
python main.py
```

Executes:
1. Data generation with regime switching
2. Model calibration (Kyle, OW, Bouchaud) with confidence intervals
3. Parameter stability analysis across regimes
4. Cross-regime validation
5. Constraint validation (γ ∈ [0,1], β ∈ [0.3,0.8])

## Results

### Calibration Tables

Saved to `results/tables/`:
- `kyle_calibration.csv`: λ estimates, standard errors, 95% CI, R²
- `ow_calibration.csv`: γ, ρ, half-life, constraint validation
- `bouchaud_calibration.csv`: A, β, Hurst estimates, long-memory detection
- `cross_regime_validation.csv`: Out-of-sample errors
- `parameter_stability.csv`: CV and stability metrics

### Diagnostic Plots

Saved to `results/figures/`:
- `kyle_diagnostics.png`: λ by regime, residual diagnostics
- `ow_diagnostics.png`: Impact decomposition, decay curves
- `bouchaud_diagnostics.png`: Power-law validation, memory analysis
- `data_generation_diagnostics.png`: Autocorrelation, impact linearity

### Reports

Comprehensive analysis in `report/`:
- `market_impact_analysis.md`: Full research report with model comparison
- `failure_analysis.md`: Systematic failure mode analysis with mitigation strategies

## Key Findings

### Kyle's Lambda
- Positive and significant across all regimes (p < 0.001)
- Linearity holds for 75% of orders (Q1-Q3)
- Breaks down for large orders (Q4-Q5): λ increases 7× in low liquidity
- R² < 2% is realistic—noise dominates impact in liquid markets

### Obizhaeva-Wang
- All constraints satisfied: γ ∈ [4%, 11%], ρ > 0
- Transient impact dominates: 89-96% of impact decays
- Fast recovery: half-life < 0.2 time units
- Regime-dependent: γ varies 2.7× across regimes (CV = 0.51)

### Bouchaud Propagator
- Power-law validated: log-log R² = 1.0
- Long memory detected: Hurst > 0.98 for all regimes
- Regime-dependent decay: β = 0.30 (low liquidity) vs β = 0.80 (high liquidity)
- Necessary only for slow decay (β < 0.5) and long horizons

### Model Selection
- Small orders (< 75th percentile): Kyle sufficient
- Medium orders with decay: OW preferred
- Large orders with long execution: Bouchaud (if β < 0.5)
- All models require regime-specific calibration

## Model Limitations

### Kyle
- Non-linearity for large orders (> 75th percentile)
- Parameter instability across regimes (CV = 0.67)
- Cannot capture transient impact decay

### Obizhaeva-Wang
- Assumes exponential decay (may be power-law)
- Regime-dependent parameters (CV = 0.51)
- Time-varying γ and ρ within regimes

### Bouchaud
- Computational cost: O(T²) vs OW's O(1)
- Parameter instability (CV = 0.46)
- Valid only for execution < 60 periods
- Overkill when β > 0.7 (exponential sufficient)

## Documentation

**For detailed analysis, see:**
- `report/market_impact_analysis.md`: Complete research report with mathematical derivations, calibration results, economic interpretation, and model comparison
- `report/failure_analysis.md`: Systematic failure mode analysis with quantitative evidence, economic explanations, and mitigation strategies

**For implementation details, see:**
- `src/data_generation.py`: fBm generation with regime switching
- `src/kyle_model.py`: OLS estimation with statistical tests
- `src/obizhaeva_wang.py`: Constrained optimization with decay validation
- `src/bouchaud_model.py`: Convolution-based calibration with regularization
- `main.py`: End-to-end pipeline orchestration

## Project Structure

```
market-impact-modeling/
├── src/
│   ├── data_generation.py      # fBm data with regime switching
│   ├── kyle_model.py            # Linear impact with statistical tests
│   ├── obizhaeva_wang.py        # Permanent/transient decomposition
│   └── bouchaud_model.py        # Power-law kernel with regularization
├── results/
│   ├── tables/                  # Calibration results (CSV)
│   └── figures/                 # Diagnostic plots (PNG)
├── report/
│   ├── market_impact_analysis.md    # Research report
│   └── failure_analysis.md          # Failure mode analysis
├── main.py                      # Main execution script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## References

1. Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." Econometrica, 53(6), 1315-1335.

2. Obizhaeva, A. A., & Wang, J. (2013). "Optimal Trading Strategy and Supply/Demand Dynamics." Journal of Financial Markets, 16(1), 1-32.

3. Bouchaud, J. P., Gefen, Y., Potters, M., & Wyart, M. (2004). "Fluctuations and Response in Financial Markets: The Subtle Nature of 'Random' Price Changes." Quantitative Finance, 4(2), 176-190.

4. Gatheral, J. (2010). "No-Dynamic-Arbitrage and Market Impact." Quantitative Finance, 10(7), 749-759.

5. Almgren, R., & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions." Journal of Risk, 3, 5-40.

## License

MIT License - See LICENSE file for details
