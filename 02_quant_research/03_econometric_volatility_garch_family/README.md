# Econometric Volatility Modeling

Production-grade implementation of classical GARCH-family models with rolling forecasts, risk metrics, and rough volatility benchmarking.

## Overview

This project implements a complete volatility forecasting and risk evaluation system for financial returns. It compares classical econometric models (ARCH, GARCH, EGARCH, GJR-GARCH, HARCH) against rough volatility benchmarks (rBergomi) using formal statistical tests and comprehensive backtesting procedures.

**Key features**:
- Maximum likelihood estimation with full diagnostics
- Rolling out-of-sample forecasts with Diebold-Mariano tests
- VaR/ES calculation and backtesting (Kupiec, Christoffersen tests)
- Rough volatility simulation and structural benchmarking
- Production-quality code with comprehensive documentation

## Motivation

Financial returns exhibit well-documented stylized facts that violate constant volatility assumptions:
- Fat tails and volatility clustering
- Leverage effects (asymmetric response to shocks)
- Long memory in volatility

Classical GARCH models capture these features through conditional heteroskedasticity but exhibit systematic limitations at short maturities. This project implements both classical and rough volatility approaches to understand where each framework succeeds and fails.

## Data

- **Asset**: S&P 500 Index (^GSPC)
- **Frequency**: Daily
- **Period**: 2010-present
- **Returns**: Log returns r_t = log(P_t / P_{t-1})

Data preprocessing includes outlier detection, missing value handling, and demeaning for model estimation.

## Models Implemented

### Classical Econometric Models

**ARCH(q)** - Engle (1982)
```
σ²_t = ω + Σ α_i ε²_{t-i}
```

**GARCH(p,q)** - Bollerslev (1986)
```
σ²_t = ω + Σ α_i ε²_{t-i} + Σ β_j σ²_{t-j}
```

**EGARCH(p,q)** - Nelson (1991)
```
log(σ²_t) = ω + Σ α_i g(z_{t-i}) + Σ β_j log(σ²_{t-j})
```
Captures leverage effect through asymmetric g(z) function.

**GJR-GARCH(p,q)** - Glosten, Jagannathan, Runkle (1993)
```
σ²_t = ω + Σ (α_i + γ_i I_{t-i}) ε²_{t-i} + Σ β_j σ²_{t-j}
```
Asymmetric response: negative shocks have impact (α + γ).

**HARCH** - Müller et al. (1997)
```
σ²_t = ω + Σ α_i (Σ ε_{t-j})² / lag_i
```
Multi-horizon aggregation for heterogeneous market participants.

All models estimated via maximum likelihood with parameter constraints and stationarity checks.

### Rough Volatility Models

**Fractional Brownian Motion (fBm)**
- Hurst parameter H < 0.5 for rough paths
- Simulation via Cholesky decomposition and Davies-Harte FFT

**rBergomi** - Bayer, Friz, Gatheral (2016)
```
V_t = ξ_0 exp(η W^H_t - 0.5 η² t^{2H})
```
Rough volatility with mean reversion and leverage effect.

## Forecasting and Evaluation

### Rolling-Window Forecasts

- Fixed or expanding estimation windows
- 1-step and multi-step ahead forecasts
- Out-of-sample evaluation against realized volatility proxies

### Evaluation Metrics

- MSE, RMSE, MAE, QLIKE
- R-squared (Mincer-Zarnowitz regression)
- Diebold-Mariano test for pairwise model comparison

**Critical insight**: Because volatility forecasts are noisy and R² values are typically low (0.2-0.4), formal statistical tests like Diebold-Mariano are essential to assess whether forecast differences are statistically significant.

## Risk Metrics and Backtesting

### Value-at-Risk (VaR)

Parametric VaR at 95% and 99% confidence levels:
```
VaR_α = -σ_t z_α
```

### Expected Shortfall (ES)

Expected loss beyond VaR:
```
ES_α = σ_t φ(z_α) / α
```

### Backtesting Procedures

- **Kupiec test**: Unconditional coverage (violation rate = α?)
- **Christoffersen test**: Independence of violations
- **McNeil-Frey test**: ES consistency
- **Basel traffic light test**: Regulatory framework

## Key Findings

### Model Performance

- GARCH(1,1) achieves typical persistence (α + β ≈ 0.96-0.98)
- Asymmetric models (EGARCH, GJR-GARCH) capture leverage effects
- Diebold-Mariano tests show marginal differences between models
- R² ≈ 0.25-0.35 (typical for volatility forecasting)

### VaR Backtesting

- 95% VaR: Violation rates 5-6% (within acceptable range)
- Kupiec test: Cannot reject correct specification
- Some evidence of violation clustering (Christoffersen test)

### Rough Volatility Insights

- Autocorrelation: Power-law decay vs GARCH exponential decay
- Impulse response: Fast initial reaction vs slow GARCH decay
- Daily data partially masks roughness (H ≈ 0.3-0.4 vs H ≈ 0.1 at tick level)

## Limitations and Model Risk

These limitations are inherent to volatility modeling rather than artifacts of implementation.

### Realized Volatility Proxy Noise

Squared returns have signal-to-noise ratio ≈ 1:1 at daily frequency. This limits attainable R² in volatility forecasting (typically 0.2-0.4) and necessitates formal statistical tests (Diebold-Mariano) to distinguish genuine forecast improvements from noise.

### Frequency Limitations for Rough Volatility

Rough volatility effects are strongest at intraday scales (H ≈ 0.1). Daily data partially masks this roughness (H ≈ 0.3-0.4 at daily frequency). The rough volatility simulations here serve as structural benchmarks rather than direct calibration targets. For derivatives pricing at maturities < 1 month, intraday data is required.

### Regime Sensitivity and Parameter Instability

GARCH parameters are not stable across market regimes. Persistence increases from ~0.95 (calm) to ~0.99 (crisis), and leverage effects strengthen in high-volatility periods. Operational risk management requires rolling estimation windows, stress testing with crisis-period data, and model risk capital buffers.

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- numpy, pandas, scipy
- matplotlib, seaborn
- statsmodels
- yfinance
- jupyter

## Usage

### Full Pipeline

```bash
python main.py
```

Executes complete analysis:
1. Data loading and stylized facts
2. Model estimation (ARCH, GARCH, EGARCH, GJR, HARCH)
3. Rolling forecasts with Diebold-Mariano tests
4. VaR/ES backtesting
5. Rough volatility benchmarking
6. Visualization and reporting

**Output**:
- `reports/figures/` - All plots
- `reports/tables/` - Model comparisons and DM test results
- `data/processed/` - Cleaned return data

### Interactive Analysis

```bash
jupyter notebook
```

Navigate to `notebooks/` and run sequentially:
1. `01_stylized_facts.ipynb` - Empirical return properties
2. `02_arch_garch_models.ipynb` - Classical GARCH estimation
3. `03_asymmetric_models.ipynb` - EGARCH and GJR-GARCH
4. `04_harch_models.ipynb` - Multi-horizon models
5. `05_forecasting_pipeline.ipynb` - Out-of-sample forecasts
6. `06_var_es_backtesting.ipynb` - Risk metrics
7. `07_rough_volatility_benchmark.ipynb` - Rough vol comparison

### Custom Analysis

```python
from src.data_loader import DataLoader
from src.models import GARCHModel
from src.risk import VaRCalculator

# Load data
loader = DataLoader()
returns, _ = loader.prepare_dataset(ticker="^GSPC")

# Fit model
model = GARCHModel(p=1, q=1)
model.fit(returns.values)
print(model.summary())

# Calculate VaR
var_calc = VaRCalculator(confidence_level=0.95)
var = var_calc.var_from_model(model, returns.values, horizon=1)
```

## Project Structure

```
econometric-volatility-models/
├── data/
│   ├── raw/              # Downloaded price data
│   └── processed/        # Cleaned returns
├── src/
│   ├── models/           # ARCH, GARCH, EGARCH, GJR, HARCH
│   ├── forecasting/      # Rolling forecasts and metrics
│   ├── risk/             # VaR, ES, backtesting
│   ├── rough_vol/        # fBm and rBergomi
│   ├── data_loader.py    # Data fetching
│   ├── returns.py        # Stylized facts
│   └── diagnostics.py    # Model diagnostics
├── notebooks/            # Interactive analysis
├── reports/
│   ├── figures/          # Generated plots
│   ├── tables/           # Model comparisons
│   └── TECHNICAL_REPORT.md
├── main.py               # Full pipeline
├── test_installation.py  # Verification
└── requirements.txt
```

## Testing

```bash
python test_installation.py
```

Verifies:
- All packages installed
- All modules import correctly
- Basic functionality works
- Data download succeeds

## References

1. Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

2. Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.

3. Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. *The Journal of Finance*, 48(5), 1779-1801.

4. Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. *Econometrica*, 59(2), 347-370.

5. Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

6. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance*, 18(6), 933-949.

7. Bayer, C., Friz, P., & Gatheral, J. (2016). Pricing under rough volatility. *Quantitative Finance*, 16(6), 887-904.

8. Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*, 39(4), 841-862.

9. Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73-84.

## License

MIT

## Author

Quantitative Researcher  
2026
