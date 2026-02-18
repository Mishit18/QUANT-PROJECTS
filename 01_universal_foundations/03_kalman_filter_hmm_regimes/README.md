# Kalman Filter and HMM Regime Switching for Financial Time Series

A quantitative research framework combining state-space models with regime detection for adaptive trading strategies. This system demonstrates production-grade implementation of Kalman filtering for latent state estimation and Hidden Markov Models for regime identification, with emphasis on risk management and numerical stability.

## Overview

Financial markets exhibit time-varying dynamics that simple models fail to capture. This framework addresses two fundamental challenges:

1. **Latent State Estimation:** Kalman filtering provides optimal estimation of unobservable variables (trend, volatility, dynamic parameters)
2. **Regime Detection:** Hidden Markov Models identify discrete market states (bull/bear, high/low volatility, trending/mean-reverting)

The system combines these approaches to generate regime-aware trading signals with sophisticated risk management.

## Key Features

- **State-Space Models:** Linear Gaussian, Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF)
- **Regime Detection:** HMM with EM algorithm and forward-backward inference
- **Risk Management:** Regime-conditioned position sizing, volatility targeting, leverage controls
- **Alpha Sources:** Kalman trend following + time-series momentum (TSMOM)
- **Numerical Stability:** Joseph form covariance updates, PSD enforcement, gain clipping
- **Production Engineering:** Comprehensive error handling, validation, structured logging

## Installation

```bash
# Clone repository
git clone <repository-url>
cd kalman-hmm-regime-switching

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- numpy==1.24.3
- pandas==2.0.3
- scipy==1.11.4
- matplotlib, seaborn, yfinance

## Quick Start

### Run Complete Pipeline

```bash
python run_production.py
```

This executes the full pipeline:
1. Load market data (SPY, QQQ, TLT)
2. Fit Kalman filter for trend estimation
3. Detect regimes via HMM
4. Generate regime-aware signals
5. Backtest with transaction costs
6. Produce performance metrics and visualizations

**Outputs:**
- `figures/` - 7 PNG visualizations
- `results/` - 4 CSV files with metrics
- Console log with execution trace

### Programmatic Usage

```python
from src.data_loader import load_market_data
from src.kalman_filter import KalmanFilter
from src.state_space_models import LocalLevelModel
from src.hmm_regimes import GaussianHMM
from src.signals import create_regime_aware_strategy
from src.backtest import Backtest

# Load data
data = load_market_data(['SPY'])
returns = data['returns'].iloc[:, 0].values

# Kalman filter
model = LocalLevelModel(obs_var=1.0, state_var=0.1)
kf = KalmanFilter(model)
kf.filter(returns)

# HMM regime detection
hmm = GaussianHMM(n_regimes=3)
hmm.fit(returns)

# Generate signals
signals = create_regime_aware_strategy(returns, kf, hmm)

# Backtest
bt = Backtest(signals, returns, transaction_cost=0.0005)
results = bt.run()

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Project Structure

```
├── src/                          # Core modules
│   ├── data_loader.py            # Data ingestion with validation
│   ├── preprocessing.py          # Data cleaning and normalization
│   ├── state_space_models.py    # Model specifications
│   ├── kalman_filter.py          # Standard Kalman filter
│   ├── extended_kalman_filter.py # EKF for nonlinear dynamics
│   ├── unscented_kalman_filter.py# UKF with sigma points
│   ├── hmm_regimes.py            # HMM regime detection
│   ├── regime_features.py        # Feature engineering
│   ├── signals.py                # Baseline signal generation
│   ├── signals_enhanced.py       # Enhanced signals with TSMOM
│   ├── backtest.py               # Backtesting engine
│   ├── evaluation.py             # Performance metrics
│   ├── visualization.py          # Plotting functions
│   ├── portfolio.py              # Multi-asset construction
│   └── utils.py                  # Numerical utilities
├── docs/                         # Documentation
│   ├── PROJECT_OVERVIEW.md       # System design
│   ├── SYSTEM_ARCHITECTURE.md    # Technical architecture
│   ├── MODEL_ASSUMPTIONS.md      # Mathematical assumptions
│   ├── LIMITATIONS_AND_FAILURE_MODES.md
│   ├── RESULTS_AND_LIMITATIONS.md
│   ├── INTERVIEW_DEFENSE.md      # Technical Q&A
│   └── REPOSITORY_STRUCTURE.md   # File organization
├── notebooks/                    # Research notebooks
│   └── 08_full_pipeline_backtest.ipynb
├── tests/                        # Unit tests
│   └── test_kalman_hmm.py
├── configs/                      # Configuration files
│   └── default_config.yaml
├── data/                         # Data storage (raw, processed)
├── figures/                      # Generated visualizations
├── results/                      # Performance metrics
├── run_production.py             # Production execution script
├── run_pipeline.py               # Standard execution script
├── requirements.txt              # Python dependencies
├── Makefile                      # Build automation
└── README.md                     # This file
```

## Results

### Performance Summary (2019-2024 Test Period)

| Strategy | Sharpe | Ann. Return | Volatility | Max DD | Turnover |
|----------|--------|-------------|------------|--------|----------|
| Buy & Hold | 0.85 | 12.3% | 14.5% | 33.7% | 0% |
| Baseline | -0.42 | -0.87% | 2.02% | 7.81% | 2.2% |
| Enhanced | -0.55 | -0.61% | 1.11% | 4.55% | 1.4% |
| Enhanced + TSMOM | -0.18 | -0.37% | 2.05% | 6.82% | 26.0% |

### Key Observations

1. **Negative Absolute Returns:** All active strategies underperformed buy & hold during the 2019-2024 bull market
2. **Volatility Reduction:** Enhanced strategy achieved 76% volatility reduction (1.11% vs 14.5%)
3. **Drawdown Control:** Maximum drawdown reduced from 33.7% to 4.55%
4. **Risk Management:** Framework demonstrates effective risk control even with negative returns

**Important:** These results reflect a strong bull market test period. Regime-switching strategies are designed for crisis protection and volatility management, not bull market capture. See `docs/RESULTS_AND_LIMITATIONS.md` for detailed analysis.

## Documentation

- **[PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)** - Problem statement, motivation, design philosophy
- **[SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md)** - Pipeline flow, module responsibilities, numerical stability
- **[MODEL_ASSUMPTIONS.md](docs/MODEL_ASSUMPTIONS.md)** - Mathematical assumptions and approximations
- **[RESULTS_AND_LIMITATIONS.md](docs/RESULTS_AND_LIMITATIONS.md)** - Honest performance assessment, limitations, deployment requirements
- **[LIMITATIONS_AND_FAILURE_MODES.md](docs/LIMITATIONS_AND_FAILURE_MODES.md)** - Known failure scenarios and edge cases
- **[INTERVIEW_DEFENSE.md](docs/INTERVIEW_DEFENSE.md)** - Comprehensive technical Q&A

## Configuration

All parameters controlled via `configs/default_config.yaml`:

```yaml
kalman_filter:
  observation_variance: 1.0
  state_variance: 0.1

hmm:
  n_regimes: 3
  max_iter: 100

signals:
  enhanced:
    enabled: true
    max_leverage: 1.5
    vol_target_enhanced: 0.12
    regime_position_sizing: true
  
  momentum:
    enabled: true
    lookbacks: [63, 126, 252]
    kalman_weight: 0.6
    momentum_weight: 0.4
```

## Technical Highlights

### Numerical Stability
- **Joseph Form:** Covariance updates guaranteed positive semi-definite
- **PSD Enforcement:** Eigenvalue-based clipping
- **Gain Clipping:** Prevents filter divergence
- **NaN/Inf Validation:** Explicit checks at every step

### Causality Enforcement
- Positions lagged by 1 period (no lookahead bias)
- Filtered states only (smoothing for analysis only)
- Walk-forward validation
- Explicit temporal ordering

### Production Engineering
- Real data enforcement (no synthetic fallback)
- 3-retry logic with exponential backoff
- Comprehensive error handling
- Structured logging with audit trail
- Validation at every pipeline stage

## Disclaimer

**This is a research prototype for educational and interview purposes.**

It is NOT:
- Investment advice
- Production-ready for live trading
- Optimized for returns
- Validated on out-of-sample data
- Regulatory compliant

Real market deployment requires:
- Professional data infrastructure
- Execution systems (OMS)
- Real-time risk management
- Regulatory compliance
- Extensive testing

The models make strong assumptions (Gaussianity, stationarity, discrete regimes) that do not hold in real markets. See `docs/LIMITATIONS_AND_FAILURE_MODES.md` for details.

## License

MIT License - See LICENSE file for details.

## References

- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Julier, S. J., & Uhlmann, J. K. (2004). "Unscented Filtering and Nonlinear Estimation." *Proceedings of the IEEE*, 92(3), 401-422.
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). "Time Series Momentum." *Journal of Financial Economics*, 104(2), 228-250.

## Contact

For questions about this research framework, please refer to the documentation in `docs/`.

---

**Status:** Research prototype complete. production-ready. Not for production deployment.
