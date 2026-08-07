# Kalman Filter and HMM Regime Switching for Financial Time Series

A quantitative research framework for state-space modeling, regime detection, and regime-aware trading signals. The project combines Kalman filtering for latent trend estimation with a Gaussian Hidden Markov Model (HMM) for market-state inference, then routes the outputs through risk controls, volatility targeting, and a transaction-cost-aware backtest.

The emphasis is methodological correctness: causal signal generation, volatility-ordered regimes, numerical stability, reproducible real-data runs, and honest performance reporting.

## Overview

Financial markets exhibit time-varying behavior that static models miss. This framework addresses two problems:

1. **Latent state estimation:** Kalman filters estimate unobservable trend or state variables from noisy returns.
2. **Regime detection:** HMMs estimate discrete market states such as low-volatility, medium-volatility, and crisis/high-volatility regimes.

The active strategy is intentionally conservative. It is a research and interview-grade framework, not a claim of discovered alpha or a live-trading system.

## Key Features

- **State-space models:** Local level, local linear trend, dynamic regression, EKF, and UKF modules.
- **Regime detection:** Gaussian HMM with log-domain EM, covariance floors, volatility-ordered labels, and filtered probabilities for trading.
- **Causal signals:** Trading signals use filtered HMM probabilities and filtered Kalman states; smoothed probabilities are reserved for offline diagnostics.
- **Risk controls:** Regime-conditioned position sizing, volatility targeting, leverage caps, and transaction costs.
- **Alpha sources:** Kalman trend following plus medium-term time-series momentum (63, 126, 252 trading-day lookbacks).
- **Reproducibility:** The production run loads the latest cached real Yahoo Finance snapshot before attempting a live download.
- **Production hygiene:** Validation, structured logging, explicit failures, headless-safe plot generation, and unit tests.

## Installation

```bash
pip install -r requirements.txt
```

The code has been validated in this workspace with Python 3.11, pandas 2.0.3, scikit-learn 1.3.0, and NumPy 1.26.4. The pinned requirements file records the original research environment.

## Quick Start

```bash
python run_production.py --config configs/default_config.yaml
```

The production script:

1. Loads cached or live real market data for SPY, QQQ, and TLT.
2. Fits a Kalman filter to the primary asset return stream.
3. Fits a regularized HMM and orders regimes by volatility.
4. Generates filtered-probability regime-aware signals.
5. Runs a transaction-cost-aware backtest.
6. Saves figures and CSV metrics.

Outputs:

- `figures/` - seven PNG visualizations
- `results/` - performance, regime, and comparison CSVs
- `production_run.log` - execution audit trail

## Programmatic Usage

```python
from src.data_loader import load_sample_data
from src.kalman_filter import KalmanFilter
from src.state_space_models import LocalLevelModel
from src.hmm_regimes import GaussianHMM
from src.signals import create_regime_aware_strategy
from src.backtest import Backtest

data = load_sample_data(tickers=["SPY", "QQQ", "TLT"], prefer_cache=True)
returns = data["returns"].iloc[:, 0].values

model = LocalLevelModel(
    observation_variance=1.0,
    state_variance=0.1,
    initial_state_variance=10.0,
)
kf = KalmanFilter(model)
kf.filter(returns)

hmm = GaussianHMM(n_regimes=3, n_iter=50, random_state=42)
hmm.fit(returns)

signals = create_regime_aware_strategy(returns, kf, hmm, vol_target=0.15)
bt = Backtest(signals, returns, transaction_cost=0.0005)
results = bt.run()

print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Project Structure

```text
src/
  data_loader.py             Real-data loading, cache-first workflow
  preprocessing.py           Missing data, outliers, feature prep
  state_space_models.py      Model specifications
  kalman_filter.py           Linear Kalman filter and RTS smoother
  extended_kalman_filter.py  EKF utilities
  unscented_kalman_filter.py UKF utilities
  hmm_regimes.py             Regularized Gaussian HMM
  regime_features.py         Regime feature engineering
  signals.py                 Baseline regime-aware signals
  signals_enhanced.py        Risk sizing, volatility targeting, TSMOM
  backtest.py                Single-asset backtesting engine
  portfolio.py               Multi-asset portfolio construction
  evaluation.py              Diagnostics and performance metrics
  visualization.py           Plotting utilities
configs/default_config.yaml  Default run configuration
docs/                        Assumptions, architecture, and limitations
tests/test_kalman_hmm.py     Unit and integration tests
run_production.py            Production-style pipeline
run_pipeline.py              Simpler research pipeline
```

## Current Results

Latest reproduced production run:

- **Data:** SPY primary return stream from cached SPY/QQQ/TLT Yahoo Finance data
- **Return window:** 2021-04-13 to 2026-04-10
- **Transaction cost:** 5 bps per unit turnover
- **Signal mode:** enhanced regime-aware strategy with TSMOM enabled and regime gating disabled

| Strategy | Sharpe | Annual Return | Volatility | Max Drawdown | Avg Turnover |
|---|---:|---:|---:|---:|---:|
| Buy & Hold | 0.67 | 10.42% | 17.05% | 26.22% | 0.08% |
| Kalman Trend | -0.71 | -12.68% | 17.07% | 53.59% | 48.84% |
| Regime-Aware | -0.35 | -3.18% | 8.18% | 26.03% | 27.69% |

Regime diagnostics from posterior-dominant HMM states:

| Regime | Interpretation | Frequency | Daily Mean | Daily Volatility |
|---|---|---:|---:|---:|
| 0 | Low volatility / constructive | 65.42% | 0.0914% | 0.7104% |
| 1 | Medium volatility | 33.63% | 0.0045% | 1.3079% |
| 2 | Crisis / high volatility | 0.96% | -1.6658% | 4.7755% |

Interpretation: buy-and-hold still wins on absolute return over this window, but the active framework materially reduces realized volatility versus simple Kalman trend and avoids the catastrophic drawdown of the naive trend-only strategy. It does not beat passive SPY in this sample and should not be presented as alpha discovery.

## Technical Notes

- HMM regimes are relabeled by ascending emission volatility after fit, so strategy logic is not tied to arbitrary EM labels.
- Trading functions default to filtered HMM probabilities. Smoothed probabilities and posterior-dominant labels are used for diagnostics and plots.
- Gaussian HMM covariance floors prevent degenerate regimes that memorize individual outliers.
- Backtests lag positions by one period to avoid lookahead bias.
- Figures save cleanly in headless environments without noisy `plt.show()` warnings.

## Documentation

- [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) - pipeline and module design
- [MODEL_ASSUMPTIONS.md](docs/MODEL_ASSUMPTIONS.md) - mathematical assumptions and where they break
- [RESULTS_AND_LIMITATIONS.md](docs/RESULTS_AND_LIMITATIONS.md) - performance interpretation and deployment requirements
- [LIMITATIONS_AND_FAILURE_MODES.md](docs/LIMITATIONS_AND_FAILURE_MODES.md) - known edge cases and failure modes
- [ATS_SCREENING_PACK.md](docs/ATS_SCREENING_PACK.md) - resume bullets, keywords, interview defense, and upgrade path

## Disclaimer

This is a research prototype for education, quant interview preparation, and model-engineering demonstration. It is not investment advice, not a live-trading system, not regulatory-compliant, and not validated for deployment.

Live use would require professional data infrastructure, execution systems, real-time risk controls, monitoring, compliance review, and substantial out-of-sample validation.

## License

MIT License. See `LICENSE`.

## References

- Durbin, J., and Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*. Oxford University Press.
- Hamilton, J. D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." *Econometrica*, 57(2), 357-384.
- Julier, S. J., and Uhlmann, J. K. (2004). "Unscented Filtering and Nonlinear Estimation." *Proceedings of the IEEE*, 92(3), 401-422.
- Moskowitz, T. J., Ooi, Y. H., and Pedersen, L. H. (2012). "Time Series Momentum." *Journal of Financial Economics*, 104(2), 228-250.

**Status:** Research prototype with production-style safeguards. Not a live-trading system.
