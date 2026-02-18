# Cointegration-Based Statistical Arbitrage with Regime-Aware Risk Management

## Overview

This project implements a pairs trading strategy that exploits mean-reverting relationships between cointegrated assets. The strategy combines classical econometric techniques (Johansen cointegration, Kalman Filtering) with regime-aware risk management to improve risk-adjusted returns.

**Key Features**:
- Johansen cointegration testing for pair selection
- Kalman Filter for time-varying hedge ratio estimation
- Regime instability analysis and conditional trading
- Comprehensive diagnostics and performance evaluation

## Project Structure

```
cointegration_stat_arb/
├── src/                    # Core implementation
│   ├── data_pipeline.py    # Data acquisition and preprocessing
│   ├── stationarity_tests.py  # ADF, KPSS tests
│   ├── cointegration_tests.py # Johansen, Engle-Granger
│   ├── kalman_filter.py    # Time-varying hedge ratios
│   ├── regime_filters.py   # Regime-aware trading controls
│   ├── kalman_confidence.py # Confidence-based gating
│   ├── signal_generation.py # Entry/exit logic
│   ├── backtest_engine.py  # Event-driven backtesting
│   ├── performance_metrics.py # Sharpe, drawdown, etc.
│   └── utils.py            # Logging and plotting
├── scripts/                # Execution scripts
│   ├── run_pipeline.py     # Main execution
│   ├── show_results.py     # Results display
│   ├── run_conditional_backtest.py # Filtered vs unfiltered
│   ├── analyze_regimes.py  # Regime instability analysis
│   └── audit_kalman.py     # Kalman Filter validation
├── config/                 # Configuration
│   └── strategy.yaml       # Strategy parameters
├── data/                   # Market data
│   ├── raw/                # Downloaded data
│   └── processed/          # Preprocessed data
├── results/                # Output
│   ├── backtests/          # Backtest results
│   ├── diagnostics/        # Diagnostic analysis
│   └── portfolio/          # Portfolio metrics
├── reports/                # Documentation and plots
│   └── figures/            # Generated plots
├── notebooks/              # Jupyter notebooks
│   ├── exploratory_analysis.ipynb
│   └── cointegration_scan.ipynb
├── docs/                   # Documentation
│   ├── methodology.md      # Econometric methodology
│   ├── regime_analysis.md  # Regime instability findings
│   ├── kalman_filter_notes.md # Kalman Filter details
│   ├── assumptions_and_limitations.md
│   ├── risk_management.md  # Risk framework
│   ├── interview_defense.md # Q&A preparation
│   └── reproduction_guide.md # Step-by-step reproduction
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python scripts/run_pipeline.py --mode full
```

**Runtime**: ~30 seconds  
**Output**: Backtest results, plots, and diagnostics

### 3. View Results

```bash
python scripts/show_results.py
```

### 4. Run Conditional Backtest

```bash
python scripts/run_conditional_backtest.py
```

## Methodology

### Pair Selection

We identify cointegrated pairs using the Johansen cointegration test:
- **Test Statistic**: Trace statistic
- **Significance Level**: 5%
- **Universe**: Sector ETFs (XLK, XLV, XLF, XLE, XLI, XLY, XLP, XLU) + QQQ

### Hedge Ratio Estimation

Time-varying hedge ratios are estimated using a Kalman Filter:
- **State Equation**: β_t = β_{t-1} + w_t
- **Observation Equation**: y_t = β_t * x_t + v_t
- **Parameters**: Q = 1e-4, R = 1e-2

### Signal Generation

- **Entry**: |z-score| > 2.0
- **Exit**: |z-score| < 0.5
- **Z-score**: (spread - mean) / std over rolling window

### Regime-Aware Risk Management

Trades are suppressed when:
1. Cointegration test fails (Johansen)
2. Spread is non-stationary (ADF)
3. Half-life outside 5-60 days
4. Variance > 3× historical

## Key Findings

### Baseline Performance (2015-2023)

```
Cointegrated Pairs: 2 (XLK-XLV, QQQ-XLV)
Total Trades: 10
Win Rate: 70%
Total Return: 0.007%
Sharpe Ratio: 0.02
Max Drawdown: -0.14%
```

### Regime Instability

- **Cointegration Failure Rate**: 94%
- **Stationarity Failure Rate**: 89%
- **Half-Life Instability**: 87%

**Interpretation**: Favorable trading conditions are rare. The low baseline return reflects this reality.

### Regime-Filtered Performance

**XLK-XLV**:
- Baseline: Sharpe = 0.05, Return = -9.91%
- Filtered: Sharpe = 0.21, Return = +2.04% (4× Sharpe improvement)

**QQQ-XLV**:
- Baseline: Sharpe = 0.23, Return = +26.78%
- Filtered: Sharpe = 0.36, Return = +2.73% (57% Sharpe improvement)

**Trade Suppression**: 99% (regime filters correctly identify when NOT to trade)

## Documentation

- **[Methodology](docs/methodology.md)**: Econometric theory and implementation
- **[Regime Analysis](docs/regime_analysis.md)**: Instability findings and implications
- **[Kalman Filter Notes](docs/kalman_filter_notes.md)**: State-space model details
- **[Assumptions and Limitations](docs/assumptions_and_limitations.md)**: Known constraints
- **[Risk Management](docs/risk_management.md)**: Multi-layered risk framework
- **[Interview Defense](docs/interview_defense.md)**: Q&A preparation
- **[Reproduction Guide](docs/reproduction_guide.md)**: Step-by-step instructions

## Requirements

- Python 3.9+
- numpy >= 1.26.4
- pandas >= 2.0.0
- scipy >= 1.11.0
- statsmodels >= 0.14.0
- matplotlib >= 3.7.0
- yfinance >= 0.2.28

See `requirements.txt` for complete list.

## Limitations

1. **Small Universe**: Only 9 ETFs tested
2. **Regime Detection Lag**: 252-day rolling windows
3. **Low Capital Utilization**: Strategy idle 99% of time
4. **Sample Period**: 2015-2023 includes extreme events
5. **Daily Rebalancing**: Misses intraday opportunities

See `docs/assumptions_and_limitations.md` for details.

## Future Enhancements

1. Expand universe to 50-100 pairs
2. Implement real-time regime monitoring
3. Add intraday execution capabilities
4. Optimize transaction costs
5. Integrate into multi-strategy portfolio

## License

This project is for research and educational purposes.

## Contact

For questions or collaboration, please open an issue or contact the author.

---

**Note**: This is a research prototype. Production deployment requires infrastructure enhancements for real-time execution and operational risk management.
