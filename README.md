# Quantitative Finance Portfolio

A comprehensive collection of 19 quantitative finance projects demonstrating expertise across market data engineering, statistical arbitrage, algorithmic trading, C++ systems development, and advanced mathematical modeling.

## Portfolio Overview

This repository showcases end-to-end quantitative finance capabilities. Each project emphasizes:

- Statistical Rigor: Proper hypothesis testing, walk-forward validation, honest failure analysis
- Production Quality: Modular design, comprehensive testing, performance optimization
- Mathematical Depth: Advanced models with proper theoretical foundations
- Engineering Excellence: Clean code, documentation, reproducibility

---

## Repository Structure

### 01. Universal Foundations (3 Projects)
Core quantitative research infrastructure and methodologies

### 02. Quant Research (4 Projects)
Advanced statistical models and econometric analysis

### 03. Quant Trading (6 Projects)
High-frequency trading and execution strategies

### 04. Quant Developer (6 Projects)
Low-latency C++ systems for production trading

---

## Project Catalog

### 01_universal_foundations

#### 01. Real Market Data Engineering Pipeline
Production-grade market data infrastructure for quantitative trading

- Robust data cleaning with Hampel filter outlier detection
- Comprehensive stationarity testing (ADF, KPSS, Phillips-Perron, Ljung-Box, ARCH LM)
- HP filter drift removal and decomposition
- 40-80 engineered features including advanced volatility estimators
- Realistic backtesting with transaction costs and slippage
- Walk-forward validation framework

Tech Stack: Python, Pandas, NumPy, Statsmodels, SciPy

#### 02. Cross-Asset Statistical Arbitrage
Volatility-scaled target engineering for cross-sectional alpha research

- Volatility-scaled forward returns (30-40x IC improvement)
- Risk neutralization revealing hidden alpha (82% IC increase)
- Horizon-matched execution testing
- Comprehensive IC stability analysis
- XGBoost model achieving 90.8% prediction accuracy
- Honest assessment: IC 0.0947, Sharpe 0.45 (frozen signal)

Tech Stack: Python, XGBoost, Scikit-learn

#### 03. Kalman Filter & HMM Regime Switching
State-space models with regime detection for adaptive trading

- Kalman filtering for latent state estimation (Linear, EKF, UKF)
- Hidden Markov Models for regime identification
- Regime-aware position sizing and risk management
- Time-series momentum (TSMOM) integration
- Monte Carlo validation (500+ paths)
- Numerical stability (Joseph form, PSD enforcement)

Tech Stack: Python, NumPy, SciPy

---

### 02_quant_research

#### 01. Factor Modeling via PCA & Eigenportfolios
Statistical and economic factor models for equity risk premia

- Advanced PCA methodology (standardized covariance, raw returns)
- Classical factors (Momentum, Value, Size, Quality, Low-Vol)
- Three-lens evaluation (statistical, explanatory, economic)
- Regime-dependent factor analysis
- Fama-MacBeth cross-sectional regressions
- Quality factor: Sharpe 4.05, 90.5% annual return

Tech Stack: Python, Pandas, NumPy, Statsmodels

#### 02. Cointegration Statistical Arbitrage
Pairs trading with Kalman filtering and regime detection

- Johansen cointegration testing for pair selection
- Kalman Filter for time-varying hedge ratios
- Ornstein-Uhlenbeck spread modeling
- Regime instability analysis (94% cointegration failure rate)
- Quality gates rejecting weak signals
- Regime-filtered performance: 4x Sharpe improvement

Tech Stack: Python, Statsmodels, SciPy

#### 03. Econometric Volatility Modeling
GARCH-family models with rough volatility benchmarking

- Classical models (ARCH, GARCH, EGARCH, GJR-GARCH, HARCH)
- Maximum likelihood estimation with diagnostics
- Rolling out-of-sample forecasts with Diebold-Mariano tests
- VaR/ES calculation and backtesting (Kupiec, Christoffersen)
- Rough volatility (rBergomi) structural benchmarking
- Typical R² 0.25-0.35 (realistic for volatility forecasting)

Tech Stack: Python, Statsmodels, SciPy

#### 04. Rough Volatility Modeling
Fractional Brownian motion for derivatives pricing

- rBergomi model with Hurst parameter H < 0.5
- Hybrid scheme (Cholesky + FFT convolution, 100x speedup)
- Monte Carlo pricing with variance reduction
- Calibration to implied volatility surfaces
- Comparison to Heston and SABR
- Short-maturity skew: T^(H-1/2) decay

Tech Stack: Python, NumPy, SciPy

---

### 03_quant_trading

#### 01. HFT Microstructure Alpha Signals
Tick-by-tick price prediction using full-depth limit order book

- Order Flow Imbalance (OFI) and microstructure features (59 total)
- XGBoost model with walk-forward validation
- Event-driven backtesting with realistic execution
- Three execution strategies (Market, Limit, EV-based)
- Alpha decay analysis (1-20 tick horizons)
- Regime analysis (volatility, liquidity, time-of-day)

Tech Stack: Python, XGBoost, NumPy, Pandas

#### 02. Avellaneda-Stoikov Market Making
Optimal market-making with HJB equation and Monte Carlo rigor

- Closed-form optimal quotes from HJB equation
- Self-financing PnL decomposition (spread/inventory/adverse selection)
- Multi-agent competition with proper order allocation
- Monte Carlo validation (500+ paths)
- Regime analysis showing failure modes
- Sharpe 0.5-2.0 (realistic, not overfitted)

Tech Stack: Python, NumPy, SciPy

#### 03. Market Impact Models
Comparative analysis of Kyle, Obizhaeva-Wang, and Bouchaud

- Kyle's Lambda (linear adverse selection)
- Obizhaeva-Wang (permanent/transient decomposition)
- Bouchaud Propagator (power-law memory kernel)
- Statistical rigor with confidence intervals
- Cross-regime validation
- Explicit failure mode analysis

Tech Stack: Python, NumPy, SciPy

#### 04. Statistical Arbitrage Execution
Cointegration-based pairs trading with OU modeling

- Engle-Granger cointegration testing
- Ornstein-Uhlenbeck parameter estimation
- Kalman filtering for dynamic hedge ratios
- Regime detection via HMM
- Quality gates rejecting weak signals
- Portfolio-level diversification

Tech Stack: Python, Statsmodels, SciPy

#### 05. Hawkes Process Market Making
Self-exciting point processes for order flow modeling

- Multidimensional Hawkes process (6 event types)
- Ogata's thinning algorithm for simulation
- Maximum likelihood estimation
- Time-rescaling diagnostics (KS test)
- Inventory-aware market making
- Reservation price framework

Tech Stack: Python, NumPy, SciPy

#### 06. Optimal Execution with Reinforcement Learning
Combining Almgren-Chriss with offline RL

- Almgren-Chriss closed-form solutions
- Offline RL (BCQ, TD3+BC) for stochastic liquidity
- Gymnasium environment with realistic microstructure
- Stress testing (liquidity collapse, volatility spikes)
- RL outperforms under stochastic liquidity
- AC optimal under stated assumptions

Tech Stack: Python, PyTorch, Gymnasium

---

### 04_quant_developer

#### 01. C++ Derivatives Pricing Engine
Heston stochastic volatility with Monte Carlo and PDE

- Heston model with full stochastic volatility dynamics
- Monte Carlo (Euler, QE discretization, antithetic variates)
- PDE solver (explicit finite difference)
- Calibration to implied volatility surfaces
- Multithreading support
- Sobol quasi-random number generation

Tech Stack: C++17, CMake

#### 02. C++ Limit Order Book Engine
Single-threaded matching engine with deterministic execution

- Price-time priority (FIFO at each price level)
- Pre-allocated memory pools (allocation-free hot path)
- Intrusive lists for cache locality
- Fixed-point prices (no floating-point non-determinism)
- Typical latency: 200-400ns per order
- p99: 500-800ns, p99.9: 1-2μs

Tech Stack: C++17, CMake

#### 03. C++ Market Data Gateway
Low-latency feed handler with multi-threaded pipeline

- Lock-free SPSC queues for inter-thread communication
- FIX and binary protocol parsing
- Zero-copy pointer-based parsing
- Cache-line aligned data structures
- Binary parser: ~70ns per message
- FIX parser: ~300ns per message

Tech Stack: C++17, CMake, epoll

#### 04. C++ Smart Order Router
Multi-venue execution with cost-based routing

- TWAP, VWAP, POV, aggressive liquidity-seeking strategies
- Cost model (spread + impact + latency + queue risk)
- Stochastic queue model for fill probability
- Adaptive VWAP with regime detection
- Almgren-Chriss square-root impact model
- Implementation shortfall tracking

Tech Stack: C++17, CMake

#### 05. C++ Portfolio Risk Engine
Monte Carlo risk engine with multithreading

- VaR/CVaR at 95% and 99% confidence
- Analytical Greeks (Delta, Gamma, Vega)
- Stress testing (volatility and correlation shocks)
- Multi-asset correlated GBM
- Custom thread pool: 7.14x speedup on 8 cores
- Throughput: 238K paths/sec single-threaded, 1.9M+ with 8 threads

Tech Stack: C++17, CMake

#### 06. Event-Driven Market Simulator
Continuous limit order book with microsecond resolution

- Event-driven architecture with deterministic replay
- Price-time priority matching engine
- Configurable latency models (fixed, uniform, normal, queueing)
- Throughput: 600K-800K orders/sec
- Deterministic (same input → same output)
- Logical clock (no wall-clock dependency)

Tech Stack: C++17, CMake

---

## Technical Skills Demonstrated

### Programming Languages
- Python: NumPy, Pandas, SciPy, Statsmodels, Scikit-learn, XGBoost, PyTorch
- C++17: Modern C++, STL, templates, multithreading, SIMD
- Build Systems: CMake, Make

### Quantitative Methods
- Time series analysis (ARIMA, GARCH, cointegration)
- State-space models (Kalman filtering, HMM)
- Monte Carlo simulation and variance reduction
- PDE solvers (finite difference methods)
- Stochastic calculus (Brownian motion, Itô calculus)
- Optimization (MLE, gradient descent, differential evolution)

### Machine Learning
- Supervised learning (XGBoost, Random Forest, Logistic Regression)
- Reinforcement learning (BCQ, TD3+BC, offline RL)
- Feature engineering and selection
- Walk-forward validation

### Systems Engineering
- Low-latency C++ (sub-microsecond)
- Multithreading and lock-free data structures
- Memory management (custom allocators, object pools)
- Cache optimization and SIMD
- Event-driven architectures

### Risk Management
- VaR/CVaR calculation and backtesting
- Greeks computation (Delta, Gamma, Vega)
- Stress testing and scenario analysis
- Transaction cost modeling
- Market impact estimation

---

## Key Achievements

### Research Discipline
- Honest Failure Analysis: Transparent reporting of negative results
- No Overfitting: Fixed parameters, walk-forward validation
- Statistical Rigor: Proper hypothesis testing, confidence intervals
- Reproducibility: Seeded randomness, version control, documentation

### Production Quality
- Performance: Sub-microsecond latency, millions of messages/sec
- Numerical Stability: Joseph form, PSD enforcement, gain clipping
- Memory Management: Pre-allocated pools, cache-friendly layouts
- Testing: Comprehensive unit tests, benchmarks, diagnostics

### Mathematical Sophistication
- Advanced Models: Rough volatility, Hawkes processes, stochastic control
- Proper Theory: Rigorous derivations, assumptions documented
- Model Comparison: Formal statistical tests (Diebold-Mariano, KS)
- Calibration: MLE, regularization, multi-start optimization

---

## Documentation

This repository includes technical documentation in the `00_README_OVERVIEW/` folder:

- TECHNICAL_SUMMARY.md - Overview of technical stack, methodologies, and performance metrics

Each project contains detailed documentation in its own folder.

---

## Getting Started

### Prerequisites
- Python 3.8+ with scientific stack (NumPy, Pandas, SciPy, Statsmodels)
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15+

### Quick Start
Each project contains:
- README.md - Project overview and quick start
- requirements.txt or CMakeLists.txt - Dependencies
- main.py or main.cpp - Entry point
- docs/ - Detailed documentation

### Example: Run Market Data Pipeline
```bash
cd 01_universal_foundations/01_real_market_data_engineering
pip install -r requirements.txt
python src/pipeline.py
```

### Example: Build C++ LOB Engine
```bash
cd 04_quant_developer/02_cpp_limit_order_book_engine
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./lob_engine
```

---

## Performance Highlights

| Project | Key Metric | Value |
|---------|-----------|-------|
| Market Data Pipeline | Features Engineered | 40-80 |
| Cross-Asset Stat Arb | XGBoost Accuracy | 90.8% |
| Kalman Filter HMM | Monte Carlo Paths | 500+ |
| Factor Modeling PCA | Quality Factor Sharpe | 4.05 |
| Cointegration Pairs | Regime Filter Rate | 99% |
| GARCH Volatility | Forecast R² | 0.25-0.35 |
| HFT Microstructure | Prediction Accuracy | 90.8% |
| Avellaneda-Stoikov | Monte Carlo Paths | 500+ |
| Market Impact | Regime Validation | 3 regimes |
| C++ LOB Engine | Latency (p50) | 250-350ns |
| C++ Market Data | Throughput | 10-15M msg/sec |
| C++ Risk Engine | Speedup (8 cores) | 7.14x |
| Event Simulator | Throughput | 600K-800K orders/sec |
| Rough Volatility | Complexity Reduction | 100x |
| Hawkes Process | Event Types | 6 |
| Optimal Execution | Stress Scenarios | 4 |

---

## Applications

This portfolio demonstrates capabilities across:
- Quantitative trading and systematic strategies
- High-frequency trading and market microstructure
- Market making and liquidity provision
- Quantitative research and factor modeling
- Technology and infrastructure development

---

## License

This repository is for educational and portfolio demonstration purposes.

Individual projects may have specific licenses (typically MIT). See each project's LICENSE file.

Disclaimer: This is research and educational code. Not investment advice. Not production-ready for live trading without significant additional work (infrastructure, risk management, regulatory compliance).

---

## Quick Links

- [Technical Summary](00_README_OVERVIEW/TECHNICAL_SUMMARY.md) - Technical stack and methodologies

---

Last Updated: February 2026  
Status: Production-Ready Portfolio
