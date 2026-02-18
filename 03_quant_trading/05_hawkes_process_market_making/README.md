# Hawkes Process-Driven Limit Order Book Simulation

A research-grade simulation and modeling framework for limit order book dynamics using multidimensional Hawkes processes. This project implements order flow modeling, parameter estimation, statistical validation, and a market-making agent with inventory control.

**Status:** Research prototype. Not intended for production trading.

---

## Problem Definition

Traditional limit order book models assume Poisson arrivals for order events, which fail to capture empirical properties observed in real markets:

- **Self-excitation**: Orders trigger subsequent orders of the same type
- **Cross-excitation**: Events of one type influence the arrival rate of other types
- **Clustering**: Events arrive in bursts rather than uniformly over time

This project addresses these limitations by modeling order flow as a multidimensional Hawkes process, which naturally captures self-exciting and clustering behavior through its intensity structure.

---

## Modeling Approach

### Order Flow Model

We model six order event types using a multidimensional Hawkes process:
1. Limit buy orders
2. Limit sell orders
3. Market buy orders
4. Market sell orders
5. Cancel buy orders
6. Cancel sell orders

### Limit Order Book

An event-driven limit order book simulator processes the Hawkes-generated event stream:
- Price-time priority matching
- Order addition, execution, and cancellation
- Real-time book state tracking

### Market-Making Agent

A market-making agent operates on the simulated book:
- Inventory-aware quoting using reservation price framework
- Dynamic spread adjustment based on inventory position and adverse selection risk
- PnL tracking and performance evaluation

---

## Mathematical Formulation

### Hawkes Process Intensity

For event type *i*, the intensity at time *t* is:

```
λᵢ(t) = μᵢ + Σⱼ ∫₀ᵗ φᵢⱼ(t - s) dNⱼ(s)
```

Where:
- `μᵢ > 0`: Baseline intensity for type *i*
- `φᵢⱼ(t)`: Excitation kernel from type *j* to type *i*
- `Nⱼ(s)`: Counting process for type *j*

### Exponential Kernel

```
φᵢⱼ(t) = αᵢⱼ βᵢⱼ exp(-βᵢⱼ t)  for t > 0
```

Parameters:
- `αᵢⱼ ≥ 0`: Excitation amplitude (branching ratio)
- `βᵢⱼ > 0`: Decay rate

### Stationarity Condition

The process is stationary if the spectral radius of the branching matrix `A = [αᵢⱼ]` satisfies:

```
ρ(A) < 1
```

### Reservation Price

The market-making agent computes a reservation price incorporating inventory penalty:

```
r(t) = m(t) - γ q(t) σ²
```

Where:
- `m(t)`: Mid price
- `q(t)`: Current inventory
- `γ`: Risk aversion parameter
- `σ²`: Price variance estimate

---

## System Architecture

```
hawkes/
├── kernels.py          # Exponential kernel implementation
├── simulation.py       # Ogata's thinning algorithm
├── estimation.py       # Maximum likelihood estimation
└── diagnostics.py      # Time-rescaling residual diagnostics

lob/
├── order.py            # Order types and data structures
├── limit_order_book.py # Order book with price-time priority
└── event_processor.py  # Hawkes event to LOB operation mapping

agents/
├── inventory_control.py # Reservation price and inventory management
└── market_maker.py      # Market-making strategy

backtest/
├── metrics.py          # Performance metrics (Sharpe, drawdown, etc.)
└── engine.py           # Event-driven backtesting engine

experiments/
├── run_simulation.py   # Complete simulation pipeline
└── demo_features.py    # Component demonstrations

configs/
└── default.yaml        # Simulation parameters
```

---

## Installation and Usage

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: numpy, scipy, matplotlib, pyyaml, pandas

### Running Simulations

Complete pipeline (simulation, estimation, diagnostics, backtesting):

```bash
python experiments/run_simulation.py
```

Component demonstrations:

```bash
python experiments/demo_features.py
```

### Configuration

Edit `configs/default.yaml` to modify:
- Hawkes process parameters (baseline intensities, excitation matrix, decay rates)
- LOB parameters (tick size, initial depth, price levels)
- Agent parameters (inventory limits, risk aversion, spread targets)
- Simulation parameters (duration, random seed)

---

## Experimental Setup

### Hawkes Process Simulation

- **Method**: Ogata's thinning algorithm
- **Event types**: 6 (limit/market/cancel × buy/sell)
- **Parameters**: Configurable baseline intensities, excitation matrix, decay rates
- **Validation**: Stationarity check via spectral radius

### Parameter Estimation

- **Method**: Maximum likelihood estimation (MLE)
- **Optimization**: L-BFGS-B with positivity constraints
- **Complexity**: O(N²) per likelihood evaluation for N events

### Statistical Validation

- **Method**: Time-rescaling theorem
- **Tests**: Kolmogorov-Smirnov test for Exponential(1) residuals
- **Visualization**: Q-Q plots comparing sample to theoretical quantiles

### Backtesting

- **Engine**: Event-driven processing of Hawkes-generated order flow
- **Metrics**: PnL, Sharpe ratio, maximum drawdown, spread capture, inventory statistics
- **Frequency**: Quote updates every 0.1 time units

---

## Results Summary

### Hawkes Process

Typical simulation with default parameters:
- Total events: ~10,000
- Simulation time: 1,000 units
- Spectral radius: 0.65 (stationary)
- Event distribution: 60% limit orders, 25% market orders, 15% cancellations

### Statistical Validation

- KS test p-values: > 0.05 for all event types (model not rejected)
- Q-Q plots: Residuals align with Exponential(1) distribution
- Interpretation: Exponential kernel provides reasonable fit for simulated data

### Market-Making Performance

Typical results (default configuration):
- Final PnL: $150-250
- Sharpe ratio: 1.5-2.5
- Maximum drawdown: $30-50 (15-20% of peak)
- Number of trades: 200-400
- Spread capture: 1.5-2.0 ticks
- Inventory range: ±50 units (within limits)

---

## Assumptions and Limitations

This project makes several modeling assumptions and simplifications. **See `ASSUMPTIONS_AND_LIMITATIONS.md` for comprehensive technical analysis.**

### Key Assumptions

1. **Exponential kernels**: Closed-form integrals enable tractable MLE, but real markets may exhibit power-law decay
2. **Stationarity**: Parameters constant over time; real markets have regime changes
3. **Homogeneous excitation**: All events of same type treated equally regardless of size
4. **No market impact**: Orders don't move prices; real large orders have temporary and permanent impact
5. **Constant order sizes**: Simplifies dynamics; real markets have heavy-tailed size distributions
6. **Single asset**: No cross-asset correlation or portfolio effects
7. **Simplified adverse selection**: Imbalance-based filter; sophisticated informed traders may not be detected

### Limitations

- Exponential kernels only (power-law kernels not implemented)
- Stationary parameters (no regime-switching)
- No market impact modeling
- Constant order sizes (no size heterogeneity)
- Simple adverse selection detection
- No competitor market makers
- Stylized latency model (constant delay)
- Single asset only

### Classification

**Research-grade components:**
- Hawkes process theory and implementation
- MLE parameter estimation
- Time-rescaling diagnostics
- LOB matching engine
- Inventory control framework

**Simulation approximations:**
- Kernel choice (exponential only)
- Parameter stationarity
- Market impact (none)
- Order size distribution
- Adverse selection detection
- Competitor modeling (none)
- Multi-asset effects (none)

---

## Failure Modes

### Identified Failure Scenarios

1. **Flash crash**: Explosive dynamics when spectral radius approaches 1
2. **Adverse selection**: Agent picked off by informed traders with large orders
3. **Parameter estimation failure**: Local optima or insufficient data
4. **Inventory runaway**: Directional markets cause unbounded inventory accumulation
5. **Latency arbitrage**: Stale quotes exploited by faster traders

### Inappropriate Use Cases

Hawkes modeling is not suitable for:
1. Long-horizon forecasting (> 1 hour)
2. Low-frequency data (> 1 minute resolution)
3. Illiquid assets (< 10 events per minute)
4. Scheduled events (earnings announcements, FOMC)
5. Multi-day analysis with overnight gaps

---

## Future Extensions

**See `EXTENSIONS_ROADMAP.md` for detailed technical roadmap.**

### Priority 1: Production-Critical

1. Regime-switching Hawkes process (handle non-stationarity)
2. Machine learning-based adverse selection detection
3. Market impact model (concave temporary + linear permanent)
4. Online parameter adaptation (EWMA or Kalman filter)
5. Real data calibration (LOBSTER, NASDAQ ITCH)

### Priority 2: Research Extensions

6. Power-law kernels (long memory effects)
7. Multi-asset Hawkes (cross-asset correlation)
8. Order size heterogeneity (size-dependent excitation)
9. Competitor modeling (game-theoretic interactions)
10. Stochastic volatility integration

### Priority 3: Advanced Features

11. Neural Hawkes process (non-parametric learning)
12. Optimal execution algorithms (VWAP, TWAP)
13. Multi-venue trading
14. Regulatory constraints (position limits, circuit breakers)

---

## References

1. Bacry, E., Mastromatteo, I., & Muzy, J. F. (2015). "Hawkes processes in finance." *Market Microstructure and Liquidity*, 1(01), 1550005.

2. Abergel, F., & Jedidi, A. (2013). "A mathematical approach to order book modeling." *International Journal of Theoretical and Applied Finance*, 16(05), 1350025.

3. Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

4. Stoikov, S., & Waeber, R. (2016). "Reducing transaction costs with low-latency trading algorithms." *Quantitative Finance*, 16(10), 1445-1451.

5. Ogata, Y. (1981). "On Lewis' simulation method for point processes." *IEEE Transactions on Information Theory*, 27(1), 23-31.

6. Daley, D. J., & Vere-Jones, D. (2003). *An Introduction to the Theory of Point Processes: Volume I: Elementary Theory and Methods*. Springer.

---

## License

MIT License. See `LICENSE` file for details.

**Disclaimer:** This software is provided for educational and research purposes only. It is not intended for use in production trading systems. Trading financial instruments involves substantial risk of loss.

---

## Documentation

- `README.md` (this file): Project overview and technical summary
- `ASSUMPTIONS_AND_LIMITATIONS.md`: Detailed analysis of modeling assumptions, failure modes, and limitations
- `EXTENSIONS_ROADMAP.md`: Technical roadmap for addressing limitations
- `report/report_outline.md`: Comprehensive technical report with mathematical derivations
- `data/README.md`: Data directory documentation

---

## Contact

This is a research prototype demonstrating Hawkes process modeling for limit order book dynamics. For questions about the methodology or implementation, refer to the documentation files and academic references listed above.
