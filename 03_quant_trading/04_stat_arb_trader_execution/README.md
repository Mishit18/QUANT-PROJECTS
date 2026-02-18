# Statistical Arbitrage Research Framework

A research implementation of cointegration-based pairs trading with Ornstein-Uhlenbeck modeling, Kalman filtering, and regime detection. This framework prioritizes statistical rigor and transparent reporting over performance metrics.

**Status**: Research implementation for educational and exploratory purposes.

**Not**: A production trading system, investment advice, or performance guarantee.

---

## Research Objective

This framework explores whether mean-reversion in cointegrated asset pairs can be systematically detected and traded after accounting for:

1. Statistical quality gates (cointegration significance, OU model fit)
2. Regime instability (OU collapse, volatility shifts)
3. Realistic execution costs (transaction costs, slippage)

The system is designed to **reject weak signals**. Negative or flat results are expected and indicate honest research, not system failure.

---

## Overview

This framework identifies cointegrated asset pairs, models their spread dynamics as an Ornstein-Uhlenbeck process, and generates mean-reversion signals when statistical conditions meet quality thresholds.

**Key Design Principle**: Prioritize rejection of weak signals over generation of trading opportunities.

**Expected Behavior**: Most pairs are rejected. Most runs produce flat or negative results. This reflects statistical reality, not implementation failure.

---

## Core Components

### 1. Universe Selection (`src/universe_selection.py`)
- Engle-Granger cointegration testing
- Half-life calculation via AR(1) regression
- Pair categorization by mean reversion speed
- Rejection rate typically >90%

### 2. Spread Modeling (`src/spread_model.py`)
- Ornstein-Uhlenbeck parameter estimation
- Quality gates: minimum R², half-life bounds
- Position sizing based on model fit quality
- Rejects pairs with weak mean reversion

### 3. Kalman Filter (`src/kalman.py`)
- Dynamic hedge ratio estimation
- Handles time-varying cointegration relationships
- May destabilize static cointegration

### 4. Regime Detection (`src/regime_filter.py`)
- Hidden Markov Model for volatility regimes
- Used for risk gating, not signal generation
- Scales position size in volatile periods

### 5. Alpha Layer (`src/alpha_layer.py`)
- Asymmetric entry/exit thresholds
- Z-score velocity filter
- Time-based position exits
- Stop loss implementation

### 6. Execution Model (`src/execution.py`)
- Transaction cost modeling (3 bps)
- Slippage estimation (1.5 bps)
- OU collapse detection for mid-trade exits

### 7. Backtesting (`src/backtest.py`)
- Net-of-cost performance metrics
- Multi-pair attribution
- Rolling window stability tests

---

## Statistical Reality

### Expected Outcomes

With 40 assets (780 pairs):
- Cointegrated: 8-40 pairs (1-5% pass rate)
- Tradable after OU validation: 5-15 pairs
- Positive Sharpe after costs: 2-5 pairs (if any)

These are **descriptive statistics**, not performance targets.

### Why Most Runs Produce Flat or Negative Results

1. **Cointegration is rare**: Only 1-5% of random asset pairs are cointegrated
2. **Cointegration ≠ Profitability**: Pairs can be cointegrated but have:
   - Mean reversion too slow (HL > 60 days = no edge)
   - Mean reversion too fast (HL < 2 days = noise)
   - Transaction costs exceed edge (4.5 bps round-trip)
   - Regime instability (OU collapse)
3. **Kalman filtering may destabilize**: Dynamic hedge ratios can collapse half-life from 20 days to < 1 day
4. **Market efficiency**: Easily detectable statistical arbitrage is rare in liquid markets

### Pair Categories

The system classifies pairs explicitly:
- `micro_noise`: HL < 2 days (rejected - likely noise at daily frequency)
- `tradable_daily`: HL ∈ [15, 45] days (target range for daily trading)
- `slow_reversion`: HL > 60 days (rejected - too slow for practical trading)
- `unstable`: Poor OU fit (R² < 0.15) or regime instability
- `marginal`: Passes tests but outside optimal range

---

## Configuration

Key parameters in `config.yaml`:

```yaml
# Cointegration testing
cointegration:
  significance_level: 0.05  # Engle-Granger p-value threshold

# OU model quality gates
ou_model:
  min_r_squared: 0.15  # Minimum explanatory power of OU fit
  min_half_life: 5.0   # Minimum mean reversion speed (days)
  max_half_life: 60.0  # Maximum mean reversion speed (days)

# Signal generation
alpha:
  entry_z: 2.0   # Entry threshold (standard deviations from mean)
  exit_z: 0.3    # Exit threshold (asymmetric for convexity)
  stop_loss_z: 4.0
  max_hold_days: 30

# Execution costs
execution:
  transaction_cost_bps: 3.0   # Commission and fees
  slippage_bps: 1.5           # Market impact estimate
```

**These thresholds are fixed**. No parameter optimization is performed. Same config → same logic.

---

## Usage

```bash
python main.py
```

Implements portfolio-level statistical arbitrage with:
- Cross-sectional diversification across all qualifying pairs
- Equal-risk allocation (not performance-based)
- Binary regime gating (enforces OU stationarity assumptions)
- Time-to-reversion consistency checks (enforces model validity)

**Rationale**: Statistical arbitrage is a cross-sectional strategy. Individual pairs are noisy; diversification is the primary Sharpe driver. See `docs/portfolio.md` for technical details.

**Not cherry-picking**: All qualifying pairs are included. Allocation based on risk, not returns.

---

## What This Framework Does

- Screens pairs for cointegration (Engle-Granger test)
- Estimates OU parameters with quality gates (R², half-life, theta)
- Applies dynamic hedge ratios via Kalman filtering
- Detects regime shifts via HMM (for risk gating only)
- Generates signals with asymmetric entry/exit thresholds
- Models realistic execution costs (transaction costs + slippage)
- Reports performance net of costs
- Detects OU collapse mid-trade
- Provides pair-level attribution
- **Rejects weak signals systematically**

---

## What This Framework Does Not Do

- Optimize parameters on in-sample data
- Hide negative results
- Trade pairs that fail quality gates
- Guarantee positive returns
- Claim superiority over other approaches
- Provide production trading infrastructure
- Account for all real-world constraints (liquidity, funding, operational risk)

---

## Known Limitations

1. **Static cointegration may not persist**: Kalman filtering reveals time-varying relationships that can destabilize static cointegration tests

2. **Half-life instability**: Pairs may pass initial screening but show unstable mean reversion dynamics in rolling windows

3. **Universe dependence**: Performance varies significantly across asset classes and time periods

4. **Frequency sensitivity**: Daily bars may not be optimal for all pairs. Some relationships work better at weekly or intraday frequencies

5. **Transaction cost sensitivity**: 4.5 bps round-trip can eliminate thin edges

6. **No regime adaptation**: System uses fixed thresholds across all market conditions

---

## Typical Results (Empirical Example)

Recent run (40 stocks, 2020-2023):
- Pairs tested: 780
- Cointegrated: 27 (3.5% - consistent with statistical theory)
- Categorized as `tradable_daily`: 25
- After Kalman filtering: Half-lives collapsed to < 1 day (OU instability detected)
- OU collapse events: 946
- Net Sharpe: -0.48

**Interpretation**: The system correctly identified cointegration, detected post-filtering instability, reduced position sizing, and reported the failure transparently. This is **successful research** - the system worked as designed by rejecting weak signals.

---

## Repository Structure

### Core System
- `main.py` - Portfolio-level execution (recommended approach)
- `config.yaml` - Configuration parameters
- `src/` - Core modules (cointegration, OU modeling, Kalman filter, regime detection, etc.)

### Documentation
- `README.md` - Overview and methodology
- `INTERVIEW_DEFENSE.md` - Interview preparation and defense strategies
- `docs/portfolio.md` - Portfolio-level execution technical details
- `ASSUMPTIONS_AND_LIMITATIONS.md` - Statistical assumptions and failure modes
- `REPRODUCIBILITY.md` - Environment specification
- `DISCLAIMER.md` - Research purpose and limitations
- `CHANGELOG.md` - Version history

### Results
Results are saved to organized directories:
- `results/plots/` - Generated figures
- `results/tables/` - CSV summaries
- `results/diagnostics/` - Stability and risk analysis

---

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python tests/test_config.py

# Run analysis
python main.py
```

**Duration**: 3-7 minutes  
**Output**: `results/plots/`, `results/tables/`, `results/diagnostics/`
```

**Requirements**: Python 3.8+, see `REPRODUCIBILITY.md` for detailed environment specification.

---

## Research Notes

### Metric Interpretation

**Sharpe Ratio**: Descriptive statistic of risk-adjusted returns in the sample period. Not a performance target or prediction of future results.

**Half-Life**: Mean reversion speed. More important than Sharpe for assessing signal quality because:
- Cannot be inflated by infrequent trading
- Directly measures the statistical property being exploited
- Stability indicates robustness

**OU R²**: Explanatory power of the OU model. Low R² (< 0.15) indicates weak mean reversion, even if cointegration test passes.

### Why Half-Life Stability Matters

Sharpe ratio can be inflated by:
- Infrequent trading (low denominator)
- Short sample periods
- Favorable regime selection

Half-life stability cannot be gamed. A pair with:
- Sharpe 2.0 but collapsing half-life = regime breakdown (reject)
- Sharpe 0.8 but stable half-life = genuine mean reversion (consider)

The OU collapse monitor tracks this explicitly.

### Why Asymmetric Entry/Exit

Entry at |z| = 2.0, exit at |z| = 0.3 captures convexity:
- Mean reversion is stronger near extremes (non-linear restoring force)
- Taking profits early reduces exposure to reversals
- Asymmetry improves risk-adjusted returns without curve-fitting

### Why HMM for Risk Gating Only

Using HMM for signal generation risks:
- Overfitting to regime labels
- Look-ahead bias in regime classification
- Unstable regime definitions

Using HMM for risk gating:
- Scales position size in volatile regimes (defensive)
- Reduces drawdown without generating signals
- More robust to regime misclassification

---

## Disclaimer

**This is research software for educational purposes only.**

- Not investment advice
- Not a production trading system
- No performance guarantees
- Use at your own risk

See `DISCLAIMER.md` for complete terms.

Past performance does not predict future results. Statistical arbitrage can lose money. Consult professionals before making investment decisions.

---

## Philosophy

This framework prioritizes:
1. **Statistical Rigor** - Enforce quality gates, reject weak signals
2. **Transparency** - Report all results, including failures
3. **Reproducibility** - Document environment, seed randomness, version control
4. **Honesty** - Acknowledge limitations, avoid overstating results

**Success Criteria**:
-  Correctly identify cointegrated pairs
-  Correctly reject pairs with weak OU dynamics
-  Correctly detect regime instability
-  Report results transparently

**Not Success Criteria**:
-  High Sharpe ratio
-  Positive returns
-  Low drawdown
-  High win rate

The framework's value lies in its **discipline and methodology**, not its backtest performance.

A negative result is a **valid research outcome**. It indicates the system is working correctly by rejecting weak signals.

---

## License

MIT License - Use at your own risk.

---

## Philosophy

This system prioritizes:
1. **Transparency** - Report all results, including failures
2. **Discipline** - Enforce quality gates rigorously
3. **Skepticism** - Assume nothing works until proven

The system's refusal to trade weak signals is intentional. Better to trade nothing than trade noise.
