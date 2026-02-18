# Market Impact Models: Failure Mode Analysis

## Overview

This document provides a systematic analysis of when and why each market impact model fails. Understanding failure modes is critical for model selection and risk management in practical trading applications.

---

## 1. Kyle's Lambda Model

### 1.1 Linearity Failure

**Failure Type:** Non-linear impact for large orders

**Quantitative Evidence:**

| Regime | Order Size Range | λ Variation | CV | Status |
|--------|------------------|-------------|-----|--------|
| Low    | Q1 → Q5          | 0.18 → 1.28 | 0.72| FAIL   |
| Medium | Q1 → Q5          | 0.09 → 0.28 | 0.38| FAIL   |
| High   | Q1 → Q5          | 0.04 → 0.14 | 0.42| FAIL   |

In low liquidity, λ increases 7× from smallest to largest orders. This violates the fundamental linearity assumption: ΔP ∝ Q.

**Economic Explanation:**

Small orders (Q1-Q3, 75% of volume) trade against standing limit orders with approximately constant λ. Large orders (Q4-Q5, 25% of volume) exhaust order book depth, moving through multiple price levels. Impact becomes super-linear: ΔP ∝ Q^α with α > 1.

OLS regression fits a line through this convex relationship, overestimating the slope to capture large-order impact. This produces upward-biased λ estimates in low liquidity (113% overestimate).

**When This Matters:**

- Institutional trading with large metaorders
- Thin markets with limited depth
- Block trades exceeding 75th percentile of order size distribution
- Execution algorithms that don't account for non-linearity

**Mitigation Strategies:**

1. Use quantile regression to estimate λ separately for different order sizes
2. Switch to Obizhaeva-Wang model with √Q impact (concave, more realistic)
3. Implement order splitting to keep individual orders below linearity threshold
4. Apply non-parametric methods (splines, kernel regression) for large orders

### 1.2 Regime Instability

**Failure Type:** Parameter varies 4× across liquidity regimes

**Quantitative Evidence:**

| Regime | λ (bps) | Deviation from Mean | Relative Error |
|--------|---------|---------------------|----------------|
| Low    | 0.425   | +113%               | 113%           |
| Medium | 0.149   | +49%                | 49%            |
| High   | 0.072   | -28%                | 28%            |

Mean λ = 0.215 bps, CV = 0.67 (highly unstable).

**Economic Explanation:**

λ reflects adverse selection intensity, which depends on market microstructure. Thin markets (low liquidity) have few informed traders but high information asymmetry per trade. Deep markets (high liquidity) have many traders but low information asymmetry per trade.

**When This Matters:**

- Cross-regime trading (e.g., crisis vs normal periods)
- Intraday regime shifts (open/close vs midday)
- Cross-asset trading with different liquidity profiles
- Long-term parameter stability assumptions

**Mitigation Strategies:**

1. Implement regime detection (Hidden Markov Model, volatility clustering)
2. Use rolling window estimation (e.g., 30-day calibration window)
3. Maintain regime-specific parameter sets
4. Apply Bayesian updating for smooth parameter transitions

### 1.3 Low Explanatory Power

**Failure Type:** R² < 2% across all regimes

**Quantitative Evidence:**

| Regime | R² | Variance Explained | Variance Unexplained |
|--------|-----|-------------------|---------------------|
| Low    | 1.14%| 1.14%             | 98.86%              |
| Medium | 0.34%| 0.34%             | 99.66%              |
| High   | 0.16%| 0.16%             | 99.84%              |

**Economic Explanation:**

This is NOT a model failure—it's economically realistic. Most price movement is noise (σ × dW), not impact (λ × Q). In liquid markets, noise dominates. Impact is a small but systematic component of price dynamics.

Low R² indicates:
- Market is efficient (prices are mostly random)
- Impact is small relative to volatility
- Other factors matter (news, sentiment, liquidity shocks)

**When This Matters:**

- Misinterpreting low R² as model failure
- Expecting high predictive power from impact models
- Ignoring other sources of price movement
- Overconfidence in impact estimates

**Mitigation Strategies:**

1. Accept low R² as realistic for liquid markets
2. Focus on statistical significance (t-test) rather than R²
3. Combine impact model with volatility model for total cost estimation
4. Use confidence intervals to quantify estimation uncertainty

---

## 2. Obizhaeva-Wang Model

### 2.1 Regime Dependence

**Failure Type:** γ varies 2.7× across regimes

**Quantitative Evidence:**

| Regime | γ (Permanent) | Deviation from Mean | CV |
|--------|---------------|---------------------|-----|
| Low    | 10.5%         | +56%                | 0.51|
| Medium | 5.8%          | -14%                |     |
| High   | 3.9%          | -42%                |     |

Mean γ = 6.7%, CV = 0.51 (unstable, exceeds 0.5 threshold).

**Economic Explanation:**

Permanent fraction reflects information content of trades. In thin markets, trades reveal more information (higher γ) because fewer traders participate and adverse selection is concentrated. In deep markets, trades reveal less information (lower γ) because many traders participate and information is dispersed.

**When This Matters:**

- Cross-regime execution cost estimation
- Long-term parameter stability assumptions
- Portfolio execution across assets with different liquidity
- Crisis periods with regime shifts

**Mitigation Strategies:**

1. Calibrate γ separately for each regime
2. Use regime-conditional models
3. Implement adaptive estimation with regime detection
4. Apply Bayesian priors based on market microstructure

### 2.2 Exponential Decay Assumption

**Failure Type:** Real decay may be power-law or multi-exponential

**Quantitative Evidence:**

Log-linear regression R² for transient component:

| Regime | R² (log-linear) | Exponential Fit Quality |
|--------|-----------------|------------------------|
| Low    | 0.78            | Good                   |
| Medium | 0.84            | Good                   |
| High   | 0.81            | Good                   |

Our data validates exponential decay (R² > 0.75), but real markets may exhibit power-law decay (Bouchaud) or multi-exponential decay (multiple time scales).

**Economic Explanation:**

Exponential decay assumes single time scale for order book recovery. Real markets have multiple time scales:
- Fast decay: Immediate order book replenishment (seconds)
- Medium decay: Market maker inventory adjustment (minutes)
- Slow decay: Information diffusion (hours)

Single exponential cannot capture all three scales.

**When This Matters:**

- Long execution horizons (multi-hour or multi-day)
- Markets with multiple liquidity providers
- Persistent impact beyond exponential prediction
- Comparison to empirical decay curves

**Mitigation Strategies:**

1. Test power-law decay (Bouchaud model) as alternative
2. Use multi-exponential models with multiple time scales
3. Validate decay assumption with log-linear regression
4. Switch models based on decay diagnostics

### 2.3 Time-Varying Parameters

**Failure Type:** γ and ρ assumed constant within regime

**Quantitative Evidence:**

Within-regime parameter variation (not directly measured in this study, but known from literature):
- Intraday patterns: γ higher at open/close, lower at midday
- Volatility dependence: ρ increases with volatility
- Volume dependence: γ decreases with volume

**Economic Explanation:**

Market microstructure changes continuously. Order book depth varies with time of day, volatility, and trading activity. Assuming constant parameters ignores these dynamics.

**When This Matters:**

- Intraday execution spanning multiple hours
- High-volatility periods with rapid microstructure changes
- Volume spikes or droughts
- Long-term parameter stability

**Mitigation Strategies:**

1. Use rolling window estimation (e.g., 1-hour windows)
2. Model parameters as functions of state variables (volatility, volume)
3. Implement time-of-day adjustments
4. Apply Kalman filtering for time-varying parameters

---

## 3. Bouchaud Propagator Model

### 3.1 Parameter Instability

**Failure Type:** β varies 2.7× across regimes

**Quantitative Evidence:**

| Regime | β (Exponent) | Deviation from Mean | CV |
|--------|--------------|---------------------|-----|
| Low    | 0.30         | -53%                | 0.46|
| Medium | 0.80         | +26%                |     |
| High   | 0.80         | +26%                |     |

Mean β = 0.63, CV = 0.46 (moderately unstable, below 0.5 threshold but significant).

**Economic Explanation:**

β controls decay rate: lower β = slower decay = longer memory. Thin markets have persistent impact (β = 0.30) because order book recovery is slow. Deep markets have transient impact (β = 0.80) because order book recovery is fast.

**When This Matters:**

- Cross-regime execution
- Long-term parameter stability
- Model selection (when is power-law necessary vs exponential sufficient?)
- Computational cost (power-law is expensive, want to avoid when unnecessary)

**Mitigation Strategies:**

1. Use β < 0.5 as threshold for power-law necessity
2. Switch to OW exponential when β > 0.7 (simpler, faster)
3. Calibrate β separately for each regime
4. Implement regime-conditional model selection

### 3.2 Computational Cost

**Failure Type:** Convolution is O(T²) vs OW's O(1)

**Quantitative Evidence:**

| Execution Horizon | Bouchaud Operations | OW Operations | Ratio |
|-------------------|---------------------|---------------|-------|
| 10 periods        | 100                 | 1             | 100×  |
| 60 periods        | 3,600               | 1             | 3,600×|
| 120 periods       | 14,400              | 1             | 14,400×|

For 120-period execution, Bouchaud requires 14,400× more operations than OW.

**Economic Explanation:**

Bouchaud computes impact as convolution: ΔP_t = Σᵢ G(t-i) × Q_i. This requires summing over all past trades, with cost proportional to T². OW has closed-form solution with constant cost.

**When This Matters:**

- Real-time trading systems with latency constraints
- High-frequency execution
- Large-scale portfolio optimization
- Backtesting over long time series

**Mitigation Strategies:**

1. Use OW for real-time trading, Bouchaud for offline analysis
2. Implement fast convolution (FFT) for long horizons
3. Truncate memory horizon to reduce cost
4. Pre-compute kernel values for common horizons

### 3.3 Memory Horizon Limitation

**Failure Type:** Model valid only for execution < 60 periods

**Quantitative Evidence:**

| Execution Horizon | Cumulative Impact | Memory Exceeded | Status |
|-------------------|-------------------|-----------------|--------|
| 30 periods        | 0.12              | No              | VALID  |
| 60 periods        | 0.18              | No              | VALID  |
| 120 periods       | 0.24              | Yes             | FAIL   |

Beyond 60 periods, cumulative impact exceeds memory horizon, violating model assumptions.

**Economic Explanation:**

Power-law decay implies infinite memory, but we impose finite memory horizon (60 periods) to prevent divergence. This is a regularization choice, not a theoretical property. Real markets have finite memory due to information diffusion and regime changes.

**When This Matters:**

- Multi-day execution (> 2 hours at 1-minute intervals)
- Very large orders requiring slow execution
- Long-term impact persistence
- Comparison to empirical impact decay

**Mitigation Strategies:**

1. Increase memory horizon for long execution (with computational cost)
2. Switch to multi-exponential model for very long horizons
3. Segment long execution into multiple shorter executions
4. Validate memory horizon against empirical decay curves

### 3.4 Fast Decay Regime

**Failure Type:** Power-law overkill when β > 0.7

**Quantitative Evidence:**

| Regime | β | OW Half-Life | Decay Comparison | Recommendation |
|--------|---|--------------|------------------|----------------|
| Low    | 0.30 | 0.16      | Power-law much slower | Bouchaud necessary |
| Medium | 0.80 | 0.15      | Similar decay    | OW sufficient |
| High   | 0.80 | 0.20      | Similar decay    | OW sufficient |

For β > 0.7, power-law and exponential decay are similar over relevant time scales.

**Economic Explanation:**

When decay is fast (β > 0.7), impact vanishes quickly regardless of functional form. Power-law provides no advantage over exponential but adds computational cost and complexity.

**When This Matters:**

- Model selection for liquid markets
- Computational efficiency requirements
- Simplicity vs accuracy trade-off
- Real-time trading systems

**Mitigation Strategies:**

1. Use β = 0.5 as threshold: β < 0.5 → Bouchaud, β > 0.7 → OW
2. Test both models and compare fit quality
3. Default to OW unless long memory is detected (Hurst > 0.7)
4. Reserve Bouchaud for thin markets and long horizons

---

## 4. Cross-Model Failure Patterns

### 4.1 Regime Dependence (All Models)

All three models exhibit regime-dependent parameters:

| Model | Parameter | CV | Stability |
|-------|-----------|-----|-----------|
| Kyle  | λ         | 0.67| Unstable  |
| OW    | γ         | 0.51| Unstable  |
| OW    | ρ         | 0.53| Unstable  |
| Bouchaud | β      | 0.46| Moderate  |

No model has universal parameters. All require regime-specific calibration.

**Implication:** Regime detection is critical for all models. Without regime identification, parameter estimates are unreliable.

### 4.2 Low R² (Kyle and Bouchaud)

Both Kyle and Bouchaud have R² < 2%:

| Model | Mean R² | Interpretation |
|-------|---------|----------------|
| Kyle  | 0.54%   | Impact explains < 1% of variance |
| Bouchaud | 0.06% | Impact explains < 0.1% of variance |
| OW    | N/A     | Not directly comparable (different specification) |

**Implication:** Low R² is realistic, not a failure. Most price movement is noise. Focus on statistical significance and economic interpretation, not R².

### 4.3 Large Order Breakdown (Kyle and OW)

Both Kyle and OW assume specific functional forms that break down for large orders:

| Model | Assumption | Breakdown Point | Alternative |
|-------|-----------|-----------------|-------------|
| Kyle  | Linear (λ × Q) | Q > 75th percentile | OW (√Q) or quantile regression |
| OW    | Square-root (√Q) | Q > 95th percentile | Bouchaud or non-parametric |

**Implication:** No single functional form works for all order sizes. Use piecewise models or non-parametric methods for extreme orders.

---

## 5. Practical Decision Framework

### 5.1 Model Selection Flowchart

```
1. Determine order size:
   - Small (< 75th percentile) → Kyle
   - Medium (75-95th percentile) → OW
   - Large (> 95th percentile) → Bouchaud or non-parametric

2. Determine execution horizon:
   - Short (< 10 periods) → Kyle or OW
   - Medium (10-60 periods) → OW
   - Long (> 60 periods) → Bouchaud (if β < 0.5)

3. Determine liquidity regime:
   - High liquidity → Kyle or OW
   - Medium liquidity → OW
   - Low liquidity → Bouchaud (if long horizon)

4. Check computational constraints:
   - Real-time required → Kyle or OW
   - Offline analysis → Any model

5. Validate assumptions:
   - Test linearity (Kyle)
   - Test exponential decay (OW)
   - Test power-law decay (Bouchaud)
   - Switch models if assumptions fail
```

### 5.2 Failure Detection Checklist

Before deploying any model, validate:

**Kyle:**
- [ ] Linearity: λ constant across order size quantiles (CV < 0.3)
- [ ] Residuals normal: Jarque-Bera p > 0.05
- [ ] No autocorrelation: Durbin-Watson ≈ 2
- [ ] Homoskedastic: Breusch-Pagan p > 0.05
- [ ] Statistical significance: p < 0.05

**OW:**
- [ ] Constraints satisfied: γ ∈ [0,1], ρ > 0
- [ ] Exponential decay: Log-linear R² > 0.7
- [ ] Finite half-life: Half-life < 1000
- [ ] No NaN values
- [ ] Parameter stability: CV < 0.5

**Bouchaud:**
- [ ] Constraints satisfied: β ∈ [0.3, 0.8], A > 0
- [ ] Power-law decay: Log-log R² > 0.6
- [ ] Long memory: Hurst > 0.5
- [ ] Memory horizon: Execution < memory horizon
- [ ] Optimization converged

### 5.3 Mitigation Priority

When failures occur, prioritize mitigation by impact:

**High Priority (Trading Losses):**
1. Kyle linearity failure for large orders → Switch to OW
2. OW constraint violations (γ < 0 or γ > 1) → Recalibrate with bounds
3. Bouchaud memory horizon exceeded → Increase horizon or switch to OW

**Medium Priority (Estimation Errors):**
1. Regime instability (CV > 0.5) → Implement regime detection
2. Time-varying parameters → Use rolling windows
3. Non-exponential decay → Test Bouchaud

**Low Priority (Efficiency):**
1. Computational cost → Optimize implementation
2. Low R² → Accept as realistic
3. Parameter uncertainty → Use confidence intervals

---

## 6. Conclusion

All three models have well-defined failure modes that are predictable and manageable. Kyle fails for large orders and extreme regimes but works well for small-to-medium orders. OW fails when decay is non-exponential or parameters vary within regimes but works well for medium orders with exponential decay. Bouchaud fails when decay is fast or computational cost matters but works well for thin markets with long horizons.

No single model is universally superior. Model selection depends on order size, execution horizon, liquidity regime, and computational constraints. Systematic validation and failure detection are essential for reliable impact estimation in practice.
