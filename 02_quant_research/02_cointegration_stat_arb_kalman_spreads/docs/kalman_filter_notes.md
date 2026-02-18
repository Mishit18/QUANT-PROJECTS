# Kalman Filter Implementation Notes

## Purpose

The Kalman Filter estimates time-varying hedge ratios β_t for cointegrated pairs. Unlike static OLS regression, the Kalman Filter adapts to structural changes in the relationship between assets.

## State-Space Formulation

### State Equation

β_t = β_{t-1} + w_t

where w_t ~ N(0, Q) represents process noise.

**Interpretation**: The hedge ratio evolves as a random walk with small innovations.

### Observation Equation

y_t = β_t * x_t + v_t

where v_t ~ N(0, R) represents observation noise.

**Interpretation**: The observed price y_t is a linear function of x_t with noise.

## Parameter Selection

### Transition Covariance (Q)

**Value**: 1e-4

**Rationale**: Small Q implies the hedge ratio changes slowly. This prevents overfitting to short-term noise while allowing gradual adaptation to structural shifts.

### Observation Covariance (R)

**Value**: 1e-2

**Rationale**: R represents the variance of the spread residuals. A moderate value balances responsiveness and stability.

### Q/R Ratio

**Ratio**: 0.01

**Interpretation**: The filter trusts observations 100× more than state transitions, resulting in smooth β_t estimates.

## Recursive Estimation

### Prediction Step

β_{t|t-1} = β_{t-1|t-1}  
P_{t|t-1} = P_{t-1|t-1} + Q

### Update Step

K_t = P_{t|t-1} * x_t / (x_t² * P_{t|t-1} + R)  
β_{t|t} = β_{t|t-1} + K_t * (y_t - β_{t|t-1} * x_t)  
P_{t|t} = (1 - K_t * x_t) * P_{t|t-1}

where K_t is the Kalman gain.

## Initialization

**Initial State**: β_0 = OLS estimate from training period  
**Initial Covariance**: P_0 = 1.0

## Diagnostic Checks

### Beta Volatility

We monitor the rolling standard deviation of β_t over a 60-day window. High volatility (std > 0.05) indicates:
- Regime instability
- Structural breaks
- Model mis-specification

**Finding**: Beta volatility exceeds 0.05 during regime transitions, correctly signaling instability.

### State Covariance (P_t)

The state covariance P_t quantifies uncertainty in the β_t estimate. High P_t indicates:
- Insufficient data
- High observation noise
- Rapid state changes

**Finding**: P_t remains low (< 0.1) during stable regimes and increases during transitions.

## Comparison to Static OLS

### Advantages of Kalman Filter

1. **Adaptivity**: Tracks time-varying relationships
2. **Uncertainty Quantification**: Provides P_t as a confidence measure
3. **Sequential Processing**: Updates incrementally without retraining

### Disadvantages

1. **Parameter Sensitivity**: Results depend on Q and R choices
2. **Computational Cost**: More expensive than static regression
3. **Overfitting Risk**: Can track noise if Q is too large

## Validation Results

### Model Specification

The state-space model is correctly specified:
- Residuals are approximately white noise
- β_t estimates are smooth and economically interpretable
- P_t behaves as expected (low during stable periods, high during transitions)

### Beta Volatility as a Signal

High β volatility (std > 0.05) correctly identifies regime instability:
- Correlates with cointegration test failures
- Precedes spread variance explosions
- Aligns with known market stress periods (COVID-19, Fed policy changes)

**Conclusion**: Beta volatility is a feature, not a bug. It provides valuable information about regime state.

## Confidence-Based Trading

### Kalman Confidence Gate

We suppress trades when:
1. P_t > 0.1 (high state uncertainty)
2. Rolling β volatility > 0.05 (rapid changes)

**Rationale**: Low confidence in β_t estimates implies unreliable spread calculations and increased position risk.

### Empirical Results

The Kalman confidence gate provides minimal incremental value beyond the regime filter:
- Low confidence rate: 2.6%
- Sharpe improvement: Negligible

**Interpretation**: The regime filter (based on cointegration and stationarity tests) is sufficient. Kalman confidence adds redundant information.

## Implementation Notes

### Numerical Stability

- All matrix operations use double precision
- State covariance P_t is bounded below by 1e-6 to prevent numerical underflow
- Kalman gain K_t is clipped to [0, 1] to ensure stability

### Fallback to OLS

If the Kalman Filter fails (e.g., due to numerical issues), the system falls back to static OLS regression. This ensures robustness in production.

### Determinism

Given fixed Q, R, and initial conditions, the Kalman Filter produces deterministic results. No random number generation is involved.

## Future Enhancements

Potential improvements (not implemented):
1. **Adaptive Q/R**: Estimate Q and R from data using maximum likelihood
2. **Multi-Factor Models**: Extend to multiple cointegrating vectors
3. **Regime-Switching Kalman Filter**: Allow Q and R to vary by regime

These enhancements are beyond the scope of this research project but could improve performance in production.
