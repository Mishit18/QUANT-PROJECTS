# Model Assumptions and Approximations

## Overview

This document explicitly states the mathematical assumptions underlying each model. These assumptions are approximations that do not hold perfectly in real markets. Understanding where and why they break is critical for proper model usage.

## Kalman Filter Assumptions

### 1. Linear Gaussian State-Space Model

**Assumption:**
```
x_t = F_t x_{t-1} + w_t,  w_t ~ N(0, Q_t)  [State equation]
y_t = H_t x_t + v_t,      v_t ~ N(0, R_t)  [Observation equation]
```

**Reality:**
- Market dynamics are nonlinear (volatility clustering, regime switches)
- Returns are not Gaussian (fat tails, skewness)
- Noise is not independent (autocorrelation, heteroskedasticity)

**Implications:**
- Kalman filter is optimal only under these assumptions
- Suboptimal but often reasonable in practice
- Breaks down during crises (non-Gaussian shocks)

### 2. Known Noise Covariances

**Assumption:** Q_t and R_t are known or can be estimated accurately.

**Reality:**
- True noise covariances are unknown
- Must be estimated from data (introduces error)
- May be time-varying (heteroskedasticity)

**Implications:**
- Misspecified covariances lead to suboptimal filtering
- Too small Q → filter too slow to adapt
- Too large Q → filter too noisy
- Requires tuning or adaptive estimation

### 3. Stationarity

**Assumption:** Model parameters (F, H, Q, R) are constant or slowly varying.

**Reality:**
- Markets exhibit structural breaks
- Regime shifts change dynamics
- Parameters drift over time

**Implications:**
- Filter may diverge after regime shifts
- Requires periodic re-estimation
- HMM helps by modeling discrete regime changes

### 4. Observability

**Assumption:** Latent states can be inferred from observations.

**Reality:**
- Some states may be weakly observable
- Identification issues with multiple latent factors
- Observability depends on H matrix structure

**Implications:**
- Poorly observable states have high uncertainty
- Covariance matrices may become large
- Need sufficient excitation in data

## Extended Kalman Filter (EKF) Assumptions

### 1. Mild Nonlinearity

**Assumption:** Nonlinear functions f(x) and h(x) can be approximated by first-order Taylor expansion.

**Reality:**
- Strong nonlinearities violate this assumption
- Linearization error accumulates over time
- Jacobian may not exist everywhere

**Implications:**
- EKF can diverge with strong nonlinearities
- Requires small time steps
- May need iterated EKF for better accuracy

### 2. Differentiability

**Assumption:** State and observation functions are differentiable.

**Reality:**
- Some market dynamics have discontinuities
- Regime switches are non-differentiable
- Option payoffs have kinks

**Implications:**
- EKF fails at discontinuities
- Need alternative methods (UKF, particle filter)

## Unscented Kalman Filter (UKF) Assumptions

### 1. Gaussian Posterior

**Assumption:** Posterior distribution remains Gaussian after nonlinear transformation.

**Reality:**
- Nonlinear transformations can produce non-Gaussian posteriors
- Multimodality possible with strong nonlinearities
- Skewness and kurtosis not captured

**Implications:**
- UKF approximates mean and covariance only
- May miss multimodal distributions
- Better than EKF but not perfect

### 2. Sigma Point Representation

**Assumption:** 2n+1 sigma points adequately represent distribution.

**Reality:**
- High-dimensional systems need many points
- Computational cost scales with dimension
- May miss tail behavior

**Implications:**
- Works well for moderate dimensions (n < 20)
- Expensive for high-dimensional systems
- Consider particle filters for n > 50

## Hidden Markov Model Assumptions

### 1. Discrete Regimes

**Assumption:** Market dynamics belong to one of K discrete regimes.

**Reality:**
- Regimes may be continuous spectrum
- Transitions may be gradual, not instantaneous
- Number of regimes K is unknown

**Implications:**
- Discrete approximation of continuous reality
- K must be chosen (model selection problem)
- May over-fit with too many regimes

### 2. Markov Property

**Assumption:** Regime at time t depends only on regime at t-1.

**Reality:**
- Regime persistence may have longer memory
- Exogenous factors (policy, news) matter
- Seasonality and cycles not captured

**Implications:**
- First-order Markov may be insufficient
- Could extend to higher-order Markov
- Trade-off between complexity and overfitting

### 3. Gaussian Emissions

**Assumption:** Observations within each regime are Gaussian.

**Reality:**
- Returns have fat tails even within regimes
- Skewness varies by regime
- Outliers are common

**Implications:**
- Regime detection may be noisy
- Outliers can trigger false regime switches
- Consider robust alternatives (Student-t emissions)

### 4. Stationary Transition Matrix

**Assumption:** Transition probabilities A[i,j] are constant over time.

**Reality:**
- Regime persistence changes over time
- Crisis periods have different dynamics
- Structural breaks alter transition probabilities

**Implications:**
- Model may fail to adapt to changing regime dynamics
- Requires periodic re-estimation
- Time-varying transition matrices possible but complex

### 5. Known Number of Regimes

**Assumption:** Number of regimes K is specified a priori.

**Reality:**
- True number of regimes is unknown
- May vary over time
- Model selection is difficult

**Implications:**
- Must choose K via cross-validation or information criteria
- Too few regimes → underfitting
- Too many regimes → overfitting, regime collapse

## Signal Generation Assumptions

### 1. Regime Persistence

**Assumption:** Regimes persist long enough to be exploitable.

**Reality:**
- Some regimes are very short-lived
- High-frequency regime switching is noise
- Persistence varies by market and period

**Implications:**
- Need minimum regime duration for profitability
- Transaction costs eat profits from rapid switching
- May need regime probability threshold

### 2. Regime Predictability

**Assumption:** Current regime probabilities predict future returns.

**Reality:**
- Regime detection is backward-looking (smoothed probabilities)
- Filtered probabilities are noisy
- Regime switches are often unpredictable

**Implications:**
- Signals may lag regime changes
- Need to use filtered (not smoothed) probabilities
- Performance degrades with rapid regime switching

### 3. Volatility Targeting

**Assumption:** Historical volatility predicts future volatility.

**Reality:**
- Volatility is time-varying and clustered
- Volatility spikes are unpredictable
- Estimation error in volatility

**Implications:**
- Position sizing may be wrong during volatility spikes
- Need robust volatility estimation
- Consider GARCH or stochastic volatility models

## Backtesting Assumptions

### 1. Liquidity

**Assumption:** Can trade at observed prices without market impact.

**Reality:**
- Large orders move prices
- Bid-ask spreads vary
- Liquidity dries up in crises

**Implications:**
- Backtest overstates performance
- Need to model market impact
- Capacity constraints in real trading

### 2. Transaction Costs

**Assumption:** Fixed transaction cost per unit turnover.

**Reality:**
- Costs vary by market conditions
- Slippage is nonlinear in order size
- Costs spike during volatility

**Implications:**
- Backtest may understate costs
- Need realistic cost model
- High-frequency strategies especially affected

### 3. No Regime Shift in Backtest Period

**Assumption:** Historical regimes are representative of future.

**Reality:**
- New regimes emerge (COVID-19, policy changes)
- Structural breaks invalidate historical patterns
- Non-stationarity is the norm

**Implications:**
- Out-of-sample performance may differ dramatically
- Need stress testing on different periods
- Walk-forward validation helps but doesn't eliminate risk

## Summary: Why These Assumptions Matter

These assumptions are not failures of the model—they are necessary simplifications. The question is not whether they hold (they don't), but whether the approximations are good enough for the intended use case.

**For Research:**
- Assumptions are acceptable for understanding concepts
- Violations are learning opportunities
- Sensitivity analysis reveals robustness

**For Production:**
- Assumptions must be validated on live data
- Violations must be monitored in real-time
- Fallback strategies needed when assumptions break

**For Interviews:**
- Understanding assumptions demonstrates maturity
- Discussing violations shows practical experience
- Proposing improvements shows research ability

The best quants know their models are wrong. The question is: are they useful?
