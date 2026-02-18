# System Architecture

## Pipeline Overview

The system implements a unidirectional data flow from raw market data to trading signals:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Market    │────▶│ Preprocessing│────▶│   Kalman    │────▶│     HMM      │
│    Data     │     │              │     │   Filter    │     │   Regimes    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                │                      │
                                                └──────────┬───────────┘
                                                           │
                                                           ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────────┐
│  Evaluation │◀────│   Backtest   │◀────│    Signal Generation            │
│             │     │              │     │  (Regime-Aware + Risk Mgmt)     │
└─────────────┘     └──────────────┘     └─────────────────────────────────┘
```

## Module Responsibilities

### 1. Data Ingestion (`data_loader.py`)

**Purpose:** Fetch and validate market data

**Inputs:** Ticker symbols, date range  
**Outputs:** Price and return DataFrames

**Key Operations:**
- Download from Yahoo Finance with 3-retry logic
- Validate data completeness (minimum 252 days)
- Check for excessive NaNs (< 5% threshold)
- Compute log returns
- Raise explicit errors on failure (no synthetic fallback)

**Production Guarantees:**
- Real data only
- Explicit error messages
- Retry logic with exponential backoff
- Comprehensive validation

---

### 2. Preprocessing (`preprocessing.py`)

**Purpose:** Clean and normalize data

**Inputs:** Raw returns  
**Outputs:** Cleaned returns

**Key Operations:**
- Forward fill missing values (max 5 consecutive)
- Winsorize outliers (1st/99th percentile)
- Validate no NaN/Inf remain
- Optional standardization

**Production Guarantees:**
- No NaN propagation
- Bounded outlier treatment
- Validation at output

---

### 3. State-Space Models (`state_space_models.py`)

**Purpose:** Define model specifications

**Inputs:** Model parameters (observation variance, state variance)  
**Outputs:** Model specification (F, H, Q, R matrices)

**Models Implemented:**
- Local Level: `x[t] = x[t-1] + w[t]`
- Local Linear Trend: `x[t] = x[t-1] + v[t-1] + w[t]`
- Dynamic Regression: Time-varying coefficients

**Key Feature:**
- Pure specification (no estimation logic)
- Separates model definition from inference

---

### 4. Kalman Filter (`kalman_filter.py`, `extended_kalman_filter.py`, `unscented_kalman_filter.py`)

**Purpose:** Optimal state estimation

**Inputs:** Observations, state-space model  
**Outputs:** Filtered states, smoothed states, covariances

**Algorithm:**
```
For each observation t:
  1. Predict:    x[t|t-1] = F * x[t-1|t-1]
                 P[t|t-1] = F * P[t-1|t-1] * F' + Q
  
  2. Update:     K[t] = P[t|t-1] * H' * (H * P[t|t-1] * H' + R)^-1
                 x[t|t] = x[t|t-1] + K[t] * (y[t] - H * x[t|t-1])
                 P[t|t] = (I - K[t] * H) * P[t|t-1] * (I - K[t] * H)' + K[t] * R * K[t]'
                         └─────────────── Joseph Form ──────────────────────────────┘
```

**Production Guarantees:**
- Joseph form covariance update (numerically stable)
- NaN/Inf validation at every step
- Kalman gain clipping ([-1e3, 1e3])
- Pseudo-inverse fallback for singular matrices
- PSD enforcement via eigenvalue clipping

**Extensions:**
- EKF: Jacobian-based linearization for nonlinear dynamics
- UKF: Sigma-point transform for strongly nonlinear systems

---

### 5. HMM Regime Detection (`hmm_regimes.py`)

**Purpose:** Identify discrete market regimes

**Inputs:** Returns  
**Outputs:** Regime probabilities, transition matrix, regime statistics

**Algorithm:**
```
EM Algorithm:
  E-step: Forward-backward algorithm
          - Forward pass: α[t,k] = P(regime=k, y[1:t])
          - Backward pass: β[t,k] = P(y[t+1:T] | regime=k)
          - Smoothed: γ[t,k] = P(regime=k | y[1:T])
  
  M-step: Update parameters
          - Transition matrix: A[i,j] = Σ ξ[t,i,j] / Σ γ[t,i]
          - Emission means: μ[k] = Σ γ[t,k] * y[t] / Σ γ[t,k]
          - Emission variances: σ²[k] = Σ γ[t,k] * (y[t] - μ[k])² / Σ γ[t,k]
```

**Production Guarantees:**
- Convergence monitoring (max 100 iterations)
- Regime collapse detection (minimum probability mass)
- Transition matrix validation (rows sum to 1)
- Explicit error on non-convergence

**Outputs:**
- Filtered probabilities: P(regime[t] | data[1:t])
- Smoothed probabilities: P(regime[t] | data[1:T])
- Viterbi path: Most likely regime sequence

---

### 6. Signal Generation (`signals.py`, `signals_enhanced.py`)

**Purpose:** Convert model outputs to trading signals

**Inputs:** Returns, Kalman states, HMM regimes  
**Outputs:** Trading signals in [-max_leverage, +max_leverage]

**Baseline Strategy:**
```python
# Regime-aware Kalman trend
if regime == "trending":
    signal = sign(kalman_trend)
elif regime == "mean_reverting":
    signal = -sign(kalman_trend)
else:  # crisis
    signal = 0
```

**Enhanced Strategy:**
```python
# 1. Generate base signal (Kalman + TSMOM)
kalman_signal = sign(kalman_trend)
tsmom_signal = sign(cumulative_return[63:252])
base_signal = 0.6 * kalman_signal + 0.4 * tsmom_signal

# 2. Regime-conditioned position sizing
risk_budget = f(1 / regime_volatility)
scaled_signal = base_signal * Σ(P(regime_k) * risk_budget[k])

# 3. Volatility targeting
vol_scaling = target_vol / realized_vol
final_signal = scaled_signal * clip(vol_scaling, 0.3, 2.5)

# 4. Regime gating (optional)
if P(favorable_regime) < threshold:
    final_signal = 0
```

**Production Guarantees:**
- Signal bounds enforcement
- Leverage caps respected
- Fully causal (no lookahead)
- Vectorized operations

---

### 7. Backtesting (`backtest.py`)

**Purpose:** Simulate strategy execution

**Inputs:** Signals, returns, transaction costs  
**Outputs:** Equity curve, performance metrics

**Key Operations:**
```python
for t in range(T):
    # Lag positions by 1 period (causality)
    position[t] = signal[t-1]
    
    # Compute P&L
    pnl[t] = position[t] * return[t]
    
    # Transaction costs
    turnover[t] = abs(position[t] - position[t-1])
    costs[t] = turnover[t] * transaction_cost
    
    # Net P&L
    net_pnl[t] = pnl[t] - costs[t]
    
    # Equity
    equity[t] = equity[t-1] * (1 + net_pnl[t])
```

**Production Guarantees:**
- Positions lagged by 1 period (no lookahead)
- Transaction costs modeled
- Turnover tracking
- Walk-forward validation

---

### 8. Evaluation (`evaluation.py`)

**Purpose:** Compute performance metrics

**Inputs:** Returns, positions, regime labels  
**Outputs:** Sharpe, Sortino, drawdown, regime-conditional stats

**Metrics:**
- Sharpe Ratio: `mean(returns) / std(returns) * sqrt(252)`
- Sortino Ratio: `mean(returns) / std(negative_returns) * sqrt(252)`
- Max Drawdown: `max(peak - trough) / peak`
- Win Rate: `fraction(returns > 0)`
- Regime-Conditional: All metrics computed per regime

**Production Guarantees:**
- Annualization factors correct (252 trading days)
- Handles zero volatility gracefully
- Regime attribution accurate

---

## Separation of Concerns

### Why This Architecture?

**1. Testability**
- Each module has single responsibility
- Can test Kalman filter without HMM
- Can test HMM without signal generation
- Clear failure attribution

**2. Modularity**
- Easy to swap Kalman filter for particle filter
- Easy to try different regime detection methods
- Easy to add new signal strategies
- No circular dependencies

**3. Causality**
- Unidirectional data flow
- No component has knowledge of future
- Explicit position lagging in backtest
- Walk-forward validation enforced

**4. Production Readiness**
- Clear failure boundaries
- Explicit error handling at each stage
- Validation between stages
- Audit trail via logging

---

## Numerical Stability

### Critical Techniques

**1. Joseph Form Covariance Update**
```python
# Standard form (can lose PSD)
P = (I - K*H) * P_pred

# Joseph form (guarantees PSD)
P = (I - K*H) * P_pred * (I - K*H)' + K * R * K'
```

**2. Eigenvalue-Based PSD Enforcement**
```python
eigvals, eigvecs = np.linalg.eigh(P)
eigvals_clipped = np.maximum(eigvals, epsilon)
P_psd = eigvecs @ diag(eigvals_clipped) @ eigvecs.T
```

**3. Kalman Gain Clipping**
```python
K = P_pred @ H.T @ inv(S)
K = np.clip(K, -1e3, 1e3)  # Prevent explosions
```

**4. Pseudo-Inverse Fallback**
```python
try:
    S_inv = np.linalg.inv(S)
except LinAlgError:
    S_inv = np.linalg.pinv(S)  # Graceful degradation
```

---

## Failure Mode Handling

### Data Loading
- **Failure:** Network error, invalid ticker
- **Handling:** 3 retries with backoff, explicit error
- **No Silent Fallback:** Never use synthetic data

### Kalman Filter
- **Failure:** Filter divergence (covariance explosion)
- **Handling:** Joseph form, gain clipping, PSD enforcement
- **Detection:** NaN/Inf checks at every step

### HMM
- **Failure:** Non-convergence, regime collapse
- **Handling:** Max iterations, probability mass checks
- **Detection:** Explicit convergence flag

### Signal Generation
- **Failure:** Extreme signals, NaN propagation
- **Handling:** Bounds enforcement, validation
- **Detection:** Signal range checks

---

## Causality Enforcement

### Multiple Levels

**1. Kalman Filter**
- Uses only observations up to time t
- No smoothing in signal generation (smoothing is offline analysis only)

**2. HMM**
- Filtered probabilities: P(regime[t] | data[1:t])
- Smoothed probabilities: For analysis only, not trading

**3. Signal Generation**
- Uses filtered states and filtered regime probabilities
- No future information

**4. Backtesting**
- Positions lagged by 1 period: `position[t] = signal[t-1]`
- Explicit lag prevents lookahead bias

**5. Walk-Forward Validation**
- Train on [0, t]
- Test on [t+1, t+window]
- Roll forward, never look back

---

## Data Flow Example

### Single-Asset Strategy

```python
# 1. Load data
data = load_market_data(['SPY'])
returns = data['returns'].iloc[:, 0].values  # Shape: (T,)

# 2. Fit Kalman filter
model = LocalLevelModel(obs_var=1.0, state_var=0.1)
kf = KalmanFilter(model)
kf.filter(returns)
filtered_states = kf.filtered_states  # Shape: (T, state_dim)

# 3. Fit HMM
hmm = GaussianHMM(n_regimes=3)
hmm.fit(returns)
regime_probs = hmm.predict_proba(returns)  # Shape: (T, 3)

# 4. Generate signals
signals = create_regime_aware_strategy(
    returns, kf, hmm
)  # Shape: (T,), values in [-1, 1]

# 5. Backtest
bt = Backtest(signals, returns, transaction_cost=0.0005)
results = bt.run()

# 6. Evaluate
print(f"Sharpe: {results['sharpe_ratio']:.2f}")
print(f"Drawdown: {results['max_drawdown']:.2%}")
```

---

## Extension Points

### Adding New State-Space Models
```python
# Define in state_space_models.py
class MyCustomModel(StateSpaceModel):
    def get_matrices(self):
        return F, H, Q, R, x0, P0

# Use with existing Kalman filter
model = MyCustomModel(...)
kf = KalmanFilter(model)
kf.filter(observations)
```

### Adding New Regime Detection
```python
# Implement interface
class MyRegimeDetector:
    def fit(self, returns):
        # Your algorithm
        pass
    
    def predict_proba(self, returns):
        # Return (T, n_regimes) probabilities
        pass

# Use with existing signal generation
detector = MyRegimeDetector()
detector.fit(returns)
signals = create_regime_aware_strategy(returns, kf, detector)
```

### Adding New Signal Strategies
```python
# Inherit from BaseSignal
class MySignal(BaseSignal):
    def generate(self, returns, kf, hmm):
        # Your logic
        return signals

# Use with existing backtest
signal_gen = MySignal()
signals = signal_gen.generate(returns, kf, hmm)
bt = Backtest(signals, returns)
```

---

## Production Deployment Considerations

### What This System Provides
- Research framework
- Proof of concept
- production-ready code
- Numerical stability
- Failure mode handling

### What Production Requires (Not Included)
1. **Real-time data infrastructure**
   - Professional data vendor (not yfinance)
   - Tick data for execution
   - Data quality monitoring

2. **Execution system**
   - Order management system (OMS)
   - Smart order routing
   - Market impact modeling
   - Slippage tracking

3. **Risk management**
   - Real-time P&L monitoring
   - Position limits
   - Drawdown controls
   - Volatility circuit breakers
   - Stress testing

4. **Monitoring**
   - Model health checks
   - Parameter drift detection
   - Regime collapse alerts
   - Performance attribution
   - Anomaly detection

5. **Operational infrastructure**
   - Disaster recovery
   - Failover systems
   - 24/7 monitoring
   - Incident response
   - Audit trails

6. **Regulatory compliance**
   - Trade reporting
   - Best execution
   - Risk disclosures
   - Record keeping

---

## Summary

This architecture prioritizes:
- **Clarity** over brevity
- **Testability** over performance
- **Maintainability** over cleverness
- **Explicit failures** over silent degradation

For a research prototype and interview project, these are the correct trade-offs. The system demonstrates understanding of production concerns while remaining accessible and explainable.
