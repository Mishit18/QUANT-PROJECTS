# Limitations and Failure Modes

## Overview

This document catalogs known failure modes, edge cases, and scenarios where the model breaks down. Understanding failure modes is as important as understanding successes.

## Critical Limitations

### 1. Tail Risk Not Captured

**Problem:** Gaussian assumptions miss fat tails and extreme events.

**Failure Scenario:**
- March 2020 (COVID crash): -12% daily moves
- October 1987 (Black Monday): -20% single day
- August 2015 (China devaluation): Flash crash

**Why It Fails:**
- Kalman filter assumes Gaussian noise
- HMM assumes Gaussian emissions
- Extreme events are 10+ sigma under Gaussian assumption

**Consequences:**
- Filter diverges during crises
- Regime detection fails (all regimes look the same)
- Signals become unreliable
- Drawdowns exceed historical estimates

**Mitigation:**
- Use robust filters (Student-t noise)
- Add crisis regime with fat-tailed distribution
- Implement volatility circuit breakers
- Reduce leverage during high volatility

### 2. Regime Collapse

**Problem:** HMM collapses to single regime or oscillates rapidly.

**Failure Scenario:**
- Insufficient data for K regimes
- All regimes have similar statistics
- Transition matrix becomes degenerate

**Why It Fails:**
- EM algorithm finds local optimum
- Initialization matters (k-means may fail)
- Identifiability issues with similar regimes

**Consequences:**
- Regime probabilities become uniform (uninformative)
- Signals revert to non-regime-aware baseline
- Computational waste (fitting useless model)

**Detection:**
- Monitor regime probability entropy
- Check transition matrix eigenvalues
- Validate regime separation (Bhattacharyya distance)

**Mitigation:**
- Use information criteria (BIC) to select K
- Require minimum regime duration
- Regularize transition matrix (Dirichlet prior)
- Fall back to simpler model if collapse detected

### 3. Regime Over-Switching

**Problem:** HMM switches regimes too frequently.

**Failure Scenario:**
- High-frequency noise mistaken for regime change
- Transition probabilities too high
- Insufficient regime persistence

**Why It Fails:**
- Model interprets volatility clustering as regime switches
- Overfitting to training data
- Lack of regime duration constraint

**Consequences:**
- Excessive turnover (transaction costs)
- Signals become noisy
- Strategy degenerates to random trading

**Detection:**
- Monitor average regime duration
- Check transition matrix diagonal elements
- Measure signal autocorrelation

**Mitigation:**
- Add minimum regime duration constraint
- Penalize rapid switching in EM algorithm
- Use regime probability threshold (e.g., > 0.7)
- Smooth regime probabilities

### 4. Kalman Filter Divergence

**Problem:** Filter covariance explodes or state estimates become unrealistic.

**Failure Scenario:**
- Misspecified noise covariances (Q, R)
- Numerical instability in matrix operations
- Observability issues

**Why It Fails:**
- Joseph form helps but doesn't eliminate all issues
- Extreme observations can cause divergence
- Poorly conditioned covariance matrices

**Consequences:**
- State estimates become meaningless
- Signals based on diverged filter are garbage
- System must be restarted

**Detection:**
- Monitor state covariance trace
- Check Kalman gain magnitude
- Validate innovation sequence

**Mitigation:**
- Clip Kalman gains (already implemented)
- Reset filter if divergence detected
- Use square-root filtering for better numerics
- Adaptive noise covariance estimation

### 5. Lookahead Bias

**Problem:** Using future information in signal generation.

**Failure Scenario:**
- Using smoothed (not filtered) regime probabilities
- Fitting models on full dataset including test period
- Peeking at future returns for parameter tuning

**Why It Fails:**
- Smoothed probabilities use future data
- Overfitting to specific historical period
- Unrealistic backtest performance

**Consequences:**
- Backtest Sharpe ratio inflated
- Live trading performance disappoints
- Strategy fails in production

**Detection:**
- Verify position lag in backtest
- Check that models use only historical data
- Walk-forward validation

**Mitigation:**
- Use filtered (not smoothed) probabilities for signals
- Strict train/test separation
- Walk-forward backtesting
- Out-of-sample validation

### 6. Parameter Instability

**Problem:** Optimal parameters change over time.

**Failure Scenario:**
- Kalman filter noise covariances estimated on 2010-2015
- Deployed in 2020 (COVID era)
- Regime structure has changed

**Why It Fails:**
- Markets are non-stationary
- Structural breaks invalidate historical estimates
- Regime dynamics evolve

**Consequences:**
- Filter becomes suboptimal
- Regime detection misses new regimes
- Strategy performance degrades

**Detection:**
- Monitor out-of-sample performance
- Track parameter drift over time
- Compare recent vs historical regime statistics

**Mitigation:**
- Periodic re-estimation (e.g., quarterly)
- Adaptive parameter estimation
- Ensemble of models with different parameters
- Monitor and alert on parameter drift

### 7. Overfitting

**Problem:** Model fits noise rather than signal.

**Failure Scenario:**
- Too many regimes (K = 10)
- Too many state dimensions
- Optimizing on limited data

**Why It Fails:**
- Degrees of freedom exceed information in data
- Model memorizes specific historical patterns
- No generalization to new data

**Consequences:**
- Perfect in-sample fit
- Poor out-of-sample performance
- Strategy fails in live trading

**Detection:**
- Compare in-sample vs out-of-sample Sharpe
- Check parameter stability across periods
- Information criteria (BIC penalizes complexity)

**Mitigation:**
- Use simple models (K = 2 or 3 regimes)
- Regularization (L2 penalty on parameters)
- Cross-validation
- Require minimum data per parameter

## Specific Crisis Scenarios

### 2008 Financial Crisis

**What Happens:**
- Volatility spikes 5x
- Correlations go to 1
- Liquidity disappears

**Model Behavior:**
- Kalman filter diverges (extreme observations)
- HMM detects crisis regime (if trained on similar data)
- Signals may go risk-off (good) or freeze (bad)

**Outcome:**
- If crisis regime exists: strategy reduces exposure (good)
- If no crisis regime: model breaks, large losses (bad)

### 2020 COVID Crash

**What Happens:**
- Fastest drawdown in history
- Unprecedented policy response
- V-shaped recovery

**Model Behavior:**
- Regime detection lags (backward-looking)
- By the time crisis regime detected, worst is over
- Misses recovery (still in crisis mode)

**Outcome:**
- Likely underperforms during crash and recovery
- Transaction costs from whipsawing
- Demonstrates regime detection lag

### Flash Crashes

**What Happens:**
- Intraday spike down and recovery
- Liquidity vacuum
- Prices disconnect from fundamentals

**Model Behavior:**
- Daily data may not capture intraday dynamics
- If captured, looks like extreme outlier
- Kalman filter may diverge

**Outcome:**
- Daily model unaffected (good)
- Intraday model would break (bad)
- Demonstrates importance of data frequency

## What Would Be Needed for Production

### Data Infrastructure
- Real-time data feeds (not yfinance)
- Tick data for execution
- Corporate actions handling
- Survivorship bias correction

### Execution System
- Order management system
- Smart order routing
- Market impact modeling
- Slippage tracking

### Risk Management
- Real-time P&L monitoring
- Position limits
- Drawdown controls
- Volatility circuit breakers
- Correlation monitoring

### Monitoring and Alerting
- Model health checks
- Parameter drift detection
- Regime collapse alerts
- Performance attribution
- Anomaly detection

### Regulatory and Compliance
- Trade reporting
- Best execution
- Risk disclosures
- Audit trails

### Operational
- Disaster recovery
- Failover systems
- 24/7 monitoring
- Incident response

## When NOT to Deploy This Model

**Don't deploy if:**
1. You don't have crisis regime in training data
2. You can't monitor model health in real-time
3. You don't have proper risk management
4. You can't handle model divergence gracefully
5. You don't understand the failure modes
6. You're using it for high-frequency trading (too slow)
7. You're trading illiquid assets (model assumes liquidity)
8. You don't have out-of-sample validation
9. You can't explain it to risk committee
10. You're using it with real money without paper trading first

## Summary

This model is a research prototype. It demonstrates concepts and techniques used in production systems, but it is not production-ready. The gap between research and production is large:

**Research:** Does it work on historical data?  
**Production:** Does it work on live data, every day, for years, without blowing up?

The answer to the first question is "sometimes."  
The answer to the second question is "not without significant additional work."

Use this for learning, interviews, and research. Not for trading real money.
