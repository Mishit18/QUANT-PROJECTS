# Methodology

## Overview

This project implements a cointegration-based statistical arbitrage strategy with regime-aware risk management. The approach combines classical econometric techniques with adaptive filtering to identify and exploit mean-reverting relationships in equity markets.

## Econometric Foundation

### Cointegration Testing

We employ the Johansen cointegration test to identify pairs of assets that share a long-run equilibrium relationship. The test examines whether a linear combination of non-stationary price series is stationary.

**Test Statistic**: Trace statistic  
**Significance Level**: 5%  
**Null Hypothesis**: No cointegration (r = 0)

### Stationarity Testing

Spread stationarity is verified using the Augmented Dickey-Fuller (ADF) test:
- **Null Hypothesis**: Unit root present (non-stationary)
- **Significance Level**: 5%
- **Test Regression**: Constant term included

## Time-Varying Hedge Ratios

### Kalman Filter Specification

The hedge ratio β_t evolves according to a state-space model:

**State Equation**:  
β_t = β_{t-1} + w_t, where w_t ~ N(0, Q)

**Observation Equation**:  
y_t = β_t * x_t + v_t, where v_t ~ N(0, R)

**Parameters**:
- Q (transition covariance): 1e-4
- R (observation covariance): 1e-2

The Kalman Filter recursively estimates β_t and its uncertainty (state covariance P_t).

## Mean-Reversion Metrics

### Half-Life Estimation

The half-life measures the expected time for the spread to revert halfway to its mean. We estimate it using an Ornstein-Uhlenbeck (OU) process:

dS_t = θ(μ - S_t)dt + σdW_t

**Half-life**: τ = ln(2) / θ

**Economically Meaningful Range**: 5-60 trading days

### Z-Score Calculation

Standardized spread deviation:

z_t = (S_t - μ) / σ

where μ and σ are computed over a rolling lookback window.

## Signal Generation

### Entry Rules

- **Long Spread**: z_t < -2.0 (spread undervalued)
- **Short Spread**: z_t > +2.0 (spread overvalued)

### Exit Rules

- **Close Position**: |z_t| < 0.5 (spread near equilibrium)

### Position Sizing

Fixed notional allocation per pair, with equal dollar exposure to both legs.

## Regime-Aware Risk Management

### Regime Filter

A regime is classified as tradeable only if ALL conditions are satisfied:

1. **Cointegration**: Johansen trace statistic > critical value (5% level)
2. **Stationarity**: ADF p-value < 0.05
3. **Half-Life**: 5 ≤ τ ≤ 60 days
4. **Variance Stability**: Current variance < 3× historical variance

**Rolling Window**: 252 trading days

### Kalman Confidence Gate

Trades are suppressed when:
- State covariance P_t > 0.1 (high uncertainty)
- Rolling β volatility > 0.05 (rapid regime change)

**Lookback Window**: 60 trading days

## Backtesting Framework

### Event-Driven Simulation

- Daily rebalancing
- Mark-to-market P&L calculation
- Transaction costs: 5 bps per trade
- Slippage: 2 bps per trade

### Performance Metrics

- Sharpe Ratio (annualized, risk-free rate = 0)
- Maximum Drawdown
- Win Rate
- Average Holding Period

## Data

**Universe**: Sector ETFs (XLK, XLV, XLF, XLE, XLI, XLY, XLP, XLU) + QQQ  
**Period**: 2015-2023  
**Frequency**: Daily adjusted close prices  
**Source**: Yahoo Finance

## Implementation Notes

- All tests use rolling windows to avoid lookahead bias
- Regime decisions are made at the start of each trading day
- Kalman Filter state is updated sequentially (no batch processing)
- Results are deterministic given fixed random seed
