# Regime Instability Analysis

## Motivation

Statistical arbitrage strategies rely on the assumption that cointegration relationships are stable over time. However, structural breaks, regime shifts, and changing market dynamics can cause these relationships to deteriorate. This analysis quantifies the stability of cointegration over the backtest period.

## Rolling Cointegration Tests

### Methodology

We apply the Johansen cointegration test over a rolling 252-day window throughout the entire sample period. This allows us to track the strength and persistence of cointegration relationships.

### Key Findings

**Cointegration Failure Rate**: 94%

The trace statistic falls below the 5% critical value approximately 94% of the time, indicating that cointegration is not consistently present.

**Interpretation**: The pairs identified as cointegrated in-sample do not maintain stable relationships out-of-sample. This explains the low baseline returns and justifies the need for regime-aware trading controls.

## Rolling Stationarity Tests

### Methodology

We apply the ADF test to the spread over rolling windows to assess whether mean-reversion properties persist.

### Key Findings

**Stationarity Failure Rate**: 89%

The spread fails the ADF test (p-value > 0.05) approximately 89% of the time.

**Interpretation**: The spread exhibits non-stationary behavior during most of the sample period, violating the core assumption of mean-reversion strategies.

## Half-Life Instability

### Methodology

We estimate the half-life of mean-reversion over rolling windows using OU process parameters.

### Key Findings

**Unstable Half-Life Rate**: 87%

The half-life falls outside the economically meaningful range (5-60 days) approximately 87% of the time. Common failure modes include:
- Half-life > 60 days (slow reversion, high holding risk)
- Half-life → ∞ (no reversion, random walk behavior)

**Interpretation**: Mean-reversion speed is highly variable and often too slow for practical trading.

## Variance Explosions

### Methodology

We monitor rolling spread variance and flag periods where variance exceeds 3× the historical average.

### Key Findings

**Variance Explosion Rate**: 23%

Spread variance explodes during approximately 23% of the sample period, particularly during:
- 2020 COVID-19 market disruption
- 2021-2022 Fed policy uncertainty
- 2023 banking sector stress

**Interpretation**: Volatility regimes change dramatically, increasing position risk during unstable periods.

## Implications for Trading

### Why Baseline Returns Are Low

The baseline strategy (without regime filters) trades continuously, including during the 94% of time when cointegration assumptions fail. This results in:
- Frequent false signals
- Positions held during non-mean-reverting periods
- Exposure to regime shifts

**Baseline Return**: 0.007% over 8 years

This is not a bug—it is information. The low return reflects the reality that favorable trading conditions are rare.

### Why Regime Filters Improve Performance

By suppressing trades when statistical tests fail, the regime filter:
- Reduces exposure during unstable periods
- Improves risk-adjusted returns (Sharpe ratio)
- Accepts long idle periods as the cost of capital preservation

**Filtered Sharpe Improvements**:
- XLK-XLV: 0.05 → 0.21 (4× improvement)
- QQQ-XLV: 0.23 → 0.36 (57% improvement)

## Regime Transition Dynamics

### Persistence

Regimes are not persistent. Tradeable periods are short-lived, typically lasting 5-20 days before conditions deteriorate.

### Detection Lag

Rolling tests introduce a detection lag of approximately 252 days (the window size). By the time a regime is confirmed, it may already be ending.

**Implication**: Real-time deployment would require shorter detection windows or alternative regime identification methods.

## Conclusion

The regime instability analysis reveals that:
1. Cointegration is not stable over the 2015-2023 period
2. Mean-reversion properties are intermittent
3. Variance regimes shift unpredictably
4. Trading should be conditional on regime state

These findings justify the regime-aware risk management framework and explain why the strategy is idle 99% of the time.
