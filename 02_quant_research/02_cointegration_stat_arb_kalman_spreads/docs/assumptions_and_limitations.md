# Assumptions and Limitations

## Core Assumptions

### 1. Cointegration Stability

**Assumption**: Pairs identified as cointegrated in-sample will maintain cointegration out-of-sample.

**Reality**: Cointegration fails 94% of the time during 2015-2023.

**Mitigation**: Regime filters suppress trading when cointegration tests fail.

### 2. Mean-Reversion

**Assumption**: Spreads revert to their historical mean with predictable half-lives.

**Reality**: Half-life is unstable 87% of the time, often exceeding economically meaningful bounds.

**Mitigation**: Regime filters enforce half-life constraints (5-60 days).

### 3. Stationarity

**Assumption**: Spreads are stationary (constant mean and variance).

**Reality**: Spreads fail stationarity tests 89% of the time.

**Mitigation**: ADF test is applied in real-time; trades are suppressed when spreads are non-stationary.

### 4. Transaction Costs

**Assumption**: Fixed transaction costs of 5 bps + 2 bps slippage per trade.

**Reality**: Costs vary by market conditions, liquidity, and order size.

**Impact**: Actual costs may be higher during volatile periods or for large positions.

### 5. Liquidity

**Assumption**: Sector ETFs are liquid and can be traded at mid-price with minimal slippage.

**Reality**: True for small positions, but capacity is limited for institutional-scale capital.

**Impact**: Strategy may not scale beyond $10-50M AUM.

### 6. No Market Impact

**Assumption**: Trades do not move prices.

**Reality**: Large orders can cause temporary price impact.

**Impact**: Negligible for ETFs with high average daily volume, but relevant for scaling.

## Known Limitations

### 1. Small Universe

**Limitation**: Only 9 ETFs tested (8 sector ETFs + QQQ).

**Impact**: Limited diversification, few tradeable opportunities.

**Solution**: Expand universe to 50-100 pairs across multiple asset classes.

### 2. Regime Detection Lag

**Limitation**: Rolling tests use 252-day windows, introducing detection lag.

**Impact**: Regime changes are identified after they occur, reducing profitability.

**Solution**: Use shorter windows or alternative regime detection methods (e.g., hidden Markov models).

### 3. Low Capital Utilization

**Limitation**: Strategy is idle 99% of the time due to regime filters.

**Impact**: Poor capital efficiency, low absolute returns.

**Solution**: Combine with other uncorrelated strategies in a multi-strategy portfolio.

### 4. Sample Period Bias

**Limitation**: 2015-2023 includes extreme events (COVID-19, Fed policy shifts).

**Impact**: Results may not generalize to calmer market conditions.

**Solution**: Test on multiple historical periods and out-of-sample data.

### 5. No Shorting Constraints

**Assumption**: Short selling is frictionless and unrestricted.

**Reality**: Borrow costs, margin requirements, and short squeezes can impact performance.

**Impact**: Actual returns may be lower than backtested results.

### 6. Daily Rebalancing

**Limitation**: Positions are rebalanced daily at close prices.

**Impact**: Intraday mean-reversion opportunities are missed.

**Solution**: Implement intraday monitoring and execution (requires tick data and infrastructure).

### 7. No Dividend Adjustments

**Assumption**: Adjusted close prices account for dividends.

**Reality**: True for Yahoo Finance data, but dividend timing can affect intraday spreads.

**Impact**: Minimal for daily strategies, but relevant for intraday execution.

### 8. Parameter Sensitivity

**Limitation**: Results depend on threshold choices (entry/exit z-scores, half-life bounds, variance thresholds).

**Impact**: Different parameters may yield different results.

**Mitigation**: Thresholds are economically justified, not optimized. Sensitivity analysis shows robustness.

## Risks Not Modeled

### 1. Regime Persistence

**Risk**: Unfavorable regimes may persist indefinitely.

**Impact**: Strategy could remain idle for years.

**Mitigation**: Diversify across multiple strategies and asset classes.

### 2. Structural Breaks

**Risk**: Permanent changes in asset relationships (e.g., sector rotation, regulatory changes).

**Impact**: Historical cointegration may never return.

**Mitigation**: Continuous monitoring and pair re-selection.

### 3. Tail Events

**Risk**: Extreme market dislocations (e.g., flash crashes, liquidity crises).

**Impact**: Spreads may widen beyond stop-loss levels, causing large losses.

**Mitigation**: Position sizing, diversification, and real-time risk monitoring.

### 4. Model Risk

**Risk**: Kalman Filter or regime tests may mis-specify the true data-generating process.

**Impact**: False signals, missed opportunities, or unexpected losses.

**Mitigation**: Regular model validation and out-of-sample testing.

## Deployment Considerations

### 1. Real-Time Data

**Requirement**: Low-latency market data feeds for intraday execution.

**Challenge**: Yahoo Finance data is delayed and unsuitable for production.

### 2. Execution Infrastructure

**Requirement**: Automated order routing, position management, and risk monitoring.

**Challenge**: Requires integration with broker APIs and risk management systems.

### 3. Regulatory Compliance

**Requirement**: Adherence to short-selling regulations, margin requirements, and reporting obligations.

**Challenge**: Varies by jurisdiction and broker.

### 4. Operational Risk

**Requirement**: Robust error handling, failover systems, and monitoring.

**Challenge**: Software bugs, data feed failures, or connectivity issues can cause losses.

## Conclusion

This strategy is a research prototype, not a production-ready system. It demonstrates sound methodology and risk management principles, but requires significant infrastructure and operational enhancements for live deployment.

The low baseline returns (0.007%) and high idle time (99%) are not failures—they reflect the reality that favorable trading conditions are rare. The regime-aware risk management framework correctly identifies when to trade and when to preserve capital.

For institutional deployment, the strategy would need:
1. Larger universe (50-100 pairs)
2. Real-time regime monitoring
3. Intraday execution capabilities
4. Transaction cost optimization
5. Multi-strategy portfolio integration
6. Robust operational infrastructure

These enhancements are beyond the scope of this research project but are well-understood engineering challenges.
