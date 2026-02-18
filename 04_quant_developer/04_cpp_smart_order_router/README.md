# Execution Engine

Multi-venue execution simulator implementing TWAP, VWAP, POV, and aggressive liquidity-seeking strategies with cost-based order routing.

## Overview

This system simulates order execution across multiple venues with independent order books. The smart order router minimizes expected execution cost using an explicit cost model. Strategies adapt to market conditions and track execution performance.

## Components

- **Order Book**: Central limit order book with price-time priority
- **Venues**: Simulated exchanges with configurable latency and liquidity
- **Strategies**: TWAP, VWAP (adaptive), POV, aggressive liquidity seeking
- **Smart Order Router**: Cost-based venue selection
- **Analytics**: Execution cost breakdown and performance metrics

## Building

```bash
# Windows
build.bat

# Linux/Mac
mkdir build && cd build
cmake ..
make
```

## Running

```bash
./build/execution_engine
```

Runs three example scenarios (TWAP, VWAP, Aggressive) and outputs:
- Console logs with order/fill details
- Performance reports
- CSV files: `twap_fills.csv`, `vwap_fills.csv`, `aggressive_fills.csv`

## Configuration

Venue and strategy parameters are configured in `main.cpp`. Key parameters:

**Venues** (`create_venues()`):
- Latency (mean, std dev)
- Initial spread and depth
- Fee structure

**Strategies**:
- TWAP: num_slices, slice_interval, aggression
- VWAP: volume_profile, deviation thresholds
- POV: target_participation, max_participation
- Aggressive: urgency_threshold, sweep_percentage

**Cost Model** (`CostModel::Config`):
- temporary_impact_coeff (default: 0.1)
- latency_penalty_bps (default: 0.001)
- queue_risk_coeff (default: 0.5)

**Queue Model** (`QueueModel::Config`):
- market_order_arrival_rate (default: 10.0/sec)
- cancellation_rate (default: 0.05/sec)
- queue_decay_rate (default: 0.1)

## Smart Order Router

Routes orders to minimize expected cost:

```
ExpectedCost = SpreadCost + TemporaryImpact + LatencyPenalty + QueueRiskPenalty
```

- **SpreadCost**: Bid-ask spread (full for market, partial for limit)
- **TemporaryImpact**: Square-root model `k * σ * sqrt(Q/V)`
- **LatencyPenalty**: Adverse selection cost
- **QueueRiskPenalty**: Fill probability uncertainty

Cost is adjusted by fill probability from stochastic queue model.

## Queue Model

Estimates fill probability using:
- Poisson process for market order arrivals
- Exponential survival for cancellations
- Latency-adjusted queue position

## Adaptive VWAP

Deviates from volume schedule when:
- Spread widens >10 bps
- Participation would exceed 1%
- Volume deviates >30% from forecast

Reduces order size by 50% when conditions are unfavorable.

## Market Impact

Temporary impact using Almgren-Chriss square-root model:
```
Impact = k * σ * sqrt(Q / V)
```

Where:
- k: Impact coefficient (0.1)
- σ: Volatility (estimated from spread)
- Q: Order size
- V: Venue liquidity

## Output

**Console**: Real-time order/fill logs and performance summary

**CSV**: Fill-by-fill data with timestamp, venue, price, quantity

**Metrics**:
- Implementation shortfall
- Slippage breakdown (spread + impact + delay)
- Fill rate and time to complete
- Venue distribution

## Assumptions

- Temporary impact dominates (no permanent impact modeled)
- Homogeneous order flow in queue model
- Constant arrival/cancellation rates
- No feedback loop (orders don't move simulated market)
- Single asset execution

## Dependencies

C++17 standard library only. No external dependencies.

## Structure

```
include/          Headers
src/              Implementation
  order_book/     CLOB matching engine
  venue/          Venue simulation
  execution/      Strategy implementations
  sor/            Smart order router
  cost_model/     Cost calculation
  queue_model/    Queue position modeling
  analytics/      Performance metrics
config/           YAML configuration examples
tests/            Unit tests
```

See ARCHITECTURE.md for system design details.
