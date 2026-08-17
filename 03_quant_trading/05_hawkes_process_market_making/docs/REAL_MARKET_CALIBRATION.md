# Real-Market Calibration Evidence

Empirical priors were calculated from 185,782 official Binance Vision BTCUSDT aggregate trades on 2026-08-15.

- Trade intensity: 2.15 trades/second
- Interarrival P50/P90: 174.2/1329.1 ms
- Trade-count variance/mean: 17.24, evidencing clustered arrivals
- One-second volatility: 0.101 bps
- Daily realized volatility: 0.30%

## Scope Boundary

Trade interarrival dispersion and signed-flow dependence validate clustering priors; the six-type LOB Hawkes kernel remains simulation-based because aggregate trades contain no limit-order or cancellation events.
