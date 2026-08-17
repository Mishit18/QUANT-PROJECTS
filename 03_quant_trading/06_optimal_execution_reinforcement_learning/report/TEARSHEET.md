# Optimal Execution Tear Sheet

## Objective

Compare analytical execution baselines and offline RL policies under stochastic liquidity, volatility, and market-impact assumptions.

## Strategies

- TWAP
- VWAP
- Almgren-Chriss risk-neutral
- Almgren-Chriss risk-averse
- BCQ offline RL
- TD3+BC offline RL

## Benchmark Snapshot

| Strategy | Mean Cost | Completion | Sharpe-like |
|---|---:|---:|---:|
| TWAP | 69,657.76 | 94.85% | -12.41 |
| VWAP | 69,650.05 | 93.90% | -15.54 |
| AC-Neutral | 70,658.37 | 94.85% | -12.85 |
| AC-Averse | 26,350.54 | 98.22% | -2.36 |
| BCQ | -326,555.97 | 99.92% | 4.27 |
| TD3+BC | -50,852.37 | 99.80% | 0.95 |

## Stress Tests

| Scenario | Purpose |
|---|---|
| Liquidity collapse | Test completion under poor market depth |
| Volatility spike | Test inventory-risk sensitivity |
| Impact regime shift | Test robustness to changed impact assumptions |
| Liquidity shocks | Test adaptation under discontinuous liquidity |

## Important Caveat

The simulator uses a liquidation cashflow convention, so negative cost must not be described as real trading PnL. The correct claim is relative policy behavior inside a fixed simulator.

## Resume-Safe Bullet

Benchmarked BCQ and TD3+BC offline RL against TWAP/VWAP and Almgren-Chriss baselines in a stochastic execution simulator, stress-testing liquidity collapse, volatility spikes, impact shifts, and liquidity shocks while documenting implementation-shortfall framing and simulator limitations.

## Added Desk Evidence

- `report/EXECUTION_ANALYTICS_PACK.md`
- `results/benchmark_ranked.csv`
- `results/stress_summary_by_strategy.csv`
- `results/figures/execution_cost_by_strategy.png`
- `results/figures/completion_cost_frontier.png`
- `results/figures/stress_cost_by_scenario.png`
- `results/figures/stress_tail_risk.png`
