# Execution Analytics Pack

## Desk-Style Question

Which execution policy should be preferred under stochastic liquidity, volatility, and impact assumptions, and where does it fail?

## Benchmark Ranking

| Strategy | Mean Cost | Std Cost | Completion | Cost Improvement vs TWAP |
|---|---:|---:|---:|---:|
| BCQ | -326,555.97 | 76,564.85 | 99.92% | 396,213.74 |
| TD3+BC | -50,852.37 | 53,791.24 | 99.80% | 120,510.13 |
| AC-Averse | 26,350.54 | 11,167.36 | 98.22% | 43,307.23 |
| VWAP | 69,650.05 | 4,481.15 | 93.90% | 7.71 |
| TWAP | 69,657.76 | 5,611.02 | 94.85% | 0.00 |
| AC-Neutral | 70,658.37 | 5,498.17 | 94.85% | -1,000.61 |

## Stress Ranking

| Strategy | Avg Stress Cost | Worst 95th Cost | Avg Completion | Max Failure Rate | Avg Tail Risk |
|---|---:|---:|---:|---:|---:|
| BCQ | -1,093,742.66 | -233,048.02 | 99.92% | 0.00% | 358,383.84 |
| TD3+BC | -616,754.32 | 16,121.12 | 99.84% | 0.00% | 253,313.27 |
| AC | -96,590.47 | 52,819.06 | 98.22% | 0.00% | 55,740.80 |
| TWAP | 27,821.20 | 100,121.92 | 94.85% | 0.00% | 21,868.27 |

## Decision

- Best mean-cost strategy in the simulator: BCQ with mean cost -326,555.97.
- Best completion strategy: BCQ with completion 99.92%.
- Best average stress-cost strategy: BCQ with average stress cost -1,093,742.66.
- TWAP baseline cost: 69,657.76; this is the benchmark any adaptive policy must beat.

## Caveat

The simulator cashflow convention allows negative "cost" values, so these results should be described as relative simulator performance, not live trading PnL. The correct interview framing is that the project builds an execution research harness with baselines, offline RL policies, stress tests, and explicit limitations.

## Interview Defense

**Why compare to Almgren-Chriss?** It is the analytical baseline for optimal liquidation under temporary/permanent impact assumptions.

**Why use offline RL?** Execution is a sequential decision problem where liquidity and impact states change over the schedule; offline RL can learn adaptive policies from simulated trajectories without online market risk.

**How would this go production?** Replace simulated liquidity with historical L2/L3 replay, calibrate impact parameters by symbol and participation rate, enforce kill-switches, and validate against implementation shortfall after fees and market impact.

## Artifacts

- `results/figures/execution_cost_by_strategy.png`
- `results/figures/completion_cost_frontier.png`
- `results/figures/stress_cost_by_scenario.png`
- `results/figures/stress_tail_risk.png`
