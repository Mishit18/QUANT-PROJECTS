# Technical Report: Optimal Execution with Offline RL

## Summary

This project combines the Almgren-Chriss optimal execution framework with a stochastic execution simulator and offline reinforcement learning agents. The implemented research stack includes:

- TWAP and VWAP baseline schedules.
- Almgren-Chriss risk-neutral and risk-averse policies.
- Stochastic liquidity and market-impact simulation.
- BCQ and TD3+BC offline RL agents.
- Trained checkpoints from 500 offline TWAP episodes and 2,000 training iterations per agent.
- Benchmark and stress-test result exports.

The strongest technical value is not only the RL model implementation, but the ability to compare analytical execution baselines against adaptive policies under changed liquidity, volatility, and impact assumptions.

## Almgren-Chriss Framework

The discrete liquidation problem is:

```text
min_v E[C] + lambda * Var[C]
```

with:

```text
x_{t+1} = x_t - v_t
S_{t+1} = S_t + sigma * sqrt(dt) * epsilon_t - gamma * v_t
```

Risk-neutral AC collapses toward uniform liquidation. Risk-averse AC trades earlier to reduce inventory risk, accepting higher instantaneous impact for lower exposure.

## Simulator

The environment exposes a six-dimensional state:

```text
[inventory, time_remaining, price, volatility, liquidity, recent_impact]
```

The action is a scalar fraction of remaining inventory. The environment includes temporary impact, permanent impact, random slippage, stochastic liquidity, and terminal inventory penalties.

## Offline RL

BCQ constrains action selection to the support of the offline behavior data through a VAE-style behavioral model. TD3+BC combines critic-guided policy improvement with behavior cloning regularization. Both agents are useful for discussing the central offline-RL risk: a policy may look strong inside the simulator while still being unsafe outside the logged data distribution.

## Verified Artifacts

Smoke test:

```text
python test_installation.py
Passed: 6/6
```

Training command:

```text
python experiments/train_rl.py --agent both --offline_episodes 500 --training_iterations 2000 --batch_size 256 --device cpu
```

Generated artifacts:

- `models/checkpoints/bcq.pt`
- `models/checkpoints/td3_bc.pt`
- `results/benchmark_results.csv`
- `results/stress_test_results.csv`

## Benchmark Interpretation

The simulator's `total_cost` field follows a cashflow convention for liquidation, so negative values should not be described as real profit. They indicate the policy's cashflow profile inside this environment.

In the verified benchmark, BCQ and TD3+BC both achieve near-complete liquidation, while AC-Averse improves completion and cost profile versus TWAP/VWAP baselines under the chosen assumptions.

## Failure Modes

- Simulator-based RL can exploit modeling assumptions.
- Offline data is generated from TWAP behavior, so action support is limited.
- There is no real order book, queue position, adverse selection, or partial-fill simulator.
- Market impact is simplified through parametric temporary/permanent terms.
- Cost sign convention must be handled carefully in resume and interview discussion.

## Next Research Extensions

- Add real LOB/TAQ data calibration.
- Add implementation shortfall as the primary metric.
- Include participation-rate constraints and order-size limits.
- Add walk-forward evaluation across liquidity regimes.
- Compare against Almgren-Chriss with calibrated market impact.
- Add Conservative Q-Learning or Implicit Q-Learning baselines.
