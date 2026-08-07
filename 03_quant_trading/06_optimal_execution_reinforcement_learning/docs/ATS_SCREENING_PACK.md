# ATS Screening Pack: Optimal Execution with Offline RL

## Best-Fit Resume Profiles

- Quant Research
- Quant Trader
- ML for Trading
- Execution Research
- Systematic Trading Research

## Recruiter Summary

Built an optimal execution research framework combining Almgren-Chriss analytical baselines, stochastic liquidity simulation, market impact modeling, offline RL agents (BCQ and TD3+BC), trained checkpoints, benchmark evaluation, stress testing, and exportable result tables.

## ATS Keyword Coverage

Optimal execution, Almgren-Chriss, transaction cost analysis, implementation shortfall, market impact, temporary impact, permanent impact, liquidity shocks, stochastic liquidity, market microstructure, TWAP, VWAP, offline reinforcement learning, BCQ, TD3+BC, behavior cloning, replay buffer, Gymnasium environment, stress testing, benchmark evaluation, risk aversion, inventory liquidation, Python, NumPy, PyTorch, pandas, quantitative finance.

## Quant Research Resume Bullets

- Implemented optimal execution research stack with Almgren-Chriss analytical policies, TWAP/VWAP baselines, stochastic liquidity simulator, and offline RL agents for liquidation policy comparison.
- Trained BCQ and TD3+BC checkpoints from 500 offline TWAP episodes over 2,000 iterations per agent, then benchmarked against AC/TWAP/VWAP baselines across 100 simulated episodes.
- Added stress-test harness for liquidity collapse, volatility spikes, impact regime shifts, and liquidity shocks, exporting reproducible benchmark tables to CSV.
- Fixed evaluation bugs in RL action-shape handling and mixed strategy interfaces, enabling clean comparison across rule-based, analytical, and RL policies.

## Quant Trader Resume Bullets

- Built execution-cost simulator with inventory, time, price, volatility, liquidity, recent impact, temporary impact, permanent impact, slippage, and terminal inventory penalties.
- Compared TWAP, VWAP, AC risk-neutral, AC risk-averse, BCQ, and TD3+BC policies under normal and stressed execution conditions.
- Achieved near-complete liquidation in benchmark runs for trained BCQ and TD3+BC agents, with explicit documentation of simulator cashflow convention and model-risk limitations.
- Produced reproducible execution research artifacts: trained checkpoints, benchmark CSVs, stress-test CSVs, smoke tests, and technical report.

## Strong Interview Defense

The strongest explanation is:

1. Almgren-Chriss is the analytical baseline when liquidity and impact assumptions are stable.
2. RL is useful when the policy must adapt to stochastic liquidity and shocks.
3. Offline RL must be constrained because out-of-distribution actions can look good in simulation and fail in production.
4. Benchmarks need stress scenarios, not just normal-condition averages.
5. This project is a simulator research stack; production execution would require real fill modeling, impact calibration, and live risk controls.

## Claims To Avoid

- Do not claim live execution readiness.
- Do not claim real exchange profitability.
- Do not describe negative simulator cost values as trading profit.
- Do not claim robust RL superiority without real market-data validation.

## Upgrade Path For 100/100 Screening

- Add real LOB/TAQ calibration.
- Use implementation shortfall as the primary cost metric.
- Add order-book fill probability and queue position.
- Add CQL/IQL baselines.
- Add walk-forward validation across market regimes.
- Add participation-rate, ADV, and max-child-order constraints.
