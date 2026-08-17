# ATS Screening Pack: Optimal Execution

## Keywords

Optimal execution, Almgren-Chriss, TWAP, VWAP, implementation shortfall, market impact, temporary impact, permanent impact, liquidity, volatility, offline reinforcement learning, BCQ, TD3+BC, transaction costs, stress testing, execution cost, slippage, benchmark, risk aversion.

## Resume Bullets

- Implemented optimal execution research stack comparing TWAP/VWAP, Almgren-Chriss risk-neutral/risk-averse schedules, and offline RL agents under stochastic liquidity and market-impact simulation.
- Benchmarked BCQ and TD3+BC against analytical baselines across liquidity collapse, volatility spike, impact regime shift, and liquidity-shock stress tests; exported reproducible benchmark and stress-test result tables.
- Documented simulator-specific cost convention, offline-RL action-support risk, and non-deployment caveats to keep execution research claims interview-defensible.

## Interview Anchors

- Derive the AC objective: expected cost plus risk penalty.
- Explain why execution is cost minimization, not alpha generation.
- Explain why offline RL is dangerous outside behavior support.
- Explain why stress testing matters more than a single benchmark table.
