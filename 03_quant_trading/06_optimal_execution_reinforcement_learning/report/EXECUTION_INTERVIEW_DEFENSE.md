# Execution Interview Defense

## 30-Second Pitch

This project compares classical execution baselines, Almgren-Chriss schedules, and offline RL agents inside a stochastic execution simulator. The point is not to claim live trading alpha; it is to demonstrate execution-cost modeling, liquidity stress testing, benchmark discipline, and offline-RL risk awareness.

## What The Project Includes

- TWAP and VWAP baselines
- Almgren-Chriss risk-neutral and risk-averse baselines
- Stochastic liquidity and market-impact simulator
- Offline RL agents: BCQ and TD3+BC
- Stress tests for liquidity collapse, volatility spikes, impact regime shift, and liquidity shocks
- Exported benchmark and stress-test CSVs

## Core Concepts

### Implementation Shortfall

Implementation shortfall is the difference between the decision price and the realized execution price after market impact, spread, slippage, timing, and incomplete execution. It is the right metric family for execution research because the goal is cost minimization, not directional alpha.

### Almgren-Chriss Tradeoff

Almgren-Chriss balances expected market impact against inventory risk:

```text
min E[C] + lambda * Var[C]
```

Risk-neutral schedules trade more evenly. Risk-averse schedules trade earlier to reduce inventory exposure, accepting higher immediate impact.

### Offline RL Caveat

Offline RL cannot safely learn arbitrary actions outside the logged behavior distribution. BCQ constrains actions toward the support of historical behavior, while TD3+BC combines policy improvement with behavior cloning. This is exactly why execution RL needs strong baselines and stress tests.

## Benchmark Interpretation

The simulator's cost sign convention is environment-specific. Do not describe negative simulated cost as real PnL. The correct interpretation is relative policy behavior under a fixed simulator.

## Stress-Test Interpretation

| Scenario | What It Tests |
|---|---|
| Liquidity collapse | Whether the policy can complete under poor liquidity |
| Volatility spike | Whether inventory risk dominates impact cost |
| Impact regime shift | Whether learned policies exploit fixed impact assumptions |
| Liquidity shocks | Whether schedules adapt to non-smooth liquidity |

## What To Say In Interviews

- "I compare RL against TWAP, VWAP, and AC, not against a strawman."
- "The RL result is simulator evidence, not live trading proof."
- "The strongest lesson is conditional superiority: RL can adapt when AC assumptions break, but can also exploit simulator artifacts."
- "I would not deploy without real order book calibration, participation constraints, and walk-forward evaluation."

## Resume-Safe Bullet

Benchmarked offline RL agents (BCQ, TD3+BC) against TWAP/VWAP and Almgren-Chriss baselines in a stochastic execution simulator with liquidity collapse, volatility spike, and impact-regime stress tests; documented simulator-specific cost convention and failure modes instead of overclaiming live trading performance.
