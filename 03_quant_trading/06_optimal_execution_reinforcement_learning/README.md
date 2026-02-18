# Optimal Trade Execution: Almgren–Chriss with Constrained RL

Implementation of optimal execution strategies combining Almgren–Chriss analytical solutions with offline reinforcement learning for stochastic market microstructure.

## Problem Statement

Liquidate inventory X₀ over horizon T to minimize:

```
J = E[Cost] + λ Var[Cost]
```

subject to market impact, stochastic liquidity, and execution constraints.

## Mathematical Framework

### Almgren–Chriss Model

**Dynamics:**
```
x_{t+1} = x_t - v_t                    (inventory)
S_{t+1} = S_t + σ√Δt ε_t - γv_t        (price)
```

**Cost:**
```
C = Σ v_t S_t + η Σ v_t²               (execution + temporary impact)
```

**Objective:**
```
min_{v_t} E[C] + λ Var[C]
```

**Closed-form solution:**
- Risk-neutral (λ=0): Uniform liquidation (TWAP)
- Risk-averse (λ>0): Exponential trajectory with urgency κ = √(λσ²/η)

**Assumptions:**
- Deterministic liquidity
- Linear temporary impact
- Constant volatility

**Limitations:**
- Fails under stochastic liquidity
- Cannot adapt to regime shifts
- Requires known parameters

### Stochastic Extension

AC optimality requires deterministic liquidity. Under stochastic microstructure:
- Liquidity varies (mean-reverting, regime-switching, shocks)
- Impact decays stochastically
- Parameters uncertain

RL agents learn state-dependent policies from offline data.

## Simulator Design

Gymnasium environment modeling:

**State:** `[inventory, time_remaining, price, volatility, liquidity, recent_impact]`

**Action:** Trade size as fraction of remaining inventory

**Dynamics:**
- Liquidity: Ornstein-Uhlenbeck process with optional shocks
- Impact: Linear temporary + permanent with stochastic decay
- Slippage: Gaussian noise
- Price: Diffusion + permanent impact

**Reward:** `-execution_cost - λ·risk_penalty - constraint_violations`

## Methods

### Benchmarks
- **TWAP:** Uniform liquidation
- **VWAP:** Volume-weighted (U-shaped profile)
- **AC:** Closed-form solutions (risk-neutral and risk-averse)

### Offline RL
- **BCQ:** Batch-constrained Q-learning with VAE behavioral model
- **TD3+BC:** Twin delayed DDPG with behavior cloning regularization

Both methods constrain policies to offline data support, preventing out-of-distribution actions.

## Experimental Setup

**Environment:**
- Initial inventory: 1000 shares
- Horizon: 20 periods
- Volatility: σ = 2%
- Impact: η = 0.01, γ = 0.001
- Risk aversion: λ = 0.5

**Training:**
- Offline data: 1000 episodes from TWAP
- Training iterations: 10,000
- Batch size: 256

**Evaluation:**
- 100 episodes per strategy
- Fixed seed for reproducibility

## Results

### Normal Conditions

| Strategy | Mean Cost | Std Cost | Sharpe |
|----------|-----------|----------|--------|
| TWAP | 100,250 | 1,850 | -54.19 |
| AC-Averse | 100,350 | 1,290 | -77.79 |
| BCQ | 99,850 | 1,650 | -60.52 |
| TD3+BC | 99,920 | 1,680 | -59.48 |

RL agents reduce mean cost by 0.3-0.4% while maintaining comparable variance.

### Stress Tests

| Scenario | Best Strategy | Performance |
|----------|---------------|-------------|
| Liquidity collapse | RL | 1.3% cost reduction vs AC |
| Volatility spike | AC-Averse | 30% variance reduction |
| Impact regime shift | RL | 2% cost reduction vs AC |
| Liquidity shocks | AC-Averse | Lower tail risk |

**Key findings:**
- AC optimal under stated assumptions
- RL outperforms under stochastic liquidity
- AC-Averse optimal for high volatility
- All strategies fail under extreme shocks (>80% liquidity drop)

## Installation and Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train RL agents
python experiments/train_rl.py --agent bcq --offline_episodes 1000

# Run benchmarks
python experiments/run_benchmarks.py

# Stress tests
python experiments/stress_tests.py

# Generate plots
python analysis/plots.py
```

## Repository Structure

```
├── env/              # Execution environment and market models
├── models/           # AC, TWAP, VWAP, BCQ, TD3+BC
├── experiments/      # Training and evaluation scripts
├── analysis/         # Metrics and visualization
├── report/           # Technical report with derivations
└── examples/         # Usage examples
```

## Limitations

**Model assumptions:**
- Liquidity is observable (reality: latent)
- Impact is deterministic given liquidity (reality: stochastic)
- No adverse selection
- Single asset (no cross-impact)
- No order book dynamics
- Continuous trading

**Failure modes:**
- All strategies fail under extreme liquidity shocks (>80% drop)
- RL generalizes poorly to out-of-distribution regimes
- AC cannot adapt to parameter changes
- Offline RL quality bounded by behavior policy

## Extensions

- Multi-asset portfolio execution
- Order book modeling and limit orders
- Online adaptation and meta-learning
- Robust RL for adversarial scenarios
- Calibration to market data

## References

1. Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk*, 3, 5-40.
2. Fujimoto, S., Meger, D., & Precup, D. (2019). Off-policy deep reinforcement learning without exploration. *ICML*.
3. Fujimoto, S., & Gu, S. S. (2021). A minimalist approach to offline reinforcement learning. *NeurIPS*.
