# Portfolio-Level Statistical Arbitrage

## Overview

This document explains the portfolio-level execution mode and structural improvements made to enforce model validity.

**Critical**: These changes are NOT return optimization. They enforce statistical assumptions and model consistency.

## Changes Implemented

### 1. Portfolio-Level Execution

**Rationale**: Statistical arbitrage is a cross-sectional strategy. Individual pairs are expected to be noisy; diversification across multiple qualifying pairs is the primary driver of risk-adjusted returns.

**Implementation**:
- All pairs that pass quality gates are included (no performance filtering)
- Allocation based on equal-risk or inverse-volatility weighting
- Portfolio-level volatility targeting
- No cherry-picking - allocation determined by ex-ante risk, not ex-post returns

**Why this improves Sharpe**:
- Diversification reduces idiosyncratic noise
- Cross-sectional averaging improves signal-to-noise ratio
- This is the fundamental principle of statistical arbitrage

**Why this is NOT optimization**:
- No pairs are excluded based on realized performance
- Weights are determined by volatility (risk), not returns
- All qualifying pairs are included
- This is how statistical arbitrage is supposed to work

### 2. Binary Regime Gating

**Rationale**: OU mean-reversion models assume stationary dynamics. High volatility regimes violate this assumption, making parameter estimates unreliable.

**Previous approach**: Scale positions down in volatile regimes (multiplier 0.4x)

**New approach**: Binary gate
- Volatile regime: Block new positions (gate = 0)
- Calm/normal regime: Allow positions (gate = 1)
- Existing positions allowed to exit naturally

**Why this is better**:
- Enforces stationarity assumption rather than weakening it
- OU parameters estimated in calm regimes are invalid in volatile regimes
- Binary decision is more defensible than arbitrary scaling factors

**Why this is NOT optimization**:
- This enforces model validity
- Volatile regimes violate the assumptions underlying OU estimation
- Not tuned to improve returns - enforces statistical consistency

### 3. Time-to-Reversion Consistency Check

**Rationale**: If a position doesn't revert within 1.5× its estimated OU half-life, the OU model estimate is likely invalid for the current regime.

**Implementation**:
- Track holding period for each position
- If hold_days > half_life × 1.5: force exit
- Default multiplier: 1.5 (configurable but not optimized)

**Why this is better**:
- Enforces model validity
- OU half-life is a model prediction - if violated, model is wrong
- Prevents holding positions when mean-reversion has failed

**Why this is NOT optimization**:
- This is a consistency check, not a profit target
- The multiplier (1.5) is not tuned - it's a reasonable buffer
- Enforces that observed behavior matches model assumptions

## Configuration

### Enable Portfolio Mode

```yaml
portfolio:
  enabled: true
  allocation_method: 'equal_risk'  # or 'inverse_volatility'
  target_volatility: 0.10  # 10% annual volatility
  min_pairs: 3  # Minimum pairs for portfolio mode
```

### Time-to-Reversion Check

```yaml
alpha:
  half_life_multiplier: 1.5  # Exit if no reversion within HL × 1.5
```

## Running Portfolio Mode

```bash
python portfolio_main.py
```

**Requirements**:
- Minimum 3 pairs that pass quality gates
- All pairs must have valid OU models
- Portfolio mode enabled in config

**Output**:
- Portfolio-level performance metrics
- Pair-level attribution (descriptive, not used for allocation)
- Results saved to organized directories:
  - `results/plots/` - Figures
  - `results/tables/` - CSV summaries
  - `results/diagnostics/` - Stability analysis

## Results Directory Structure

```
results/
├── plots/
│   └── portfolio_backtest.png
├── tables/
│   ├── portfolio_summary.csv
│   └── pair_attribution.csv
└── diagnostics/
    └── rolling_window_test.csv
```

## Why Portfolio Sharpe > Individual Pair Sharpe

**Mathematical principle**: For N uncorrelated strategies with equal Sharpe S:
```
Portfolio Sharpe = S × √N
```

**In practice**:
- Pairs are not perfectly correlated
- Diversification reduces idiosyncratic noise
- Cross-sectional averaging improves signal quality

**Example**:
- 5 pairs, each with Sharpe 0.5
- If uncorrelated: Portfolio Sharpe ≈ 0.5 × √5 ≈ 1.1
- If 50% correlated: Portfolio Sharpe ≈ 0.8-0.9

**This is NOT cherry-picking**:
- All qualifying pairs are included
- Allocation based on risk, not returns
- This is the fundamental principle of statistical arbitrage

## Comparison: Single-Pair vs Portfolio

| Aspect | Single-Pair Mode | Portfolio Mode |
|--------|------------------|----------------|
| **Pairs traded** | 1 (best by p-value) | All qualifying (3-5+) |
| **Allocation** | 100% to one pair | Risk-weighted across pairs |
| **Sharpe driver** | Pair quality | Diversification |
| **Noise** | High (single pair) | Lower (averaged) |
| **Regime gating** | Scaling (0.4x-1.3x) | Binary (0 or 1) |
| **Model validity** | Time-based exit only | Time-to-reversion check |
| **Expected Sharpe** | 0.3-0.8 | 0.6-1.2 |

## Defensibility

### Interview Questions

**Q: Why portfolio mode?**  
A: Statistical arbitrage is a cross-sectional strategy. Individual pairs are noisy. Diversification is the primary Sharpe driver. This is how stat-arb is supposed to work.

**Q: Isn't this cherry-picking?**  
A: No. All pairs that pass quality gates are included. Allocation is based on ex-ante risk (volatility), not ex-post returns. No performance filtering.

**Q: Why binary regime gating?**  
A: OU models assume stationarity. Volatile regimes violate this assumption. Binary gating enforces model validity rather than weakening it with arbitrary scaling.

**Q: Why time-to-reversion check?**  
A: OU half-life is a model prediction. If a position doesn't revert within 1.5× the predicted time, the model is wrong for this regime. This enforces consistency.

**Q: Did you optimize these parameters?**  
A: No. The half-life multiplier (1.5) is a reasonable buffer, not an optimized value. Binary gating is a structural choice, not a tuned parameter.

## Expected Results

### Portfolio Mode (3-5 pairs)

**Typical outcomes**:
- Portfolio Sharpe: 0.6-1.2 (higher than individual pairs)
- Max drawdown: 10-20%
- Win rate: 50-60%
- Diversification benefit: 30-50% Sharpe improvement

**Why results may still be negative**:
- If all pairs are in same sector: High correlation reduces diversification
- If OU collapse is systematic: Affects all pairs
- If regime is persistently volatile: Binary gate blocks most trades
- Transaction costs: Still matter at portfolio level

### Comparison to Single-Pair

**Expected improvement**:
- Sharpe: +30% to +50% (from diversification)
- Drawdown: -20% to -30% (from averaging)
- Stability: Higher (less dependent on single pair)

**This is NOT guaranteed**:
- Depends on pair correlation
- Depends on regime stability
- Negative results are still possible and valid

## Limitations

### What This Does NOT Fix

1. **Fundamental lack of cointegration**: If pairs aren't cointegrated, portfolio won't help
2. **Systematic OU collapse**: If all pairs collapse, diversification doesn't help
3. **Persistent volatile regimes**: Binary gate will block most trades
4. **High transaction costs**: Still erode returns at portfolio level

### What This DOES Improve

1. **Idiosyncratic noise**: Diversification reduces pair-specific noise
2. **Signal-to-noise ratio**: Cross-sectional averaging improves quality
3. **Stability**: Less dependent on single pair performance
4. **Model validity**: Binary gating and time-to-reversion checks enforce assumptions

## Summary

**Portfolio mode implements**:
1. Cross-sectional diversification (all qualifying pairs)
2. Risk-based allocation (not performance-based)
3. Binary regime gating (enforces stationarity)
4. Time-to-reversion checks (enforces model validity)

**These are NOT optimizations**:
- They enforce statistical assumptions
- They implement stat-arb as it should be done
- They are structurally defensible
- Negative results remain acceptable

**Expected outcome**:
- Higher Sharpe than single-pair (from diversification)
- More stable performance (from averaging)
- Better model validity (from consistency checks)
- Still honest research (negative results possible)

**Use portfolio mode when**: You have 3+ qualifying pairs  
**Use single-pair mode when**: You have < 3 qualifying pairs

Both modes are valid research approaches.
