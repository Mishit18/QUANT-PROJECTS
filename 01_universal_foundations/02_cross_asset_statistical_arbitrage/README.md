# Cross-Asset Statistical Arbitrage

Cross-sectional alpha research with point-in-time volatility-scaled targets, embargoed walk-forward validation, risk neutralization, OLS/Ridge/XGBoost modeling, and execution diagnostics.

## Research Objective

Test whether volatility-scaled forward returns produce a better-conditioned cross-sectional equity alpha target, then verify whether any resulting IC survives neutralization, turnover, transaction costs, and horizon-matched execution.

This is a research framework, not a live trading strategy. Current results use synthetic data only.

## Current Verdict

**Signal status: rejected for deployment.**

After correcting target construction and adding an embargo to the walk-forward split, the main XGBoost signal has near-zero IC:

| Model | Mean IC | IC-IR | Hit Rate | N |
|---|---:|---:|---:|---:|
| OLS | -0.0043 | -0.04 | 48.46% | 2342 |
| Ridge | -0.0044 | -0.04 | 48.12% | 2342 |
| XGBoost | 0.0051 | 0.05 | 52.41% | 2343 |

The fixed-signal risk check also says the baseline XGBoost IC is not statistically meaningful:

| Target Variant | Mean IC | IC-IR | Hit Rate | t-stat | p-value |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0051 | 0.05 | 52.4% | 2.19 | 0.0288 |
| Market-neutral | 0.0046 | 0.04 | 52.0% | 1.95 | 0.0509 |
| Sector-neutral | 0.0059 | 0.05 | 51.9% | 2.47 | 0.0135 |
| Market + Sector | 0.0055 | 0.05 | 52.0% | 2.31 | 0.0208 |

## Key Research Fixes

1. **Point-in-time target scaling**

   Volatility-scaled targets are now:

   ```text
   target[t] = forward_return[t, t+h] / (trailing_daily_vol[t] * sqrt(h))
   ```

   The previous implementation scaled by rolling volatility of the forward-return target itself. That was overly sparse after liquidity filtering and could contaminate target conditioning with future information.

2. **Embargoed walk-forward validation**

   The validator now drops the last `horizon` training dates before each test window. This avoids training on labels whose 5-day forward return overlaps the test period.

3. **Execution alignment**

   The generic backtest engine now uses next-period returns, charges opening turnover, tracks daily held weights between rebalance dates, and resets state on each run.

4. **Cost accounting**

   Transaction costs now use the union of previous/current positions, so entering or exiting a name is charged correctly.

5. **Faster risk neutralization**

   Sector neutralization is vectorized. The risk analysis script now decomposes the saved out-of-sample signal instead of retraining XGBoost repeatedly. Slow rolling PCA diagnostics are opt-in via `RUN_PCA_DIAGNOSTICS=1`.

6. **Reproducible synthetic data**

   Synthetic ticker generation now uses stable hash seeds and reads universe size/date range from `src/config/config.yaml`.

7. **Tests**

   A focused pytest suite covers the target denominator, cross-sectional target coverage, transaction-cost union accounting, and next-period backtest alignment.

## Target Diagnostics

With the corrected vol-scaled target:

| Metric | Value |
|---|---:|
| Horizon | 5 days |
| Vol window | 20 days |
| Mean cross-sectional target std | 1.0796 |
| Mean valid targets/date | 76.2 |
| Feature-target date alignment | 2605/2605 dates |

This is much healthier than the earlier sparse target panel and is the correct basis for evaluating the models.

## Execution Results

The most realistic test is the neutralized, costed portfolio:

| Scenario | Sharpe | Return | Max DD | Avg Turnover |
|---|---:|---:|---:|---:|
| Baseline, no neutralization, no costs | 0.14 | 8.37% | -23.71% | 138.88% |
| Market + sector neutral, no costs | 0.44 | 43.43% | -17.54% | 126.73% |
| Market + sector neutral, with costs | -0.04 | -8.24% | -25.17% | 126.73% |

Transaction costs consume the weak gross edge:

| Metric | Value |
|---|---:|
| Gross return | 43.43% |
| Net return | -8.24% |
| Total costs | 44.67% |
| Cost drag | 51.68% |
| Estimated capacity | $4.9M |

The horizon-matched signal-weighted diagnostic is still useful, but it is not an approval test because it skips neutralization and uses a more aggressive portfolio construction:

| Diagnostic | Daily Rebalance | 5-Day Rebalance |
|---|---:|---:|
| Sharpe | 0.06 | 1.29 |
| Total return | 0.50% | 213.42% |
| Max drawdown | -36.87% | -15.59% |
| Turnover | 97.25% | 126.69% |

Interpretation: horizon matching can matter materially, but the realistic neutralized/costed workflow is still the approval test and rejects this synthetic signal.

## Architecture

```text
src/
  backtest/        Portfolio construction, costs, walk-forward engine
  config/          YAML configuration
  data/            Loading, cleaning, panel building, target construction
  evaluation/      IC, decay, turnover, correlation diagnostics
  features/        Price, momentum, volatility, volume, cross-sectional features
  models/          OLS, Ridge/Lasso/ElasticNet, XGBoost models
  neutralization/  Market beta, sector, PCA residualization
  robustness/      Reality check, SPA, stress diagnostics
  utils/           Metrics, plotting, helpers

scripts/
  generate_sample_data.py
  run_features.py
  run_models.py
  run_backtest.py
  run_pipeline.py
  evaluate_targets.py
  analyze_ic_stability.py
  analyze_risk_neutralization.py
  realistic_backtest.py
  final_backtest.py

tests/
  test_backtest.py
  test_targets.py
```

## How To Run

```bash
python scripts/generate_sample_data.py
python scripts/run_pipeline.py
python scripts/analyze_ic_stability.py
python scripts/analyze_risk_neutralization.py
python scripts/realistic_backtest.py
python scripts/final_backtest.py
python -m pytest
```

Optional slow PCA risk diagnostics:

```bash
set RUN_PCA_DIAGNOSTICS=1
python scripts/analyze_risk_neutralization.py
```

## Configuration

Primary parameters live in `src/config/config.yaml`:

| Section | Current Setting |
|---|---|
| Target | `horizon=5`, `method=vol_scaled`, `vol_window=20` |
| Models | `train_window=252`, `test_window=21`, `retrain_freq=21`, `embargo=5` |
| Backtest | `rebalance_freq=5`, `vol_target=0.10`, `tcost_bps=7.5`, `slippage_bps=2.5` |
| Neutralization | market beta on, sector on, PCA factors configured but slow diagnostics opt-in |

## What This Project Demonstrates

- Correct point-in-time target engineering
- Embargoed walk-forward validation for overlapping forward labels
- Cross-sectional model comparison
- IC stability, t-stat, p-value, hit-rate diagnostics
- Market and sector neutralization
- Horizon-matched execution testing
- Transaction-cost and turnover awareness
- Honest rejection of a weak synthetic signal

## What It Does Not Claim

- No live trading approval
- No real-market alpha validation
- No institutional-capacity estimate
- No optimized portfolio construction
- No guarantee that the synthetic-data result transfers to real assets

## Next Research Step

Replace synthetic OHLCV with real survivorship-aware market data, keep the corrected target/embargo/backtest harness unchanged, and rerun the same acceptance gates. The current synthetic signal should remain frozen and rejected unless real-data evidence changes the conclusion.
