# Statistical Arbitrage Research Framework

Research implementation of cointegration-based statistical arbitrage with Engle-Granger pair screening, Kalman hedge ratios, Ornstein-Uhlenbeck spread modeling, HMM regime gating, execution costs, portfolio aggregation, and transparent negative-result reporting.

This project is strongest as a quant research artifact because it does not cherry-pick winners. It is designed to reject weak signals, report failure modes, and show that a candidate understands statistical rigor beyond attractive backtest charts.

## What This Demonstrates

- Screens 780 equity pairs across a 40-stock universe.
- Applies Engle-Granger cointegration testing and half-life categorization.
- Estimates dynamic hedge ratios with a Kalman filter.
- Fits OU spread dynamics with R-squared and half-life quality gates.
- Uses HMM regime detection as defensive risk gating, not signal generation.
- Builds an equal-risk multi-pair portfolio rather than selecting pairs by performance.
- Applies transaction costs and slippage.
- Runs rolling-window and Monte Carlo diagnostics.
- Saves reproducible result tables and plots.

## Verified Run

Command:

```bash
python main.py
```

Verification:

```bash
python -m pytest -q
# 2 passed
```

Key output from the verified run:

| Metric | Value |
| --- | ---: |
| Pairs tested | 780 |
| Valid cointegrated pairs | 27 |
| Rejection rate | 96.5% |
| Portfolio pairs selected | 5 |
| Allocation method | Equal risk |
| Target volatility | 10.0% |
| Volatility scalar | 0.15x |
| Total return | -25.25% |
| Annual return | -7.03% |
| Sharpe ratio | -0.678 |
| Sortino ratio | -0.215 |
| Max drawdown | -32.03% |
| Win rate | 23.9% |
| Rolling Sharpe mean | -1.79 |
| Probability of loss | 77.4% |
| Worst Monte Carlo case | -32.6% |

Interpretation: the framework identifies statistically plausible pairs, then shows that they still fail after dynamic hedge ratios, regime gating, and costs. This is a valid research outcome and a strong interview talking point.

## Selected Portfolio Pairs

| Pair | Cointegration p-value | Initial half-life | Category |
| --- | ---: | ---: | --- |
| BAC/PNC | 0.0016 | 18.2d | tradable_daily |
| WFC/MS | 0.0028 | 37.7d | tradable_daily |
| WFC/GS | 0.0044 | 39.5d | tradable_daily |
| C/NKE | 0.0057 | 40.4d | tradable_daily |
| C/TGT | 0.0059 | 38.8d | tradable_daily |

After Kalman filtering, OU half-lives collapse below one day for these pairs. The framework allows a reduced position multiplier but the final portfolio still fails its targets. That failure is documented instead of hidden.

## Statistical Pipeline

1. **Universe selection**: test all candidate pairs for cointegration.
2. **Spread construction**: estimate dynamic hedge ratios with Kalman filtering.
3. **OU validation**: estimate theta, half-life, equilibrium variance, and R-squared.
4. **Regime gating**: block new positions during volatile HMM regimes.
5. **Signal generation**: enter on extreme z-scores, exit near equilibrium or model invalidation.
6. **Execution modeling**: subtract transaction costs and slippage.
7. **Portfolio aggregation**: allocate equal risk across qualifying pairs.
8. **Diagnostics**: rolling-window Sharpe and Monte Carlo risk.

## Run Locally

```bash
pip install -r requirements.txt
python -m pytest -q
python main.py
```

Generated outputs:

- `results/tables/portfolio_summary.csv`
- `results/tables/pair_attribution.csv`
- `results/diagnostics/rolling_window_test.csv`
- `results/plots/portfolio_backtest.png`

The generated results are ignored by default to avoid committing run artifacts.

## Repository Structure

```text
04_stat_arb_trader_execution/
|-- src/
|   |-- universe_selection.py
|   |-- spread_model.py
|   |-- kalman.py
|   |-- regime_filter.py
|   |-- alpha_layer.py
|   |-- execution.py
|   |-- portfolio.py
|   `-- diagnostics.py
|-- tests/
|   |-- test_config.py
|   `-- README.md
|-- docs/
|   |-- portfolio.md
|   `-- ATS_SCREENING_PACK.md
|-- config.yaml
|-- main.py
|-- requirements.txt
`-- README.md
```

## Screening Positioning

Best fit:

- Quant Research
- Quant Trader
- Systematic Trading Research
- Statistical Arbitrage Research
- Portfolio Research

Do not overclaim:

- This is not a profitable strategy in the verified run.
- This is not production trading infrastructure.
- The project uses historical equity data and simplified cost assumptions.
- Negative results are part of the research design, not a bug.

## Why This Is Resume-Strong

Many student quant projects only show a high Sharpe. This one shows the full research discipline: hypothesis, statistical testing, model validation, cost modeling, portfolio construction, diagnostics, and honest rejection. That is much closer to how serious quant teams evaluate research.
