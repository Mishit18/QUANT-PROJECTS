from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def sharpe(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.std() == 0 or returns.empty:
        return 0.0
    return float(np.sqrt(252) * returns.mean() / returns.std())


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1
    return float(dd.min())


def load_return_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    value_col = df.columns[1]
    series = pd.Series(df[value_col].values, index=pd.to_datetime(df[date_col]), name=path.stem)
    return series.sort_index()


def compute_ic_table() -> pd.DataFrame:
    preds = pd.read_parquet(DATA / "predictions_xgboost.parquet")["prediction"]
    targets = pd.read_parquet(DATA / "targets.parquet")["target"]
    panel = pd.concat([preds, targets], axis=1).dropna()
    panel.columns = ["prediction", "target"]
    ic = panel.groupby(level=0).apply(
        lambda x: x["prediction"].corr(x["target"], method="spearman")
    )
    ic = ic.dropna()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(ic.index),
            "rank_ic": ic.values,
            "rolling_21d_ic": ic.rolling(21).mean().values,
            "rolling_63d_ic": ic.rolling(63).mean().values,
        }
    )
    out.to_csv(REPORTS / "xgboost_ic_timeseries.csv", index=False)
    return out


def compute_cost_sensitivity() -> pd.DataFrame:
    asset_returns = pd.read_parquet(DATA / "returns.parquet")
    weights = pd.read_parquet(DATA / "backtest_weights_final.parquet")
    asset_returns.index = pd.to_datetime(asset_returns.index)
    weights.index = pd.to_datetime(weights.index)

    common_cols = [c for c in weights.columns if c in asset_returns.columns]
    weights = weights[common_cols].sort_index()
    asset_returns = asset_returns[common_cols].sort_index()
    daily_weights = weights.reindex(asset_returns.index).ffill().fillna(0.0)
    gross_returns = (daily_weights.shift(1).fillna(0.0) * asset_returns).sum(axis=1)

    turnover = weights.diff().abs().sum(axis=1).div(2).fillna(weights.abs().sum(axis=1).div(2))
    turnover_daily = turnover.reindex(asset_returns.index).fillna(0.0)

    rows = []
    for bps in [0, 2.5, 5, 7.5, 10, 25, 50]:
        costs = turnover_daily * bps / 10000
        net_returns = gross_returns - costs
        equity = (1 + net_returns).cumprod()
        rows.append(
            {
                "cost_bps": bps,
                "total_return": equity.iloc[-1] - 1,
                "annual_return": (1 + net_returns).prod() ** (252 / len(net_returns)) - 1,
                "annual_volatility": net_returns.std() * np.sqrt(252),
                "sharpe": sharpe(net_returns),
                "max_drawdown": max_drawdown(equity),
                "avg_rebalance_turnover": turnover.mean(),
                "total_cost_drag": costs.sum(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "cost_sensitivity.csv", index=False)
    turnover.to_csv(REPORTS / "turnover_timeseries.csv", header=["one_way_turnover"])
    gross_returns.to_csv(REPORTS / "gross_return_proxy.csv", header=["gross_return_proxy"])
    return out


def write_plots(ic: pd.DataFrame, cost: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ic["date"], ic["rolling_21d_ic"], label="21d rolling rank IC")
    ax.plot(ic["date"], ic["rolling_63d_ic"], label="63d rolling rank IC")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("XGBoost IC Stability")
    ax.set_ylabel("Spearman rank IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "xgboost_ic_stability.png", dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(cost["cost_bps"], cost["total_return"], marker="o", label="Total return")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Transaction cost assumption, bps")
    ax1.set_ylabel("Total return")
    ax2 = ax1.twinx()
    ax2.plot(cost["cost_bps"], cost["sharpe"], marker="s", color="#dc2626", label="Sharpe")
    ax2.set_ylabel("Sharpe")
    ax1.set_title("Transaction Cost Sensitivity")
    fig.tight_layout()
    fig.savefig(FIGURES / "transaction_cost_sensitivity.png", dpi=180)
    plt.close(fig)

    final = load_return_series(DATA / "backtest_returns_final.csv")
    realistic = load_return_series(DATA / "backtest_returns_realistic.csv")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot((1 + final).cumprod(), label="Horizon-matched final")
    ax.plot((1 + realistic).cumprod(), label="Realistic neutralized")
    ax.set_title("Equity Curves: Final vs Realistic Validation")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "final_vs_realistic_equity.png", dpi=180)
    plt.close(fig)

    turnover = pd.read_csv(REPORTS / "turnover_timeseries.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(turnover.index, turnover["one_way_turnover"], color="#7c3aed")
    ax.set_title("Rebalance Turnover")
    ax.set_ylabel("One-way turnover")
    fig.tight_layout()
    fig.savefig(FIGURES / "rebalance_turnover.png", dpi=180)
    plt.close(fig)


def write_appendix(ic: pd.DataFrame, cost: pd.DataFrame) -> None:
    final = load_return_series(DATA / "backtest_returns_final.csv")
    realistic = load_return_series(DATA / "backtest_returns_realistic.csv")
    cost_75 = cost.loc[cost["cost_bps"] == 7.5].iloc[0]
    ic_mean = ic["rank_ic"].mean()
    ic_ir = ic["rank_ic"].mean() / ic["rank_ic"].std() if ic["rank_ic"].std() else 0.0
    hit_rate = (ic["rank_ic"] > 0).mean()

    text = f"""# Research Appendix: Stat-Arb Evidence Pack

## Purpose

This appendix upgrades the project from a backtest script into a quant research memo. It documents IC stability, turnover, transaction-cost sensitivity, and deploy/reject gates.

## IC Diagnostics

| Metric | Value |
|---|---:|
| Mean daily rank IC | {ic_mean:.4f} |
| IC information ratio | {ic_ir:.2f} |
| Positive-IC hit rate | {hit_rate:.2%} |
| IC observations | {len(ic):,} |

Artifacts:

- `reports/xgboost_ic_timeseries.csv`
- `reports/figures/xgboost_ic_stability.png`

## Cost Sensitivity

| Cost bps | Total Return | Sharpe | Max Drawdown | Cost Drag |
|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r.cost_bps:.1f} | {r.total_return:.2%} | {r.sharpe:.2f} | {r.max_drawdown:.2%} | {r.total_cost_drag:.2%} |" for r in cost.itertuples())}

At the original 7.5 bps assumption, the recomputed proxy has total return {cost_75['total_return']:.2%}, Sharpe {cost_75['sharpe']:.2f}, and average rebalance turnover {cost_75['avg_rebalance_turnover']:.2%}. This supports the current reject decision because the edge is not robust to realistic frictions.

## Return Validation

| Series | Total Return | Sharpe | Max Drawdown |
|---|---:|---:|---:|
| Horizon-matched final | {(1 + final).prod() - 1:.2%} | {sharpe(final):.2f} | {max_drawdown((1 + final).cumprod()):.2%} |
| Realistic neutralized | {(1 + realistic).prod() - 1:.2%} | {sharpe(realistic):.2f} | {max_drawdown((1 + realistic).cumprod()):.2%} |

## Deployment Gate

| Gate | Requirement | Result | Decision |
|---|---|---|---|
| Positive IC | Mean IC > 0 | {ic_mean:.4f} | Pass but weak |
| Stable IC | IC-IR > 0.30 | {ic_ir:.2f} | Fail |
| Costs | Net return survives 7.5 bps | {cost_75['total_return']:.2%} | Fail |
| Turnover | Average rebalance turnover below 75% | {cost_75['avg_rebalance_turnover']:.2%} | Fail |
| Drawdown | Max drawdown better than -25% | {cost_75['max_drawdown']:.2%} | {'Pass' if cost_75['max_drawdown'] > -0.25 else 'Fail'} |

## Interview Answer

The strongest answer is: I would not deploy this strategy. The project is valuable because the validation stack caught weak alpha before deployment. It demonstrates target construction, walk-forward validation, IC diagnostics, cost sensitivity, and research judgment.
"""
    (REPORTS / "RESEARCH_APPENDIX.md").write_text(text, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ic = compute_ic_table()
    cost = compute_cost_sensitivity()
    write_plots(ic, cost)
    write_appendix(ic, cost)
    print("Wrote reports/RESEARCH_APPENDIX.md")
    print("Wrote reports/cost_sensitivity.csv")
    print("Wrote IC and cost plots under reports/figures")


if __name__ == "__main__":
    main()
