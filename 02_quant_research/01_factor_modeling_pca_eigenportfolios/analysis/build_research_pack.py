from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
PLOTS = ROOT / "plots"


def annualized_sharpe(series: pd.Series) -> float:
    series = series.dropna()
    if series.empty or series.std() == 0:
        return 0.0
    return float(np.sqrt(252) * series.mean() / series.std())


def max_drawdown(series: pd.Series) -> float:
    equity = (1 + series.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min())


def bootstrap_sharpe_ci(returns: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for col in returns.columns:
        x = returns[col].dropna().to_numpy()
        if len(x) == 0:
            continue
        stats = []
        for _ in range(n_boot):
            sample = rng.choice(x, size=len(x), replace=True)
            sd = sample.std(ddof=1)
            stats.append(0.0 if sd == 0 else np.sqrt(252) * sample.mean() / sd)
        rows.append(
            {
                "factor": col,
                "sharpe": annualized_sharpe(returns[col]),
                "sharpe_ci_05": float(np.percentile(stats, 5)),
                "sharpe_ci_95": float(np.percentile(stats, 95)),
                "max_drawdown": max_drawdown(returns[col]),
                "hit_rate": float((returns[col] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def concentration_table() -> pd.DataFrame:
    weights = pd.read_csv(RESULTS / "eigen_portfolios.csv")
    factor_cols = [c for c in weights.columns if c.startswith("PC")]
    rows = []
    for col in factor_cols:
        abs_w = weights[col].abs()
        rows.append(
            {
                "factor": col,
                "top5_abs_weight_share": float(abs_w.sort_values(ascending=False).head(5).sum() / abs_w.sum()),
                "max_abs_weight": float(abs_w.max()),
                "gross_leverage_proxy": float(abs_w.sum()),
                "long_count": int((weights[col] > 0).sum()),
                "short_count": int((weights[col] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_plots(ci: pd.DataFrame) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ci = ci.sort_values("sharpe")
    err_low = ci["sharpe"] - ci["sharpe_ci_05"]
    err_high = ci["sharpe_ci_95"] - ci["sharpe"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(ci["sharpe"], ci["factor"], xerr=[err_low, err_high], fmt="o", color="#2563eb", ecolor="#94a3b8")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(1, color="#059669", linestyle="--", linewidth=1, label="Sharpe 1.0")
    ax.set_title("Bootstrap Sharpe Confidence Intervals")
    ax.set_xlabel("Annualized Sharpe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "bootstrap_sharpe_intervals.png", dpi=180)
    plt.close(fig)


def write_decision_memo(ci: pd.DataFrame, concentration: pd.DataFrame) -> None:
    ci = ci.copy()
    ci["decision"] = np.where(
        (ci["sharpe_ci_05"] > 0) & (ci["max_drawdown"] > -0.25),
        "research further",
        "do not deploy",
    )
    decision = ci.merge(concentration, on="factor", how="left")
    decision.to_csv(RESULTS / "factor_decision_gates.csv", index=False)

    top = decision.sort_values("sharpe", ascending=False).head(5)
    text = f"""# Factor Research Decision Memo

## Research Question

Do PCA eigen-portfolios or classical factors produce enough evidence to be treated as deployable alpha factors?

## Answer

No factor should be presented as production-ready alpha from this evidence alone. The project is strongest as a risk-decomposition and factor-research workflow. Factors with positive bootstrap lower bounds can be researched further, but still require walk-forward construction, turnover, costs, borrow constraints, and capacity checks.

## Top Factors By Sharpe

| Factor | Sharpe | 5% CI | 95% CI | Max Drawdown | Hit Rate | Decision |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(f"| {r.factor} | {r.sharpe:.2f} | {r.sharpe_ci_05:.2f} | {r.sharpe_ci_95:.2f} | {r.max_drawdown:.2%} | {r.hit_rate:.2%} | {r.decision} |" for r in top.itertuples())}

## Concentration Checks

| Factor | Top-5 Weight Share | Max Abs Weight | Long Count | Short Count |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {r.factor} | {r.top5_abs_weight_share:.2%} | {r.max_abs_weight:.2%} | {r.long_count} | {r.short_count} |" for r in concentration.head(10).itertuples())}

## Interview Framing

PCA explains covariance, not alpha. A strong PCA component can be useful for hedging, stress testing, and risk attribution even if its realized return is negative. Classical factors are more interpretable, but the strong in-sample quality result must not be overclaimed without walk-forward validation.

## Added Artifacts

- `results/factor_bootstrap_confidence_intervals.csv`
- `results/eigenportfolio_concentration.csv`
- `results/factor_decision_gates.csv`
- `plots/bootstrap_sharpe_intervals.png`
"""
    (REPORTS / "FACTOR_DECISION_MEMO.md").write_text(text, encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pca = pd.read_csv(RESULTS / "pca_factor_returns.csv", parse_dates=["Date"]).set_index("Date")
    classical = pd.read_csv(RESULTS / "classical_factor_returns.csv", parse_dates=["Date"]).set_index("Date")
    combined = pd.concat([pca, classical], axis=1)
    ci = bootstrap_sharpe_ci(combined)
    concentration = concentration_table()
    ci.to_csv(RESULTS / "factor_bootstrap_confidence_intervals.csv", index=False)
    concentration.to_csv(RESULTS / "eigenportfolio_concentration.csv", index=False)
    write_plots(ci)
    write_decision_memo(ci, concentration)
    print("Wrote reports/FACTOR_DECISION_MEMO.md")
    print("Wrote factor confidence/concentration outputs")


if __name__ == "__main__":
    main()
