from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
REPORT = ROOT / "report"


def fmt(x: float) -> str:
    return f"{x:,.2f}"


def build_summary_tables(bench: pd.DataFrame, stress: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bench = bench.copy()
    twap_cost = float(bench.loc[bench["strategy"] == "TWAP", "mean_cost"].iloc[0])
    bench["cost_improvement_vs_twap"] = twap_cost - bench["mean_cost"]
    bench["completion_gap_vs_100pct"] = 1.0 - bench["mean_completion"]
    bench = bench.sort_values("mean_cost")

    stress_summary = (
        stress.groupby("strategy", as_index=False)
        .agg(
            avg_stress_cost=("mean_cost", "mean"),
            worst_95_cost=("cost_95th", "max"),
            avg_completion=("mean_completion", "mean"),
            max_failure_rate=("failure_rate", "max"),
            avg_tail_risk=("tail_risk", "mean"),
        )
        .sort_values("avg_stress_cost")
    )
    return bench, stress_summary


def write_plots(bench: pd.DataFrame, stress: pd.DataFrame, stress_summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(bench["strategy"], bench["mean_cost"], color="#2563eb")
    ax.axhline(bench.loc[bench["strategy"] == "TWAP", "mean_cost"].iloc[0], color="black", linestyle="--", linewidth=1)
    ax.set_title("Execution Cost by Strategy")
    ax.set_ylabel("Mean simulator cost")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "execution_cost_by_strategy.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.scatter(bench["mean_completion"], bench["mean_cost"], s=90, color="#059669")
    for _, row in bench.iterrows():
        ax.annotate(row["strategy"], (row["mean_completion"], row["mean_cost"]), textcoords="offset points", xytext=(5, 4))
    ax.set_title("Completion vs Cost Frontier")
    ax.set_xlabel("Mean completion rate")
    ax.set_ylabel("Mean simulator cost")
    fig.tight_layout()
    fig.savefig(FIGURES / "completion_cost_frontier.png", dpi=180)
    plt.close(fig)

    pivot = stress.pivot_table(index="scenario", columns="strategy", values="mean_cost")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Stress-Test Mean Cost by Scenario")
    ax.set_ylabel("Mean simulator cost")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "stress_cost_by_scenario.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(stress_summary["strategy"], stress_summary["avg_tail_risk"], color="#dc2626")
    ax.set_title("Average Tail Risk Across Stress Tests")
    ax.set_ylabel("95th percentile cost minus mean cost")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "stress_tail_risk.png", dpi=180)
    plt.close(fig)


def write_report(bench: pd.DataFrame, stress_summary: pd.DataFrame) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    best_cost = bench.iloc[0]
    best_completion = bench.sort_values("mean_completion", ascending=False).iloc[0]
    best_stress = stress_summary.iloc[0]
    twap = bench.loc[bench["strategy"] == "TWAP"].iloc[0]

    text = f"""# Execution Analytics Pack

## Desk-Style Question

Which execution policy should be preferred under stochastic liquidity, volatility, and impact assumptions, and where does it fail?

## Benchmark Ranking

| Strategy | Mean Cost | Std Cost | Completion | Cost Improvement vs TWAP |
|---|---:|---:|---:|---:|
{chr(10).join(f"| {r.strategy} | {fmt(r.mean_cost)} | {fmt(r.std_cost)} | {r.mean_completion:.2%} | {fmt(r.cost_improvement_vs_twap)} |" for r in bench.itertuples())}

## Stress Ranking

| Strategy | Avg Stress Cost | Worst 95th Cost | Avg Completion | Max Failure Rate | Avg Tail Risk |
|---|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r.strategy} | {fmt(r.avg_stress_cost)} | {fmt(r.worst_95_cost)} | {r.avg_completion:.2%} | {r.max_failure_rate:.2%} | {fmt(r.avg_tail_risk)} |" for r in stress_summary.itertuples())}

## Decision

- Best mean-cost strategy in the simulator: {best_cost['strategy']} with mean cost {fmt(best_cost['mean_cost'])}.
- Best completion strategy: {best_completion['strategy']} with completion {best_completion['mean_completion']:.2%}.
- Best average stress-cost strategy: {best_stress['strategy']} with average stress cost {fmt(best_stress['avg_stress_cost'])}.
- TWAP baseline cost: {fmt(twap['mean_cost'])}; this is the benchmark any adaptive policy must beat.

## Caveat

The simulator cashflow convention allows negative "cost" values, so these results should be described as relative simulator performance, not live trading PnL. The correct interview framing is that the project builds an execution research harness with baselines, offline RL policies, stress tests, and explicit limitations.

## Interview Defense

**Why compare to Almgren-Chriss?** It is the analytical baseline for optimal liquidation under temporary/permanent impact assumptions.

**Why use offline RL?** Execution is a sequential decision problem where liquidity and impact states change over the schedule; offline RL can learn adaptive policies from simulated trajectories without online market risk.

**How would this go production?** Replace simulated liquidity with historical L2/L3 replay, calibrate impact parameters by symbol and participation rate, enforce kill-switches, and validate against implementation shortfall after fees and market impact.

## Artifacts

- `results/figures/execution_cost_by_strategy.png`
- `results/figures/completion_cost_frontier.png`
- `results/figures/stress_cost_by_scenario.png`
- `results/figures/stress_tail_risk.png`
"""
    (REPORT / "EXECUTION_ANALYTICS_PACK.md").write_text(text, encoding="utf-8")
    bench.to_csv(RESULTS / "benchmark_ranked.csv", index=False)
    stress_summary.to_csv(RESULTS / "stress_summary_by_strategy.csv", index=False)


def main() -> None:
    bench = pd.read_csv(RESULTS / "benchmark_results.csv")
    stress = pd.read_csv(RESULTS / "stress_test_results.csv")
    ranked, stress_summary = build_summary_tables(bench, stress)
    write_plots(ranked, stress, stress_summary)
    write_report(ranked, stress_summary)
    print("Wrote report/EXECUTION_ANALYTICS_PACK.md")
    print("Wrote results/benchmark_ranked.csv and results/stress_summary_by_strategy.csv")
    print("Wrote execution plots under results/figures")


if __name__ == "__main__":
    main()
