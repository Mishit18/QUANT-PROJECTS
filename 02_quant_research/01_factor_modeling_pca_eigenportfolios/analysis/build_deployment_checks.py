from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deployment_checks import deploy_gate_table, factor_capacity_proxy, turnover_from_weights


def main() -> None:
    results = ROOT / "results"
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    perf = pd.read_csv(results / "factor_bootstrap_confidence_intervals.csv")
    gates = deploy_gate_table(perf)
    gates.to_csv(results / "factor_deploy_gates_modern.csv", index=False)
    weights = pd.read_csv(results / "eigen_portfolios.csv")
    if "ticker" in weights.columns:
        weights = weights.set_index("ticker")
    adv = pd.Series(np.linspace(1_000_000, 5_000_000, len(weights)), index=weights.index)
    capacity = factor_capacity_proxy(weights, adv)
    capacity.to_csv(results / "factor_capacity_proxy.csv", index=False)
    turnover_from_weights(weights).to_csv(results / "factor_turnover_proxy.csv", header=["turnover"])
    (reports / "MODERN_DEPLOYMENT_CHECKS.md").write_text(
        "\n".join(
            [
                "# Factor Deployment Checks",
                "",
                f"- Factors checked: {len(gates)}",
                f"- Research-further gates: {int((gates['deploy_gate'] == 'research_further').sum())}",
                "- Added turnover proxy, capacity proxy, and deploy/reject gates.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

