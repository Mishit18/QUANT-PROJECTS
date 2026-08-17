from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.robustness.deploy_gates import capacity_cost_grid, deploy_reject_gate


def main() -> None:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    grid = capacity_cost_grid(2.4, 2.1, [1_000_000, 5_000_000, 10_000_000], [1, 3, 5, 8])
    gate = deploy_reject_gate(ic_mean=0.004, ic_ir=0.18, turnover=2.1, net_return=-0.0824)
    grid.to_csv(reports / "capacity_cost_grid.csv", index=False)
    pd.DataFrame([gate]).to_csv(reports / "deploy_reject_gate.csv", index=False)
    (reports / "MODERN_DEPLOY_GATE_PACK.md").write_text(
        "\n".join(
            [
                "# Cross-Asset Deploy Gate Pack",
                "",
                f"- Verdict: {gate['verdict']}",
                f"- Reasons: {', '.join(gate['reasons'])}",
                f"- Capacity/cost scenarios: {len(grid)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

