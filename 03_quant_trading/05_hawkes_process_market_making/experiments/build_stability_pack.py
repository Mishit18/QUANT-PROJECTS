from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hawkes.stability import calibration_stability_grid, hawkes_failure_modes


def main() -> None:
    results = ROOT / "results"
    report = ROOT / "report"
    results.mkdir(exist_ok=True)
    report.mkdir(exist_ok=True)
    base = np.array([[0.25, 0.08], [0.05, 0.22]])
    samples = [base * m for m in [0.8, 0.95, 1.1, 1.25, 0.9]]
    grid = calibration_stability_grid(samples)
    modes = hawkes_failure_modes(grid, pd.Series([0.30, 0.12, 0.08, 0.03, 0.20]))
    grid.to_csv(results / "hawkes_calibration_stability.csv", index=False)
    pd.DataFrame([modes]).to_csv(results / "hawkes_failure_modes.csv", index=False)
    (report / "HAWKES_STABILITY_PACK.md").write_text(
        "\n".join(
            [
                "# Hawkes Stability Pack",
                "",
                f"- Unstable-window share: {modes['unstable_window_share']:.2%}",
                f"- Poor residual share: {modes['poor_residual_share']:.2%}",
                f"- Verdict: {modes['verdict']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
