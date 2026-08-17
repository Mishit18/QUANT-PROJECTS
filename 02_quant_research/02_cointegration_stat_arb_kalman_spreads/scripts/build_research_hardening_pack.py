from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_hardening import borrow_short_constraint, paper_trade_readiness, rolling_stability_score


def main() -> None:
    out = ROOT / "results" / "modern_hardening"
    reports = ROOT / "reports"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=18, freq="ME")
    pvals = pd.Series([0.02, 0.03, 0.08, 0.04, 0.01, 0.20, 0.03, 0.02, 0.04, 0.06, 0.03, 0.02, 0.01, 0.07, 0.04, 0.02, 0.03, 0.05], index=dates)
    half_life = pd.Series(np.clip(np.linspace(8, 42, len(dates)) + np.random.default_rng(7).normal(0, 4, len(dates)), 1, 80), index=dates)
    stability = rolling_stability_score(pvals, half_life)
    constrained = borrow_short_constraint(pd.Series([1, -1, -1, 1], index=range(4)), pd.Series([True, False, True, True], index=range(4)), pd.Series([0, 60, 30, 0], index=range(4)))
    readiness = paper_trade_readiness(stability)
    stability.to_csv(out / "rolling_cointegration_stability.csv")
    constrained.to_csv(out / "borrow_constrained_signals.csv")
    pd.DataFrame([readiness]).to_csv(out / "paper_trade_readiness.csv", index=False)
    (reports / "MODERN_COINTEGRATION_HARDENING.md").write_text(
        "\n".join(
            [
                "# Cointegration Hardening Pack",
                "",
                f"- Stable-window share: {readiness['stable_window_share']:.2%}",
                f"- Recent stability: {readiness['recent_stability']:.2%}",
                f"- Verdict: {readiness['verdict']}",
                "- Added rolling stability, borrow constraints, and paper-trade readiness checks.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

