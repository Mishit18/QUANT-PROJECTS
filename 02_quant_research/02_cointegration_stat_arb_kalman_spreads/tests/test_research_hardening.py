import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_hardening import borrow_short_constraint, paper_trade_readiness, rolling_stability_score


def test_cointegration_research_hardening():
    stability = rolling_stability_score(pd.Series([0.01, 0.04, 0.20, 0.03]), pd.Series([12, 20, 8, 70]))
    readiness = paper_trade_readiness(stability, min_stable_share=0.25)
    constrained = borrow_short_constraint(pd.Series([1, -1, -1]), pd.Series([True, False, True]), pd.Series([0, 50, 25]))

    assert "rolling_stability_6m" in stability.columns
    assert readiness["verdict"] in {"paper_trade", "research_only"}
    assert constrained["tradable_signal"].iloc[1] == 0

