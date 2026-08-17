import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deployment_checks import deploy_gate_table, factor_capacity_proxy, turnover_from_weights


def test_turnover_capacity_and_gates():
    weights = pd.DataFrame({"PC1": [0.1, 0.2, 0.15], "PC2": [-0.1, -0.05, 0.0]}, index=["A", "B", "C"])
    turnover = turnover_from_weights(weights)
    capacity = factor_capacity_proxy(weights, pd.Series([1_000_000, 2_000_000, 1_500_000], index=["A", "B", "C"]))
    gates = deploy_gate_table(pd.DataFrame({"factor": ["PC1"], "sharpe_ci_05": [0.1], "max_drawdown": [-0.1], "hit_rate": [0.55]}))

    assert turnover.iloc[0] == 0
    assert capacity["capacity_proxy_notional"].notna().all()
    assert gates.iloc[0]["deploy_gate"] == "research_further"

