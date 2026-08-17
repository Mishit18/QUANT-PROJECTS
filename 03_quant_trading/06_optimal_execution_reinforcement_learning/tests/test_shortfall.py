from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis.shortfall import participation_guardrail, shortfall_decomposition


def test_shortfall_decomposition_adds_to_total():
    fills = pd.DataFrame(
        {
            "price": [100.01, 100.03, 100.04],
            "mid_price": [100.00, 100.02, 100.03],
            "quantity": [100, 150, 250],
        }
    )
    out = shortfall_decomposition(fills, arrival_price=100.0, side="buy")
    reconstructed = out["spread_cost_bps"] + out["market_drift_bps"] + out["timing_residual_bps"]

    assert abs(reconstructed - out["implementation_shortfall_bps"]) < 1e-9
    assert out["total_quantity"] == 500


def test_participation_guardrail_flags_breaches():
    fills = pd.DataFrame({"quantity": [10, 50], "market_volume": [1000, 100]})
    out = participation_guardrail(fills, max_participation=0.15)

    assert out["participation_breach"].tolist() == [False, True]
