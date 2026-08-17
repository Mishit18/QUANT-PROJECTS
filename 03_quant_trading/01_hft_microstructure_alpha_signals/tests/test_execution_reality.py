from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.analysis.execution_reality import adverse_selection_report, latency_sensitivity_table, queue_fill_probability


def test_queue_fill_probability_is_bounded():
    prob = queue_fill_probability(queue_ahead=100, trade_through_rate=4, cancel_rate=2, horizon_events=20)

    assert 0 <= prob <= 1
    assert queue_fill_probability(0, 1, 1, 1) == 1.0


def test_latency_sensitivity_table_decays_edge():
    table = latency_sensitivity_table(pd.DataFrame({"expected_edge_bps": [3.0, 2.5, 2.0]}))

    assert table.iloc[0]["gross_edge_bps"] > table.iloc[-1]["gross_edge_bps"]
    assert "tradable" in table.columns


def test_adverse_selection_report_outputs_microstructure_metrics():
    trades = pd.DataFrame(
        {
            "side": ["buy", "sell", "buy"],
            "fill_price": [100.0, 101.0, 99.5],
            "mid_after_fill": [100.02, 100.98, 99.45],
        }
    )
    report = adverse_selection_report(trades)

    assert report["trades"] == 3
    assert 0 <= report["toxic_fill_share"] <= 1

