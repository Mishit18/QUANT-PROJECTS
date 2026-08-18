from copy import deepcopy

from src.forward.paper_ledger import build_paper_ledger, verify_ledger


def _snapshots():
    return [
        {"received_at": "2026-08-18T00:00:00Z", "midpoint": 100.0, "depth_imbalance": 0.4, "microprice_deviation_bps": 0.5},
        {"received_at": "2026-08-18T00:00:01Z", "midpoint": 101.0, "depth_imbalance": -0.4, "microprice_deviation_bps": -0.5},
        {"received_at": "2026-08-18T00:00:02Z", "midpoint": 100.0, "depth_imbalance": 0.0, "microprice_deviation_bps": 0.0},
    ]


def test_ledger_is_hash_chained_and_reports_costs():
    records, summary = build_paper_ledger(_snapshots(), cost_bps=1.0)
    assert verify_ledger(records)
    assert summary["records"] == 2
    assert summary["turnover_units"] == 3
    assert summary["transaction_cost"] > 0
    assert summary["execution_boundary"].startswith("no orders submitted")


def test_tampering_breaks_verification():
    records, _ = build_paper_ledger(_snapshots())
    tampered = deepcopy(records)
    tampered[0]["net_return"] = 99
    assert not verify_ledger(tampered)
