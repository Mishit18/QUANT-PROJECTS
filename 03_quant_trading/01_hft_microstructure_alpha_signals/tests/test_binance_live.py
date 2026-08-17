from src.data.binance_live import depth_features, summarize


def test_depth_features() -> None:
    payload = {
        "lastUpdateId": 123,
        "bids": [["100.0", "2.0"], ["99.0", "1.0"]],
        "asks": [["101.0", "1.0"], ["102.0", "2.0"]],
    }
    row = depth_features(payload, "2026-08-17T00:00:00+00:00")
    assert row["midpoint"] == 100.5
    assert row["bid_depth_20"] == 3.0
    assert row["ask_depth_20"] == 3.0
    assert row["depth_imbalance"] == 0.0
    assert float(row["microprice"]) > float(row["midpoint"])


def test_summary_is_explicit_about_scope() -> None:
    rows = [
        {"spread_bps": 1.0, "depth_imbalance": 0.2, "microprice_deviation_bps": 0.1},
        {"spread_bps": 2.0, "depth_imbalance": -0.2, "microprice_deviation_bps": -0.1},
    ]
    report = summarize(rows, elapsed_seconds=2.0, trades=4)
    assert report["depth_snapshots"] == 2
    assert report["snapshot_rate_per_second"] == 1.0
    assert "feature validation only" in report["limitation"]
