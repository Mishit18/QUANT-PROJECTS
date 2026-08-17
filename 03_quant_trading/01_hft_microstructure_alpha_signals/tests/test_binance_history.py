from pathlib import Path
from zipfile import ZipFile

from src.data.binance_history import build_trade_features, load_agg_trades


def test_load_and_build_historical_features(tmp_path: Path) -> None:
    archive = tmp_path / "trades.zip"
    rows = []
    start = 1_700_000_000_000
    for index in range(80):
        rows.append(f"{index},{100 + index * 0.01},1,{index},{index},{start + index * 1000},{index % 2 == 0},True\n")
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("trades.csv", "".join(rows))
    trades = load_agg_trades(archive)
    features = build_trade_features(trades, horizon_seconds=5)
    assert len(trades) == 80
    assert len(features) > 30
    assert "volume_imbalance_30s" in features.columns
    assert features.index.is_monotonic_increasing
