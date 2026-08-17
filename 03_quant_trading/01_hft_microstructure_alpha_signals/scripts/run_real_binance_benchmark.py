from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.binance_history import (
    build_trade_features,
    download_agg_trades,
    load_agg_trades,
    run_walk_forward_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a real Binance historical trade benchmark.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", default="2026-08-15")
    args = parser.parse_args()
    archive = download_agg_trades(args.symbol, args.date, ROOT / ".cache" / "binance")
    trades = load_agg_trades(archive)
    features = build_trade_features(trades)
    metrics = run_walk_forward_benchmark(features, ROOT / "reports")
    print(f"Loaded {len(trades):,} real aggregate trades and {len(features):,} one-second feature bars")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
