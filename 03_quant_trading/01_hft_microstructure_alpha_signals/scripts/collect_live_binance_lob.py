from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.binance_live import collect


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect live Binance top-20 order-book snapshots.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--snapshots", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=Path("reports/live_binance_lob_snapshots.csv"))
    parser.add_argument("--summary", type=Path, default=Path("reports/live_binance_lob_summary.json"))
    args = parser.parse_args()
    if args.snapshots <= 0:
        parser.error("snapshots must be positive")
    report = asyncio.run(
        collect(
            symbol=args.symbol,
            snapshots=args.snapshots,
            output_csv=args.output,
            output_summary=args.summary,
        )
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
