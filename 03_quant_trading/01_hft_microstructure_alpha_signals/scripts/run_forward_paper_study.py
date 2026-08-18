from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.binance_live import collect
from src.forward.paper_ledger import build_paper_ledger, verify_ledger, write_ledger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture public LOB data and create a hash-chained paper ledger")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--snapshots", type=int, default=1_000)
    parser.add_argument("--imbalance-threshold", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "forward_runs")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    snapshots_path = output_dir / "live_snapshots.csv"
    capture_summary_path = output_dir / "capture_summary.json"
    await collect(
        symbol=args.symbol,
        snapshots=args.snapshots,
        output_csv=snapshots_path,
        output_summary=capture_summary_path,
    )
    with snapshots_path.open(newline="", encoding="utf-8") as handle:
        snapshots = list(csv.DictReader(handle))
    records, summary = build_paper_ledger(
        snapshots,
        imbalance_threshold=args.imbalance_threshold,
        cost_bps=args.cost_bps,
    )
    if not verify_ledger(records):
        raise RuntimeError("ledger verification failed before write")
    ledger_path = output_dir / "paper_ledger.jsonl"
    write_ledger(records, ledger_path)
    summary["ledger_verified"] = True
    summary["snapshots_file"] = snapshots_path.name
    summary["ledger_file"] = ledger_path.name
    (output_dir / "paper_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

