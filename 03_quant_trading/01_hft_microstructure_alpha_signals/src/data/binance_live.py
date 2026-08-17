from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets


BINANCE_WS = "wss://data-stream.binance.vision"


def depth_features(payload: dict[str, Any], received_at: str) -> dict[str, float | str]:
    bids = [(float(price), float(quantity)) for price, quantity in payload["bids"]]
    asks = [(float(price), float(quantity)) for price, quantity in payload["asks"]]
    if not bids or not asks:
        raise ValueError("depth payload must include bids and asks")
    best_bid, best_bid_size = bids[0]
    best_ask, best_ask_size = asks[0]
    if best_ask <= best_bid:
        raise ValueError("crossed or locked order book snapshot")

    bid_depth = sum(quantity for _, quantity in bids)
    ask_depth = sum(quantity for _, quantity in asks)
    total_depth = bid_depth + ask_depth
    midpoint = (best_bid + best_ask) / 2
    microprice = (best_ask * best_bid_size + best_bid * best_ask_size) / (
        best_bid_size + best_ask_size
    )
    return {
        "received_at": received_at,
        "last_update_id": int(payload["lastUpdateId"]),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": midpoint,
        "spread_bps": 10_000 * (best_ask - best_bid) / midpoint,
        "bid_depth_20": bid_depth,
        "ask_depth_20": ask_depth,
        "depth_imbalance": (bid_depth - ask_depth) / total_depth,
        "microprice": microprice,
        "microprice_deviation_bps": 10_000 * (microprice - midpoint) / midpoint,
    }


def summarize(rows: list[dict[str, float | str]], elapsed_seconds: float, trades: int) -> dict[str, Any]:
    spreads = [float(row["spread_bps"]) for row in rows]
    imbalances = [float(row["depth_imbalance"]) for row in rows]
    deviations = [float(row["microprice_deviation_bps"]) for row in rows]
    return {
        "depth_snapshots": len(rows),
        "agg_trade_messages": trades,
        "elapsed_seconds": round(elapsed_seconds, 4),
        "snapshot_rate_per_second": round(len(rows) / elapsed_seconds, 4),
        "spread_bps_mean": round(statistics.mean(spreads), 6),
        "spread_bps_p95": round(sorted(spreads)[int(0.95 * (len(spreads) - 1))], 6),
        "depth_imbalance_mean": round(statistics.mean(imbalances), 6),
        "depth_imbalance_stdev": round(statistics.pstdev(imbalances), 6),
        "microprice_deviation_bps_mean": round(statistics.mean(deviations), 6),
        "data_scope": "live public Binance spot top-20 partial-book snapshots",
        "limitation": "feature validation only; not a historical L2 training or execution dataset",
    }


async def collect(
    *,
    symbol: str,
    snapshots: int,
    output_csv: Path,
    output_summary: Path,
    websocket_base: str = BINANCE_WS,
) -> dict[str, Any]:
    stream_symbol = symbol.lower()
    streams = f"{stream_symbol}@depth20@100ms/{stream_symbol}@aggTrade"
    url = f"{websocket_base}/stream?streams={streams}"
    rows: list[dict[str, float | str]] = []
    trades = 0
    started = time.monotonic()

    async with websockets.connect(url, ping_interval=20, ping_timeout=60, max_queue=4096) as socket:
        while len(rows) < snapshots:
            message = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
            stream = message.get("stream", "")
            payload = message.get("data", message)
            if stream.endswith("@depth20@100ms"):
                received_at = datetime.now(timezone.utc).isoformat()
                rows.append(depth_features(payload, received_at))
            elif stream.endswith("@aggTrade"):
                trades += 1

    elapsed = time.monotonic() - started
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = summarize(rows, elapsed, trades)
    report.update({"symbol": symbol.upper(), "websocket_url": url})
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
