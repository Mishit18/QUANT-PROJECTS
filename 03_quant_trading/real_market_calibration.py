"""Build shared empirical market priors for simulation-heavy quant projects."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
URL = "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2026-08-15.zip"
COLUMNS = ["id", "price", "quantity", "first_id", "last_id", "timestamp", "buyer_is_maker", "best_match"]


def load_trades() -> pd.DataFrame:
    cache = ROOT / ".cache" / "BTCUSDT-aggTrades-2026-08-15.zip"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        urlretrieve(URL, cache)
    with ZipFile(cache) as zipped:
        csv_name = next(name for name in zipped.namelist() if name.endswith(".csv"))
        with zipped.open(csv_name) as handle:
            frame = pd.read_csv(handle, names=COLUMNS, header=None)
    unit = "us" if frame["timestamp"].median() > 1e14 else "ms"
    frame["time"] = pd.to_datetime(frame["timestamp"], unit=unit, utc=True)
    frame["side"] = np.where(frame["buyer_is_maker"], -1.0, 1.0)
    frame["signed_volume"] = frame["side"] * frame["quantity"]
    return frame.sort_values("time").reset_index(drop=True)


def calibration_metrics(frame: pd.DataFrame) -> dict[str, float | int | str]:
    duration = (frame["time"].iloc[-1] - frame["time"].iloc[0]).total_seconds()
    interarrival_ms = frame["time"].diff().dt.total_seconds().mul(1000).dropna()
    seconds = frame.set_index("time").resample("1s").agg(price=("price", "last"), trades=("id", "size"))
    returns = np.log(seconds["price"].ffill()).diff().dropna()
    count_mean = seconds["trades"].mean()
    return {
        "source": "Official Binance Vision BTCUSDT aggregate trades",
        "date": "2026-08-15",
        "trades": int(len(frame)),
        "duration_seconds": round(float(duration), 3),
        "trades_per_second": round(float(len(frame) / duration), 4),
        "buyer_initiated_share": round(float((frame["side"] > 0).mean()), 5),
        "interarrival_p50_ms": round(float(interarrival_ms.quantile(0.50)), 3),
        "interarrival_p90_ms": round(float(interarrival_ms.quantile(0.90)), 3),
        "trade_count_dispersion": round(float(seconds["trades"].var() / count_mean), 4),
        "one_second_volatility_bps": round(float(returns.std() * 10_000), 5),
        "daily_realized_volatility_pct": round(float(np.sqrt(np.square(returns).sum()) * 100), 5),
        "quantity_p50_btc": round(float(frame["quantity"].quantile(0.50)), 6),
        "quantity_p90_btc": round(float(frame["quantity"].quantile(0.90)), 6),
        "signed_volume_autocorrelation_lag1": round(float(frame["signed_volume"].autocorr(1)), 5),
    }


def write_project_evidence(metrics: dict[str, float | int | str]) -> None:
    projects = {
        "02_avellaneda_stoikov_market_making": (
            "Volatility and market-order arrival intensity are empirical priors; quote fills, queue position, and PnL remain simulated."
        ),
        "05_hawkes_process_market_making": (
            "Trade interarrival dispersion and signed-flow dependence validate clustering priors; the six-type LOB Hawkes kernel remains simulation-based because aggregate trades contain no limit-order or cancellation events."
        ),
        "06_optimal_execution_reinforcement_learning": (
            "Realized volatility, trade intensity, and size quantiles calibrate stress ranges; temporary/permanent impact and policy rewards remain simulated."
        ),
    }
    for project, limitation in projects.items():
        project_root = ROOT / project
        results = project_root / "results"
        docs = project_root / "docs"
        results.mkdir(parents=True, exist_ok=True)
        docs.mkdir(parents=True, exist_ok=True)
        (results / "real_binance_empirical_priors.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        report = f"""# Real-Market Calibration Evidence

Empirical priors were calculated from {metrics['trades']:,} official Binance Vision BTCUSDT aggregate trades on {metrics['date']}.

- Trade intensity: {metrics['trades_per_second']:.2f} trades/second
- Interarrival P50/P90: {metrics['interarrival_p50_ms']:.1f}/{metrics['interarrival_p90_ms']:.1f} ms
- Trade-count variance/mean: {metrics['trade_count_dispersion']:.2f}, evidencing clustered arrivals
- One-second volatility: {metrics['one_second_volatility_bps']:.3f} bps
- Daily realized volatility: {metrics['daily_realized_volatility_pct']:.2f}%

## Scope Boundary

{limitation}
"""
        (docs / "REAL_MARKET_CALIBRATION.md").write_text(report, encoding="utf-8")


def main() -> None:
    metrics = calibration_metrics(load_trades())
    write_project_evidence(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
