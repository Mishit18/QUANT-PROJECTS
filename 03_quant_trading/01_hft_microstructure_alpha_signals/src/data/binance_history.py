"""Historical Binance aggregate-trade benchmark with chronological evaluation."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BINANCE_VISION = "https://data.binance.vision/data/spot/daily/aggTrades"
COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "buyer_is_maker",
    "best_match",
]


def download_agg_trades(symbol: str, date: str, cache_dir: Path) -> Path:
    """Download one official Binance Vision daily aggregate-trade archive."""
    symbol = symbol.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"{symbol}-aggTrades-{date}.zip"
    if not archive.exists():
        url = f"{BINANCE_VISION}/{symbol}/{archive.name}"
        urlretrieve(url, archive)
    return archive


def load_agg_trades(archive: Path) -> pd.DataFrame:
    """Load aggregate trades and normalize millisecond/microsecond timestamps."""
    with ZipFile(archive) as zipped:
        files = [name for name in zipped.namelist() if name.endswith(".csv")]
        if len(files) != 1:
            raise ValueError(f"Expected one CSV in {archive}, found {len(files)}")
        with zipped.open(files[0]) as handle:
            frame = pd.read_csv(handle, names=COLUMNS, header=None)
    for column in ["price", "quantity", "timestamp"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["price", "quantity", "timestamp"])
    unit = "us" if frame["timestamp"].median() > 1e14 else "ms"
    frame["event_time"] = pd.to_datetime(frame["timestamp"], unit=unit, utc=True)
    frame["signed_quantity"] = np.where(frame["buyer_is_maker"], -frame["quantity"], frame["quantity"])
    return frame.sort_values("event_time").reset_index(drop=True)


def build_trade_features(trades: pd.DataFrame, horizon_seconds: int = 5) -> pd.DataFrame:
    """Aggregate to one-second bars and build strictly backward-looking features."""
    indexed = trades.set_index("event_time")
    bars = indexed.resample("1s").agg(
        price=("price", "last"),
        trade_count=("price", "size"),
        volume=("quantity", "sum"),
        signed_volume=("signed_quantity", "sum"),
        high=("price", "max"),
        low=("price", "min"),
    )
    bars["price"] = bars["price"].ffill()
    for column in ["trade_count", "volume", "signed_volume"]:
        bars[column] = bars[column].fillna(0.0)
    bars["return_1s"] = np.log(bars["price"]).diff()
    bars["return_5s"] = np.log(bars["price"]).diff(5)
    bars["volatility_30s"] = bars["return_1s"].rolling(30).std()
    bars["trade_intensity_10s"] = bars["trade_count"].rolling(10).sum()
    bars["volume_imbalance_10s"] = (
        bars["signed_volume"].rolling(10).sum()
        / bars["volume"].rolling(10).sum().replace(0, np.nan)
    )
    bars["volume_imbalance_30s"] = (
        bars["signed_volume"].rolling(30).sum()
        / bars["volume"].rolling(30).sum().replace(0, np.nan)
    )
    bars["range_bps"] = 10_000 * (bars["high"] - bars["low"]) / bars["price"]
    bars["future_return"] = np.log(bars["price"].shift(-horizon_seconds) / bars["price"])
    bars["target_up"] = (bars["future_return"] > 0).astype(int)
    return bars.replace([np.inf, -np.inf], np.nan).dropna()


def run_walk_forward_benchmark(features: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Fit on the first 60%, tune on 20%, and report the untouched final 20%."""
    columns = [
        "return_1s",
        "return_5s",
        "volatility_30s",
        "trade_intensity_10s",
        "volume_imbalance_10s",
        "volume_imbalance_30s",
        "range_bps",
    ]
    first, second = int(0.60 * len(features)), int(0.80 * len(features))
    train, validation, test = features.iloc[:first], features.iloc[first:second], features.iloc[second:]
    model = Pipeline(
        [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    )
    model.fit(train[columns], train["target_up"])
    validation_probability = model.predict_proba(validation[columns])[:, 1]
    thresholds = np.arange(0.50, 0.71, 0.02)
    validation_scores = [
        balanced_accuracy_score(validation["target_up"], validation_probability >= threshold)
        for threshold in thresholds
    ]
    threshold = float(thresholds[int(np.argmax(validation_scores))])

    probability = model.predict_proba(test[columns])[:, 1]
    prediction = probability >= threshold
    active = (probability >= 0.55) | (probability <= 0.45)
    direction = np.where(probability >= 0.55, 1.0, np.where(probability <= 0.45, -1.0, 0.0))
    cost = 0.0001
    net_return = direction * test["future_return"].to_numpy() - active.astype(float) * cost
    metrics = pd.DataFrame(
        [
            {
                "dataset": "Binance Vision BTCUSDT aggregate trades",
                "bars": len(features),
                "train_bars": len(train),
                "validation_bars": len(validation),
                "test_bars": len(test),
                "decision_threshold": threshold,
                "accuracy": accuracy_score(test["target_up"], prediction),
                "balanced_accuracy": balanced_accuracy_score(test["target_up"], prediction),
                "roc_auc": roc_auc_score(test["target_up"], probability),
                "active_signal_rate": float(active.mean()),
                "mean_net_return_bps_per_active_signal": float(net_return[active].mean() * 10_000) if active.any() else np.nan,
                "assumed_round_trip_cost_bps": cost * 10_000,
            }
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "real_binance_trade_walk_forward_metrics.csv", index=False)
    test_output = test[["price", "future_return", "target_up"]].copy()
    test_output["probability_up"] = probability
    test_output["signal"] = direction
    test_output.to_csv(output_dir / "real_binance_trade_holdout_predictions.csv")
    row = metrics.iloc[0]
    report = f"""# Real Binance Trade-Flow Benchmark

The benchmark uses {int(row['bars']):,} one-second bars derived from official Binance Vision BTCUSDT aggregate trades. Data is split chronologically into 60% train, 20% validation, and 20% untouched holdout partitions.

- Holdout ROC-AUC: {row['roc_auc']:.4f}
- Holdout balanced accuracy: {row['balanced_accuracy']:.4f}
- Active-signal rate: {row['active_signal_rate']:.1%}
- Mean net return per active signal after a {row['assumed_round_trip_cost_bps']:.1f} bp cost: {row['mean_net_return_bps_per_active_signal']:.3f} bps

The negative post-cost result is a research rejection, not a trading-performance claim. Aggregate trades validate trade-flow features but cannot reconstruct queue position or full-depth LOB state.
"""
    (output_dir / "REAL_BINANCE_TRADE_BENCHMARK.md").write_text(report, encoding="utf-8")
    return metrics
