"""FI-2010 real limit-order-book acquisition, parsing, and features."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd


FI2010_URL = (
    "https://raw.githubusercontent.com/zcakhaa/"
    "DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/"
    "master/data/data.zip"
)
FI2010_SHA256 = "7a5564290a504bdc104dccaf44aa6c0b3857bf1db6b0517b9b88577c2def3e5f"
HORIZON_ROWS = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_archive(destination: Path, url: str = FI2010_URL) -> Path:
    """Download the mirrored FI-2010 archive and enforce its committed checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == FI2010_SHA256:
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": "fi2010-research/1.0"})
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if sha256(temporary) != FI2010_SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError("FI-2010 archive checksum mismatch")
    temporary.replace(destination)
    return destination


def extract_archive(archive: Path, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zipped:
        zipped.extractall(destination)
    return sorted(destination.glob("*.txt"))


def load_partition(path: Path, horizon: int = 10) -> tuple[pd.DataFrame, pd.Series]:
    """Load one FI-2010 partition, retaining only 40 LOB variables and one label row."""
    if horizon not in HORIZON_ROWS:
        raise ValueError(f"Unsupported horizon {horizon}; choose {sorted(HORIZON_ROWS)}")
    matrix = np.loadtxt(path, dtype=np.float32)
    if matrix.shape[0] < 149:
        raise ValueError(f"Expected at least 149 rows, found {matrix.shape[0]}")
    features = build_lob_features(matrix[:40].T)
    labels = pd.Series(matrix[HORIZON_ROWS[horizon]].astype(np.int8) - 1, name="label")
    if not labels.isin([0, 1, 2]).all():
        raise ValueError("FI-2010 labels must map to classes 0, 1, and 2")
    return features, labels


def build_lob_features(levels: np.ndarray) -> pd.DataFrame:
    """Create leakage-free state features from ten ask/bid price-volume levels."""
    if levels.ndim != 2 or levels.shape[1] != 40:
        raise ValueError("Expected observations x 40 FI-2010 LOB variables")

    values: dict[str, np.ndarray] = {}
    ask_prices, ask_sizes, bid_prices, bid_sizes = [], [], [], []
    for level in range(10):
        offset = 4 * level
        ask_price = levels[:, offset]
        ask_size = levels[:, offset + 1]
        bid_price = levels[:, offset + 2]
        bid_size = levels[:, offset + 3]
        ask_prices.append(ask_price)
        ask_sizes.append(ask_size)
        bid_prices.append(bid_price)
        bid_sizes.append(bid_size)
        values[f"ask_price_{level + 1}"] = ask_price
        values[f"ask_size_{level + 1}"] = ask_size
        values[f"bid_price_{level + 1}"] = bid_price
        values[f"bid_size_{level + 1}"] = bid_size

    ask_prices_a = np.column_stack(ask_prices)
    ask_sizes_a = np.column_stack(ask_sizes)
    bid_prices_a = np.column_stack(bid_prices)
    bid_sizes_a = np.column_stack(bid_sizes)
    epsilon = np.float32(1e-8)
    size_total = ask_sizes_a + bid_sizes_a
    level_imbalance = (bid_sizes_a - ask_sizes_a) / (size_total + epsilon)

    for level in range(10):
        values[f"queue_imbalance_{level + 1}"] = level_imbalance[:, level]
    values["spread_l1"] = ask_prices_a[:, 0] - bid_prices_a[:, 0]
    values["midprice_l1"] = (ask_prices_a[:, 0] + bid_prices_a[:, 0]) / 2
    values["microprice_l1"] = (
        ask_prices_a[:, 0] * bid_sizes_a[:, 0]
        + bid_prices_a[:, 0] * ask_sizes_a[:, 0]
    ) / (size_total[:, 0] + epsilon)
    values["depth_imbalance_5"] = (
        bid_sizes_a[:, :5].sum(axis=1) - ask_sizes_a[:, :5].sum(axis=1)
    ) / (bid_sizes_a[:, :5].sum(axis=1) + ask_sizes_a[:, :5].sum(axis=1) + epsilon)
    values["depth_imbalance_10"] = (
        bid_sizes_a.sum(axis=1) - ask_sizes_a.sum(axis=1)
    ) / (bid_sizes_a.sum(axis=1) + ask_sizes_a.sum(axis=1) + epsilon)
    values["ask_book_slope"] = ask_prices_a[:, -1] - ask_prices_a[:, 0]
    values["bid_book_slope"] = bid_prices_a[:, 0] - bid_prices_a[:, -1]
    return pd.DataFrame(values).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def concatenate(parts: Iterable[tuple[pd.DataFrame, pd.Series]]) -> tuple[pd.DataFrame, pd.Series]:
    features, labels = zip(*parts)
    return pd.concat(features, ignore_index=True), pd.concat(labels, ignore_index=True)
