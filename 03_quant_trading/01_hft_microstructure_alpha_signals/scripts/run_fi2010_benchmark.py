"""Run a chronological real-data benchmark on FI-2010."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.fi2010 import concatenate, download_archive, extract_archive, load_partition, sha256


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, object]:
    return {
        "observations": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def main() -> None:
    cache = ROOT / ".cache" / "fi2010"
    archive = download_archive(cache / "fi2010.zip")
    files = {path.name: path for path in extract_archive(archive, cache / "extracted")}

    train = load_partition(files["Train_Dst_NoAuction_DecPre_CF_7.txt"], horizon=10)
    validation = load_partition(files["Test_Dst_NoAuction_DecPre_CF_7.txt"], horizon=10)
    holdout = concatenate(
        [
            load_partition(files["Test_Dst_NoAuction_DecPre_CF_8.txt"], horizon=10),
            load_partition(files["Test_Dst_NoAuction_DecPre_CF_9.txt"], horizon=10),
        ]
    )
    x_train, y_train = train
    x_validation, y_validation = validation
    x_holdout, y_holdout = holdout

    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_validation, y_validation)],
        verbose=False,
    )

    results = {
        "dataset": "FI-2010 Nasdaq Nordic limit order book benchmark",
        "archive_sha256": sha256(archive),
        "data_provenance": "Real Nasdaq Nordic data; decimal-precision normalized by dataset authors",
        "stocks": 5,
        "trading_days": 10,
        "book_levels": 10,
        "total_observations": int(len(y_train) + len(y_validation) + len(y_holdout)),
        "input_features": int(x_train.shape[1]),
        "label_horizon_events": 10,
        "split": {
            "train_days_1_7": int(len(y_train)),
            "validation_day_8": int(len(y_validation)),
            "holdout_days_9_10": int(len(y_holdout)),
        },
        "dummy_holdout": metrics(y_holdout, dummy.predict(x_holdout)),
        "xgboost_validation": metrics(y_validation, model.predict(x_validation)),
        "xgboost_holdout": metrics(y_holdout, model.predict(x_holdout)),
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "fi2010_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = f"""# FI-2010 Real LOB Benchmark

This experiment replaces the primary synthetic LOB benchmark with the real FI-2010 academic dataset: five Nasdaq Nordic stocks, ten levels, and ten trading days. The archive contains **{results['total_observations']:,} observations**. Only the 40 price/volume book variables and derived state features are used; the authors' engineered features are excluded to reduce leakage risk.

## Chronological protocol

- Train: days 1-7 ({results['split']['train_days_1_7']:,} observations)
- Validation: day 8 ({results['split']['validation_day_8']:,} observations)
- Untouched holdout: days 9-10 ({results['split']['holdout_days_9_10']:,} observations)
- Target: published 10-event mid-price direction label

## Holdout results

| Model | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|
| Majority baseline | {results['dummy_holdout']['accuracy']:.4f} | {results['dummy_holdout']['balanced_accuracy']:.4f} | {results['dummy_holdout']['macro_f1']:.4f} |
| XGBoost | {results['xgboost_holdout']['accuracy']:.4f} | {results['xgboost_holdout']['balanced_accuracy']:.4f} | {results['xgboost_holdout']['macro_f1']:.4f} |

## Interpretation and limits

FI-2010 is real historical equity LOB data, but it is an academic benchmark from June 2010 and is decimal-precision normalized. The classification result is not a PnL claim. Queue position, fees, latency, hidden liquidity, and current-market transfer still require separate evaluation.
"""
    (reports / "FI2010_REAL_LOB_BENCHMARK.md").write_text(report, encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
