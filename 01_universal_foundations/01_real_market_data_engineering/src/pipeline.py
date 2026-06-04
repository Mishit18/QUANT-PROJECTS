# src/pipeline.py
"""
Master pipeline orchestrator.

Runs the full market-data engineering workflow from raw CSV ingestion to
diagnostic backtesting. The emphasis is reproducibility and honest diagnostics,
not optimized strategy PnL.
"""
import json
import logging

import numpy as np
import pandas as pd

from config import (
    RAW_CSV,
    CLEANED_PARQUET,
    FEATURES_PARQUET,
    LOADED_PARQUET,
    REPORTS_DIR,
    STATIONARY_PARQUET,
    backtest_config,
    drift_config,
)
from load_data import load_market_data
from cleaning_v2 import clean_market_data
from stationarity_tests import test_price_and_returns
from drift_removal import remove_drift_and_decompose
from feature_engineering import FeatureEngineer
from backtest_engine import BacktestEngine, volatility_scaled_momentum_strategy


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def run_full_pipeline():
    """Execute the complete data engineering pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING FULL DATA ENGINEERING PIPELINE")
    logger.info("=" * 80 + "\n")

    logger.info("STEP 1: Loading raw data...")
    df = load_market_data(RAW_CSV, save_parquet=True)
    logger.info("OK Loaded %s rows", len(df))

    logger.info("\nSTEP 2: Data cleaning...")
    df = pd.read_parquet(LOADED_PARQUET)
    df, cleaning_summary = clean_market_data(df)
    df.to_parquet(CLEANED_PARQUET)
    logger.info(
        "OK Flagged %s bad ticks (%.2f%%); post-clean OHLC invalid rows: %s",
        cleaning_summary["bad_ticks"],
        cleaning_summary["bad_ticks_pct"],
        cleaning_summary["post_clean_ohlc_invalid"],
    )
    logger.info("OK Saved cleaned data to %s", CLEANED_PARQUET)

    logger.info("\nSTEP 3: Running stationarity tests...")
    test_results = test_price_and_returns(CLEANED_PARQUET, REPORTS_DIR)
    logger.info("OK Stationarity tests complete")

    logger.info("\nSTEP 4: Drift removal and HP filter decomposition...")
    df = pd.read_parquet(CLEANED_PARQUET)
    df = remove_drift_and_decompose(
        df,
        price_col="close",
        method=drift_config.detrend_method,
        hp_lambda=drift_config.hp_lambda,
        rolling_window=drift_config.rolling_window,
    )
    df.to_parquet(STATIONARY_PARQUET)
    logger.info("OK Saved stationary data to %s", STATIONARY_PARQUET)

    logger.info("\nSTEP 5: Feature engineering...")
    df = pd.read_parquet(STATIONARY_PARQUET)
    engineer = FeatureEngineer(df)
    df_features = engineer.build_all_features()
    feature_output_cols = [
        col for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "log_return",
            "simple_return",
            "flag_bad_tick",
            "flag_ohlc_invalid_raw",
            "flag_price_outlier",
            "flag_return_outlier",
            "volume_is_informative",
        ]
        if col in df_features.columns
    ]
    feature_output_cols.extend(engineer.feature_names)
    feature_output_cols.extend(engineer.target_names)
    feature_output_cols = list(dict.fromkeys(feature_output_cols))
    df_features = df_features[feature_output_cols]
    df_features.to_parquet(FEATURES_PARQUET, compression="zstd")
    logger.info(
        "OK Created %s features and %s targets",
        len(engineer.feature_names),
        len(engineer.target_names),
    )
    logger.info("OK Saved features to %s", FEATURES_PARQUET)

    logger.info("\nSTEP 6: Running leakage-safe diagnostic backtest...")
    signals = volatility_scaled_momentum_strategy(
        df_features,
        lookback=backtest_config.momentum_lookback,
        vol_window=backtest_config.signal_vol_window,
        threshold=backtest_config.signal_threshold,
    )

    engine = BacktestEngine(
        initial_capital=backtest_config.initial_capital,
        commission_rate=backtest_config.commission_rate,
        slippage_bps=backtest_config.slippage_bps,
    )
    results = engine.simulate_strategy(
        df_features,
        signals,
        execution_lag=backtest_config.execution_lag,
    )
    metrics = engine.calculate_metrics(results)

    logger.info("\n" + "-" * 60)
    logger.info("BACKTEST METRICS (Diagnostic Vol-Scaled Momentum)")
    logger.info("-" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info("  %-35s %.6f", key, value)
        else:
            logger.info("  %-35s %s", key, value)
    logger.info("-" * 60)

    backtest_cols = [
        "close",
        "signal",
        "position",
        "asset_returns",
        "turnover",
        "transaction_costs",
        "gross_returns",
        "returns",
        "equity",
    ]
    results[backtest_cols].to_parquet(REPORTS_DIR / "backtest_results.parquet", compression="zstd")
    logger.info("OK Saved backtest results")

    logger.info("\nSTEP 7: Generating summary report...")
    summary = {
        "pipeline_version": "2.0",
        "data_source": str(RAW_CSV),
        "total_rows": int(len(df_features)),
        "date_range": f"{df_features.index.min()} to {df_features.index.max()}",
        "cleaning_summary": cleaning_summary,
        "total_features": int(len(engineer.feature_names)),
        "total_targets": int(len(engineer.target_names)),
        "target_columns": engineer.target_names,
        "stationarity_tests": test_results,
        "backtest_metrics": metrics,
        "backtest_note": "Diagnostic only: India VIX is an index, not a directly tradeable instrument.",
    }

    with open(REPORTS_DIR / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, default=str)
    logger.info("OK Saved pipeline summary")

    logger.info("\nSTEP 8: Generating diagnostic plots...")
    try:
        from visualizations import generate_all_plots

        generate_all_plots()
        logger.info("OK Diagnostic plots generated")
    except Exception as exc:
        logger.warning("Plot generation failed: %s", exc)

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 80)
    logger.info("Outputs:")
    logger.info("  - Loaded data: %s", LOADED_PARQUET)
    logger.info("  - Cleaned data: %s", CLEANED_PARQUET)
    logger.info("  - Stationary data: %s", STATIONARY_PARQUET)
    logger.info("  - Features: %s", FEATURES_PARQUET)
    logger.info("  - Reports: %s", REPORTS_DIR)
    logger.info("=" * 80 + "\n")

    return summary


if __name__ == "__main__":
    run_full_pipeline()
