import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.ic import compute_ic_series, compute_ic_statistics
from src.neutralization.market import neutralize_market_beta
from src.neutralization.pca import neutralize_pca_factors
from src.neutralization.sector import neutralize_sector


def create_synthetic_sectors(tickers: list, n_sectors: int = 10) -> dict:
    """Create deterministic synthetic sector assignments."""
    sectors = [f"SECTOR_{i % n_sectors}" for i in range(len(tickers))]
    return dict(zip(tickers, sectors))


def to_wide_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Convert saved prediction parquet output to date x asset format."""
    if predictions.index.nlevels > 1:
        if "prediction" in predictions.columns:
            return predictions["prediction"].unstack()
        return predictions.unstack()
    return predictions


def to_wide_target_frame(targets: pd.DataFrame) -> pd.DataFrame:
    """Convert saved target parquet output to date x asset format."""
    if targets.index.nlevels > 1:
        if "target" in targets.columns:
            return targets["target"].unstack()
        return targets.unstack()
    return targets


def summarize_ic(label: str, predictions: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Compute and print fixed-signal IC against a target variant."""
    ic = compute_ic_series(predictions, targets)
    stats = compute_ic_statistics(ic)
    print(
        f"{label:<20s} IC={stats['mean_ic']:>8.4f} "
        f"IR={stats['ic_ir']:>6.2f} Hit={stats['hit_rate']:>6.1%} "
        f"t={stats['t_stat']:>6.2f} p={stats['p_value']:>7.4f} "
        f"N={stats['n_obs']}"
    )
    return stats


def pct_retention(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) < 1e-8:
        return np.nan
    return value / baseline * 100


def main():
    config_path = Path("src/config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pred_path = Path("data/processed/predictions_xgboost.parquet")
    if not pred_path.exists():
        raise FileNotFoundError("Run scripts/run_models.py before risk neutralization analysis.")

    predictions = to_wide_prediction_frame(pd.read_parquet(pred_path))
    returns = pd.read_parquet("data/processed/returns.parquet")
    targets = to_wide_target_frame(pd.read_parquet("data/processed/targets.parquet"))

    common_dates = predictions.index.intersection(targets.index).intersection(returns.index)
    predictions = predictions.loc[common_dates]
    targets = targets.loc[common_dates]
    returns = returns.loc[common_dates]

    market_returns = returns.mean(axis=1)
    sector_map = create_synthetic_sectors(returns.columns.tolist())

    print("=" * 78)
    print("RISK NEUTRALIZATION ANALYSIS: FIXED OUT-OF-SAMPLE SIGNAL")
    print("=" * 78)
    print(f"Data: {len(predictions)} dates, {predictions.shape[1]} assets")
    print(f"Target method: {config['targets']['method']}, horizon={config['targets']['horizon']}d")
    print("Method: neutralize realized target, then recompute IC for saved XGBoost signal")
    print("=" * 78)

    results = {}
    results["Baseline"] = summarize_ic("Baseline", predictions, targets)

    market_neutral = neutralize_market_beta(targets, returns, market_returns, window=252)
    results["Market-neutral"] = summarize_ic("Market-neutral", predictions, market_neutral)

    sector_neutral = neutralize_sector(targets, sector_map)
    results["Sector-neutral"] = summarize_ic("Sector-neutral", predictions, sector_neutral)

    combined = neutralize_sector(market_neutral, sector_map)
    results["Market+Sector"] = summarize_ic("Market+Sector", predictions, combined)

    if os.environ.get("RUN_PCA_DIAGNOSTICS") == "1":
        for n_factors in [3, 5, 10]:
            pca_neutral = neutralize_pca_factors(targets, returns, n_factors=n_factors, window=252)
            results[f"PCA-{n_factors}"] = summarize_ic(f"PCA-{n_factors}", predictions, pca_neutral)
    else:
        print("PCA diagnostics skipped by default; set RUN_PCA_DIAGNOSTICS=1 to run the slow rolling PCA pass.")

    baseline_ic = results["Baseline"]["mean_ic"]

    print("\n" + "=" * 78)
    print("IC RETENTION VS BASELINE")
    print("=" * 78)
    for label, stats in results.items():
        retention = pct_retention(stats["mean_ic"], baseline_ic)
        retention_text = "n/a" if np.isnan(retention) else f"{retention:6.1f}%"
        print(f"{label:<20s} {stats['mean_ic']:>8.4f}  retention={retention_text}")

    print("\nCONCLUSION:")
    if abs(baseline_ic) < 0.01:
        print("[FAIL] Baseline IC is statistically and economically near zero; factor retention is not meaningful.")
        print("       Treat this as a rejected synthetic signal, not a deployable alpha.")
    else:
        print("[INFO] Use the retention table to judge whether the fixed signal survives risk adjustment.")


if __name__ == "__main__":
    main()
