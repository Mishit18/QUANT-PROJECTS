import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.neutralization.market import neutralize_market_beta
from src.neutralization.sector import neutralize_sector
from src.neutralization.pca import neutralize_pca_factors
from src.backtest.walkforward import WalkForwardValidator
from src.evaluation.ic import compute_ic_statistics


def create_synthetic_sectors(tickers: list, n_sectors: int = 10) -> dict:
    """Create synthetic sector assignments."""
    np.random.seed(42)
    sectors = [f'SECTOR_{i%n_sectors}' for i in range(len(tickers))]
    return dict(zip(tickers, sectors))


def neutralize_targets(targets: pd.DataFrame, returns: pd.DataFrame, 
                      method: str, **kwargs) -> pd.DataFrame:
    """Apply neutralization to targets."""
    if method == 'market':
        market_returns = kwargs.get('market_returns')
        window = kwargs.get('window', 252)
        return neutralize_market_beta(targets, returns, market_returns, window)
    
    elif method == 'sector':
        sector_map = kwargs.get('sector_map')
        return neutralize_sector(targets, sector_map)
    
    elif method == 'pca':
        n_factors = kwargs.get('n_factors', 10)
        window = kwargs.get('window', 252)
        return neutralize_pca_factors(targets, returns, n_factors, window)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def evaluate_neutralized_ic(features: pd.DataFrame, targets_original: pd.DataFrame,
                            targets_neutralized: pd.DataFrame, config: dict,
                            label: str) -> dict:
    """Evaluate IC after neutralization."""
    validator = WalkForwardValidator(config)
    
    # Clean features
    nan_pct = features.isna().mean()
    valid_features = nan_pct[nan_pct < 0.9].index
    features_clean = features[valid_features]
    features_clean = features_clean.dropna(thresh=features_clean.shape[1] * 0.2)
    
    results = {}
    
    for model_name in ['ols', 'ridge', 'xgboost']:
        # Original IC
        result_orig = validator.validate(features_clean, targets_original, model_type=model_name)
        ic_orig = compute_ic_statistics(result_orig['test_ic']) if len(result_orig['test_ic']) > 0 else None
        
        # Neutralized IC
        result_neut = validator.validate(features_clean, targets_neutralized, model_type=model_name)
        ic_neut = compute_ic_statistics(result_neut['test_ic']) if len(result_neut['test_ic']) > 0 else None
        
        if ic_orig and ic_neut:
            results[model_name] = {
                'ic_original': ic_orig['mean_ic'],
                'ic_neutralized': ic_neut['mean_ic'],
                'ir_original': ic_orig['ic_ir'],
                'ir_neutralized': ic_neut['ic_ir'],
                'hit_original': ic_orig['hit_rate'],
                'hit_neutralized': ic_neut['hit_rate'],
                'ic_retention': ic_neut['mean_ic'] / ic_orig['mean_ic'] if ic_orig['mean_ic'] != 0 else np.nan,
                'n_periods': len(result_neut['test_ic'])
            }
    
    return results


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    features = pd.read_parquet('data/features/features.parquet')
    returns = pd.read_parquet('data/processed/returns.parquet')
    targets_df = pd.read_parquet('data/processed/targets.parquet')
    
    # Unstack targets to wide format
    if isinstance(targets_df, pd.DataFrame) and 'target' in targets_df.columns:
        targets = targets_df['target'].unstack()
    else:
        targets = targets_df.unstack() if targets_df.index.nlevels > 1 else targets_df
    
    # Align with returns
    common_dates = targets.index.intersection(returns.index)
    targets = targets.loc[common_dates]
    returns = returns.loc[common_dates]
    
    market_returns = returns.mean(axis=1)
    sector_map = create_synthetic_sectors(returns.columns.tolist())
    
    print("=" * 70)
    print("RISK NEUTRALIZATION ANALYSIS")
    print("=" * 70)
    print(f"\nData: {len(targets)} dates, {targets.shape[1]} assets")
    print(f"Target method: {config['targets']['method']}")
    print(f"Horizon: {config['targets']['horizon']} days")
    print("\nUsing XGBoost only for speed...")
    
    # Prepare features
    nan_pct = features.isna().mean()
    valid_features = nan_pct[nan_pct < 0.9].index
    features_clean = features[valid_features]
    features_clean = features_clean.dropna(thresh=features_clean.shape[1] * 0.2)
    
    validator = WalkForwardValidator(config)
    model_name = 'xgboost'
    
    # Baseline (no neutralization)
    print("\n" + "=" * 70)
    print("[BASELINE] NO NEUTRALIZATION")
    print("=" * 70)
    
    targets_stacked = targets.stack()
    result = validator.validate(features_clean, targets_stacked, model_type=model_name)
    ic_baseline = compute_ic_statistics(result['test_ic']) if len(result['test_ic']) > 0 else None
    
    if ic_baseline:
        print(f"IC={ic_baseline['mean_ic']:7.4f}, IR={ic_baseline['ic_ir']:5.2f}, Hit={ic_baseline['hit_rate']:5.1%}")
        ic_base = ic_baseline['mean_ic']
    else:
        print("No IC computed")
        return
    
    # 1. Market Neutralization
    print("\n" + "=" * 70)
    print("[1] MARKET BETA NEUTRALIZATION")
    print("=" * 70)
    
    print("Neutralizing market beta (252-day rolling window)...")
    targets_market_neut = neutralize_market_beta(targets, returns, market_returns, window=252)
    targets_market_neut_stacked = targets_market_neut.stack()
    
    result = validator.validate(features_clean, targets_market_neut_stacked, model_type=model_name)
    ic_market = compute_ic_statistics(result['test_ic']) if len(result['test_ic']) > 0 else None
    
    if ic_market:
        ic_mkt = ic_market['mean_ic']
        retention_mkt = (ic_mkt / ic_base * 100) if ic_base != 0 else np.nan
        print(f"IC={ic_mkt:7.4f} (was {ic_base:7.4f}), Retention={retention_mkt:5.1f}%")
    
    # 2. Sector Neutralization
    print("\n" + "=" * 70)
    print("[2] SECTOR NEUTRALIZATION")
    print("=" * 70)
    
    print(f"Neutralizing {len(set(sector_map.values()))} sectors...")
    targets_sector_neut = neutralize_sector(targets, sector_map)
    targets_sector_neut_stacked = targets_sector_neut.stack()
    
    result = validator.validate(features_clean, targets_sector_neut_stacked, model_type=model_name)
    ic_sector = compute_ic_statistics(result['test_ic']) if len(result['test_ic']) > 0 else None
    
    if ic_sector:
        ic_sec = ic_sector['mean_ic']
        retention_sec = (ic_sec / ic_base * 100) if ic_base != 0 else np.nan
        print(f"IC={ic_sec:7.4f} (was {ic_base:7.4f}), Retention={retention_sec:5.1f}%")
    
    # 3. PCA Factor Neutralization
    print("\n" + "=" * 70)
    print("[3] PCA FACTOR NEUTRALIZATION")
    print("=" * 70)
    
    pca_results = {}
    for n_factors in [3, 5, 10]:
        print(f"\nRemoving top {n_factors} PCA factors...")
        targets_pca_neut = neutralize_pca_factors(targets, returns, n_factors=n_factors, window=252)
        targets_pca_neut_stacked = targets_pca_neut.stack()
        
        result = validator.validate(features_clean, targets_pca_neut_stacked, model_type=model_name)
        ic_pca = compute_ic_statistics(result['test_ic']) if len(result['test_ic']) > 0 else None
        
        if ic_pca:
            ic_p = ic_pca['mean_ic']
            retention_pca = (ic_p / ic_base * 100) if ic_base != 0 else np.nan
            pca_results[n_factors] = ic_p
            print(f"  IC={ic_p:7.4f} (was {ic_base:7.4f}), Retention={retention_pca:5.1f}%")
    
    # 4. Combined Neutralization
    print("\n" + "=" * 70)
    print("[4] COMBINED: MARKET + SECTOR")
    print("=" * 70)
    
    print("Applying market + sector neutralization...")
    targets_combined = neutralize_market_beta(targets, returns, market_returns, window=252)
    targets_combined = neutralize_sector(targets_combined, sector_map)
    targets_combined_stacked = targets_combined.stack()
    
    result = validator.validate(features_clean, targets_combined_stacked, model_type=model_name)
    ic_combined = compute_ic_statistics(result['test_ic']) if len(result['test_ic']) > 0 else None
    
    if ic_combined:
        ic_comb = ic_combined['mean_ic']
        retention_comb = (ic_comb / ic_base * 100) if ic_base != 0 else np.nan
        print(f"IC={ic_comb:7.4f} (was {ic_base:7.4f}), Retention={retention_comb:5.1f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: IC RETENTION AFTER NEUTRALIZATION")
    print("=" * 70)
    
    print(f"\n  Baseline:        {ic_base:7.4f} (100.0%)")
    if ic_market:
        print(f"  Market-neutral:  {ic_mkt:7.4f} ({retention_mkt:5.1f}%)")
    if ic_sector:
        print(f"  Sector-neutral:  {ic_sec:7.4f} ({retention_sec:5.1f}%)")
    if pca_results:
        for n_f, ic_p in pca_results.items():
            print(f"  PCA-{n_f} neutral:  {ic_p:7.4f} ({ic_p/ic_base*100:5.1f}%)")
    if ic_combined:
        print(f"  Combined:        {ic_comb:7.4f} ({retention_comb:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    
    avg_retention = (retention_mkt + retention_sec) / 2 if ic_market and ic_sector else 0
    
    if avg_retention > 80:
        print("✓ Signal is ROBUST: >80% IC retention after neutralization")
        print("  → Indicates genuine idiosyncratic alpha")
        print("  → Not driven by market beta or sector tilts")
    elif avg_retention > 50:
        print("⚠ Signal is MODERATE: 50-80% IC retention")
        print("  → Partial exposure to common factors")
        print("  → Some idiosyncratic component remains")
    else:
        print("✗ Signal is WEAK: <50% IC retention")
        print("  → Primarily driven by factor exposures")
        print("  → Limited idiosyncratic alpha")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
