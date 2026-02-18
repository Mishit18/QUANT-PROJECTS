import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data.targets import construct_targets
from src.backtest.walkforward import WalkForwardValidator
from src.evaluation.ic import compute_ic_statistics


def evaluate_target_method(features: pd.DataFrame, targets: pd.DataFrame, 
                           config: dict, method_name: str) -> dict:
    """Evaluate IC for a given target method."""
    validator = WalkForwardValidator(config)
    
    # Drop features with >90% NaN
    nan_pct = features.isna().mean()
    valid_features = nan_pct[nan_pct < 0.9].index
    features_clean = features[valid_features]
    features_clean = features_clean.dropna(thresh=features_clean.shape[1] * 0.2)
    
    results = {}
    for model_name in ['ols', 'ridge', 'xgboost']:
        result = validator.validate(features_clean, targets, model_type=model_name)
        
        if len(result['test_ic']) > 0:
            ic_stats = compute_ic_statistics(result['test_ic'])
            results[model_name] = {
                'ic': ic_stats['mean_ic'],
                'ir': ic_stats['ic_ir'],
                'hit_rate': ic_stats['hit_rate'],
                'ic_std': ic_stats['std_ic'],
                'n_periods': len(result['test_ic'])
            }
        else:
            results[model_name] = {
                'ic': np.nan,
                'ir': np.nan,
                'hit_rate': np.nan,
                'ic_std': np.nan,
                'n_periods': 0
            }
    
    return results


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    features = pd.read_parquet('data/features/features.parquet')
    prices = pd.read_parquet('data/processed/prices.parquet')
    returns = pd.read_parquet('data/processed/returns.parquet')
    
    print("=" * 60)
    print("TARGET EVALUATION: RAW vs RANKED")
    print("=" * 60)
    
    # Test 1: Raw returns (old method)
    print("\n[1] RAW RETURNS (baseline)")
    config_raw = config.copy()
    config_raw['targets'] = {'horizon': 5, 'method': 'raw'}
    targets_raw, stats_raw = construct_targets(prices, config_raw)
    targets_raw_stacked = targets_raw.stack()
    
    print(f"  Range: [{stats_raw['target_min']:.4f}, {stats_raw['target_max']:.4f}]")
    print(f"  Mean: {stats_raw['target_mean']:.6f}, Std: {stats_raw['target_std']:.6f}")
    print(f"  CS std: {stats_raw['mean_cs_std']:.6f}")
    
    results_raw = evaluate_target_method(features, targets_raw_stacked, config, 'raw')
    for model, metrics in results_raw.items():
        print(f"  {model}: IC={metrics['ic']:.4f}, IR={metrics['ir']:.2f}, hit={metrics['hit_rate']:.2%}")
    
    # Test 2: Ranked returns (new method)
    print("\n[2] CROSS-SECTIONAL RANKED (new)")
    config_rank = config.copy()
    config_rank['targets'] = {'horizon': 5, 'method': 'rank', 'rank_scale': 0.5}
    targets_rank, stats_rank = construct_targets(prices, config_rank)
    targets_rank_stacked = targets_rank.stack()
    
    print(f"  Range: [{stats_rank['target_min']:.4f}, {stats_rank['target_max']:.4f}]")
    print(f"  Mean: {stats_rank['target_mean']:.6f}, Std: {stats_rank['target_std']:.6f}")
    print(f"  CS std: {stats_rank['mean_cs_std']:.6f}")
    
    results_rank = evaluate_target_method(features, targets_rank_stacked, config, 'rank')
    for model, metrics in results_rank.items():
        print(f"  {model}: IC={metrics['ic']:.4f}, IR={metrics['ir']:.2f}, hit={metrics['hit_rate']:.2%}")
    
    # Test 3: Vol-scaled returns
    print("\n[3] VOLATILITY-SCALED")
    config_vol = config.copy()
    config_vol['targets'] = {'horizon': 5, 'method': 'vol_scaled', 'vol_window': 20}
    targets_vol, stats_vol = construct_targets(prices, config_vol)
    targets_vol_stacked = targets_vol.stack()
    
    print(f"  Range: [{stats_vol['target_min']:.4f}, {stats_vol['target_max']:.4f}]")
    print(f"  Mean: {stats_vol['target_mean']:.6f}, Std: {stats_vol['target_std']:.6f}")
    print(f"  CS std: {stats_vol['mean_cs_std']:.6f}")
    
    results_vol = evaluate_target_method(features, targets_vol_stacked, config, 'vol_scaled')
    for model, metrics in results_vol.items():
        print(f"  {model}: IC={metrics['ic']:.4f}, IR={metrics['ir']:.2f}, hit={metrics['hit_rate']:.2%}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: IC IMPROVEMENT")
    print("=" * 60)
    
    for model in ['ols', 'ridge', 'xgboost']:
        ic_raw = results_raw[model]['ic']
        ic_rank = results_rank[model]['ic']
        ic_vol = results_vol[model]['ic']
        
        improvement_rank = ((ic_rank - ic_raw) / abs(ic_raw) * 100) if not np.isnan(ic_raw) and ic_raw != 0 else np.nan
        improvement_vol = ((ic_vol - ic_raw) / abs(ic_raw) * 100) if not np.isnan(ic_raw) and ic_raw != 0 else np.nan
        
        print(f"\n{model.upper()}:")
        print(f"  Raw:    IC={ic_raw:.4f}")
        print(f"  Ranked: IC={ic_rank:.4f} ({improvement_rank:+.1f}%)")
        print(f"  VolScl: IC={ic_vol:.4f} ({improvement_vol:+.1f}%)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
