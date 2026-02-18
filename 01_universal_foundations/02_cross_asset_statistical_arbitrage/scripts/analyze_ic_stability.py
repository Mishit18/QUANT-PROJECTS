import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.walkforward import WalkForwardValidator
from src.evaluation.ic import compute_ic_statistics


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    features = pd.read_parquet('data/features/features.parquet')
    targets = pd.read_parquet('data/processed/targets.parquet')
    
    # Clean features
    nan_pct = features.isna().mean()
    valid_features = nan_pct[nan_pct < 0.9].index
    features = features[valid_features]
    features = features.dropna(thresh=features.shape[1] * 0.2)
    
    validator = WalkForwardValidator(config)
    
    print("=" * 60)
    print("IC STABILITY ANALYSIS")
    print("=" * 60)
    
    for model_name in ['ols', 'ridge', 'xgboost']:
        result = validator.validate(features, targets, model_type=model_name)
        
        if len(result['test_ic']) > 0:
            ic_series = result['test_ic']
            ic_stats = compute_ic_statistics(ic_series)
            
            print(f"\n{model_name.upper()}:")
            print(f"  Mean IC:     {ic_stats['mean_ic']:.4f}")
            print(f"  Std IC:      {ic_stats['std_ic']:.4f}")
            print(f"  IC-IR:       {ic_stats['ic_ir']:.2f}")
            print(f"  Hit Rate:    {ic_stats['hit_rate']:.2%}")
            print(f"  Skew:        {ic_stats['skew']:.2f}")
            print(f"  Kurtosis:    {ic_stats['kurtosis']:.2f}")
            print(f"  N Periods:   {len(ic_series)}")
            
            # Stability metrics
            rolling_ic = ic_series.rolling(12).mean()
            ic_volatility = ic_series.std()
            ic_range = ic_series.max() - ic_series.min()
            
            print(f"  IC Range:    [{ic_series.min():.4f}, {ic_series.max():.4f}]")
            print(f"  IC Vol:      {ic_volatility:.4f}")
            print(f"  12-period rolling IC range: [{rolling_ic.min():.4f}, {rolling_ic.max():.4f}]")
            
            # Consistency
            positive_periods = (ic_series > 0).sum()
            consistency = positive_periods / len(ic_series)
            print(f"  Consistency: {consistency:.2%} ({positive_periods}/{len(ic_series)} periods)")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
