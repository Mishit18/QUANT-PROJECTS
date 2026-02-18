import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.walkforward import WalkForwardValidator
from src.evaluation.ic import compute_ic_statistics
from src.utils.plotting import plot_ic_series


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    features = pd.read_parquet('data/features/features.parquet')
    targets = pd.read_parquet('data/processed/targets.parquet')
    
    # Drop features with >90% NaN
    nan_pct = features.isna().mean()
    valid_features = nan_pct[nan_pct < 0.9].index
    features = features[valid_features]
    print(f"Using {len(valid_features)} features (dropped {len(nan_pct) - len(valid_features)} with >90% NaN)")
    
    # Drop rows with >80% NaN
    features = features.dropna(thresh=features.shape[1] * 0.2)
    
    validator = WalkForwardValidator(config)
    
    models = ['ols', 'ridge', 'xgboost']
    results = {}
    
    print("models:")
    for model_name in models:
        result = validator.validate(features, targets, model_type=model_name)
        results[model_name] = result
        
        if len(result['test_ic']) > 0:
            ic_stats = compute_ic_statistics(result['test_ic'])
            print(f"{model_name}: IC={ic_stats['mean_ic']:.4f}, IR={ic_stats['ic_ir']:.2f}")
        else:
            print(f"{model_name}: IC=nan, IR=nan")
        
        output_dir = Path('reports/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if len(result['test_ic']) > 0:
            plot_ic_series(result['test_ic'], output_dir / f'{model_name}_ic.png')
    
    for model_name, result in results.items():
        pred = result['predictions']
        if len(pred) > 0:
            if isinstance(pred, pd.Series):
                pred.to_frame('prediction').to_parquet(f'data/processed/predictions_{model_name}.parquet')
            else:
                pred.to_parquet(f'data/processed/predictions_{model_name}.parquet')


if __name__ == '__main__':
    main()
