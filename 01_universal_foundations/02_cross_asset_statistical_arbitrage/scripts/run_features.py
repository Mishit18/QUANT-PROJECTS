import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.data.panel_builder import PanelBuilder
from src.data.targets import construct_targets, validate_target_alignment
from src.features.pipeline import FeaturePipeline


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    data_dir = Path('data/raw')
    output_dir = Path('data/features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    universe_file = data_dir / 'universe.csv'
    if universe_file.exists():
        tickers = pd.read_csv(universe_file)['ticker'].tolist()
    else:
        tickers = [f.stem for f in data_dir.glob('*.csv') if f.stem != 'universe']
    
    builder = PanelBuilder(data_dir, config)
    prices, volumes, returns = builder.build(tickers)
    
    processed_dir = Path('data/processed')
    processed_dir.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(processed_dir / 'prices.parquet')
    volumes.to_parquet(processed_dir / 'volumes.parquet')
    returns.to_parquet(processed_dir / 'returns.parquet')
    
    # Construct targets with proper cross-sectional ranking
    market_returns = returns.mean(axis=1)
    targets, target_stats = construct_targets(prices, config, market_returns)
    
    # Stack to (date, asset) MultiIndex
    targets_stacked = targets.stack()
    targets_stacked.name = 'target'
    targets_stacked.to_frame().to_parquet(processed_dir / 'targets.parquet')
    
    print(f"Target: {target_stats['method']}, horizon={target_stats['horizon']}d")
    print(f"CS std: {target_stats['mean_cs_std']:.4f}, valid/date: {target_stats['mean_cs_valid']:.1f}")
    
    ohlc = builder.get_ohlc(tickers)
    
    pipeline = FeaturePipeline(config_path)
    features = pipeline.compute_all_features(prices, volumes, ohlc)
    
    # Validate alignment
    alignment = validate_target_alignment(features, targets)
    print(f"Alignment: {alignment['common_dates']}/{alignment['feature_dates']} dates ({alignment['alignment_pct']:.1f}%)")
    
    pipeline.save_features(output_dir / 'features.parquet')
    print(f"Features: {features.shape[1]} computed")


if __name__ == '__main__':
    main()
