import sys
from pathlib import Path
import yaml
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.backtest.engine import BacktestEngine
from src.neutralization.residuals import RiskNeutralizer
from src.utils.plotting import plot_equity_curve
from src.robustness.reality_check import whites_reality_check


def main():
    config_path = Path('src/config/config.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    pred_file = Path('data/processed/predictions_xgboost.parquet')
    if not pred_file.exists():
        print("No predictions found")
        return
    
    predictions = pd.read_parquet(pred_file)
    
    if len(predictions) == 0:
        print("Empty predictions")
        return
    
    returns = pd.read_parquet('data/processed/returns.parquet')
    volumes = pd.read_parquet('data/processed/volumes.parquet')
    
    predictions_wide = predictions.unstack() if predictions.index.nlevels > 1 else predictions
    
    if len(predictions_wide) == 0:
        return
    
    neutralizer = RiskNeutralizer(config)
    neutralized_alpha = neutralizer.neutralize(predictions_wide, returns)
    
    engine = BacktestEngine(config)
    backtest_results = engine.run(neutralized_alpha, returns, volumes)
    
    metrics = engine.compute_metrics(backtest_results['returns'])
    
    print(f"Sharpe: {metrics['sharpe']:.2f}")
    print(f"Sortino: {metrics['sortino']:.2f}")
    print(f"Max DD: {metrics['max_drawdown']:.2%}")
    print(f"Return: {metrics['total_return']:.2%}")
    
    output_dir = Path('reports/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_equity_curve(backtest_results['returns'], output_dir / 'equity_curve.png')
    
    backtest_results['returns'].to_csv('data/processed/backtest_returns.csv')


if __name__ == '__main__':
    main()
