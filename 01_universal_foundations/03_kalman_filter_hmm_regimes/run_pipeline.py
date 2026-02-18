"""
Main execution script for Kalman Filter + HMM trading strategy.

Usage:
    python run_pipeline.py
    python run_pipeline.py --config configs/custom_config.yaml
    python run_pipeline.py --tickers SPY QQQ --n-regimes 3
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

from src.data_loader import load_market_data, load_sample_data
from src.preprocessing import preprocess_data
from src.state_space_models import LocalLevelModel, LocalLinearTrendModel
from src.kalman_filter import KalmanFilter
from src.hmm_regimes import GaussianHMM
from src.signals import create_regime_aware_strategy
from src.backtest import Backtest, compare_strategies
from src.evaluation import StrategyEvaluator
from src.visualization import *


def load_config(config_path='configs/default_config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """Create output directories if they don't exist."""
    os.makedirs(config['visualization']['figure_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    if config['output']['save_models']:
        os.makedirs(config['output']['models_dir'], exist_ok=True)


def main(config):
    """Run complete pipeline."""
    
    print("="*80)
    print("KALMAN FILTER + HMM REGIME SWITCHING STRATEGY")
    print("="*80)
    
    # Setup
    setup_directories(config)
    
    # 1. Load data
    print("\n[1/7] Loading market data...")
    try:
        data = load_market_data(
            tickers=config['data']['tickers'],
            start_date=config['data'].get('start_date'),
            end_date=config['data'].get('end_date')
        )
        
        # Extract primary ticker
        primary_ticker = config['data']['tickers'][0]
        if isinstance(data, dict):
            returns = data['returns'][primary_ticker].values
            prices = data['prices'][primary_ticker].values
        else:
            # Use sample data instead if format is problematic
            raise ValueError("Data format issue, using sample data")
            
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using sample data instead...")
        data = load_sample_data()
        
        # Handle synthetic data format
        if 'returns' in data and 'prices' in data:
            # Get first column from returns
            returns = data['returns'].iloc[:, 0].values
            prices = data['prices'].iloc[:, 0].values
            primary_ticker = data['returns'].columns[0]
        else:
            raise ValueError("Invalid data format")
    
    print(f"Loaded {len(returns)} observations")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Volatility: {np.std(returns) * np.sqrt(252):.2%}")
    
    # 2. Fit Kalman Filter
    print("\n[2/7] Fitting Kalman Filter...")
    
    if config['kalman_filter']['model_type'] == 'local_level':
        model = LocalLevelModel(
            observation_variance=config['kalman_filter']['observation_variance'],
            state_variance=config['kalman_filter']['state_variance'],
            initial_state_variance=config['kalman_filter']['initial_state_variance']
        )
    elif config['kalman_filter']['model_type'] == 'local_linear_trend':
        model = LocalLinearTrendModel(
            observation_variance=config['kalman_filter']['observation_variance'],
            level_variance=config['kalman_filter']['state_variance'],
            slope_variance=config['kalman_filter']['state_variance'] * 0.1
        )
    else:
        raise ValueError(f"Unknown model type: {config['kalman_filter']['model_type']}")
    
    kf = KalmanFilter(model)
    filtered_states, smoothed_states = kf.filter_and_smooth(returns)
    
    print(f"Log-likelihood: {kf.get_log_likelihood():.2f}")
    
    # Diagnostics
    diagnostics = kf.diagnose()
    print(f"Innovation autocorrelation (lag 1): {diagnostics['innovation_autocorr'][1]:.3f}")
    
    # 3. Fit HMM
    print("\n[3/7] Fitting HMM for regime detection...")
    
    hmm = GaussianHMM(
        n_regimes=config['hmm']['n_regimes'],
        n_iter=config['hmm']['n_iter'],
        tol=config['hmm']['tolerance'],
        random_state=config['hmm']['random_state']
    )
    hmm.fit(returns)
    
    regime_probs = hmm.predict_proba(returns, method='smoothed')
    regimes = hmm.predict(returns)
    
    print(f"Converged in {len(hmm.log_likelihoods)} iterations")
    
    # Regime statistics
    stats = hmm.get_regime_statistics(returns)
    print("\nRegime Statistics:")
    for k in range(config['hmm']['n_regimes']):
        print(f"  Regime {k}: Mean={stats['regime_statistics'][k]['mean'][0]:.4f}, "
              f"Freq={stats['regime_statistics'][k]['frequency']:.2%}, "
              f"Duration={stats['expected_duration'][k]:.1f} days")
    
    # 4. Generate signals
    print("\n[4/7] Generating trading signals...")
    
    signals = create_regime_aware_strategy(
        returns,
        kf,
        hmm,
        vol_target=config['signals']['vol_target']
    )
    
    print(f"Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"Long positions: {(signals > 0.1).sum()} ({(signals > 0.1).sum()/len(signals):.1%})")
    print(f"Short positions: {(signals < -0.1).sum()} ({(signals < -0.1).sum()/len(signals):.1%})")
    
    # 5. Backtest
    print("\n[5/7] Running backtest...")
    
    bt = Backtest(
        signals=signals,
        returns=returns,
        transaction_cost=config['backtest']['transaction_cost'],
        leverage=config['backtest']['leverage'],
        initial_capital=config['backtest']['initial_capital']
    )
    
    results = bt.run()
    
    print("\nBacktest Results:")
    print(f"  Total Return: {results['total_return']:.2%}")
    print(f"  Annualized Return: {results['annualized_return']:.2%}")
    print(f"  Volatility: {results['volatility']:.2%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {results['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"  Win Rate: {results['win_rate']:.2%}")
    
    # 6. Comparison with baselines
    print("\n[6/7] Comparing with baseline strategies...")
    
    # Buy and hold
    bh_signals = np.ones(len(returns))
    bt_bh = Backtest(bh_signals, returns, transaction_cost=config['backtest']['transaction_cost'])
    bt_bh.run()
    
    # Simple trend
    kf_signals = np.sign(filtered_states.flatten())
    bt_kf = Backtest(kf_signals, returns, transaction_cost=config['backtest']['transaction_cost'])
    bt_kf.run()
    
    strategies = {
        'Buy & Hold': bt_bh,
        'Kalman Trend': bt_kf,
        'Regime-Aware': bt
    }
    
    comparison = compare_strategies(strategies)
    print("\nStrategy Comparison:")
    print(comparison[['annualized_return', 'volatility', 'sharpe_ratio', 'max_drawdown']])
    
    # 7. Visualization
    if config['visualization']['save_figures']:
        print("\n[7/7] Generating visualizations...")
        
        fig_dir = config['visualization']['figure_dir']
        
        # Kalman filter results
        plot_kalman_filter_results(
            returns, filtered_states, smoothed_states,
            title="Kalman Filter: Trend Extraction",
            save_path=os.path.join(fig_dir, 'kalman_trend.png')
        )
        
        # Regime probabilities
        plot_regime_probabilities(
            regime_probs, returns,
            title="HMM Regime Probabilities",
            save_path=os.path.join(fig_dir, 'regime_probabilities.png')
        )
        
        # Transition matrix
        plot_regime_transition_matrix(
            stats['transition_matrix'],
            title="Regime Transition Matrix",
            save_path=os.path.join(fig_dir, 'transition_matrix.png')
        )
        
        # Equity curve
        equity_curve = bt.get_equity_curve().values
        plot_equity_curve(
            equity_curve,
            title="Strategy Equity Curve",
            save_path=os.path.join(fig_dir, 'equity_curve.png')
        )
        
        # Strategy comparison
        equity_curves = {
            name: bt_obj.get_equity_curve().values
            for name, bt_obj in strategies.items()
        }
        plot_performance_comparison(
            equity_curves,
            title="Strategy Performance Comparison",
            save_path=os.path.join(fig_dir, 'strategy_comparison.png')
        )
        
        # Dashboard
        create_summary_dashboard(
            returns, equity_curve, regime_probs, filtered_states,
            save_path=os.path.join(fig_dir, 'summary_dashboard.png')
        )
        
        print(f"Figures saved to {fig_dir}/")
    
    # Save results
    if config['output']['save_results']:
        results_dir = config['output']['results_dir']
        
        # Save performance metrics
        results_df = pd.DataFrame([results])
        results_df.to_csv(os.path.join(results_dir, 'performance_metrics.csv'), index=False)
        
        # Save comparison
        comparison.to_csv(os.path.join(results_dir, 'strategy_comparison.csv'))
        
        # Save regime statistics
        regime_stats_df = pd.DataFrame(stats['regime_statistics'])
        regime_stats_df.to_csv(os.path.join(results_dir, 'regime_statistics.csv'), index=False)
        
        print(f"\nResults saved to {results_dir}/")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    
    return {
        'kf': kf,
        'hmm': hmm,
        'signals': signals,
        'backtest': bt,
        'results': results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Kalman Filter + HMM trading strategy')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--tickers', nargs='+', help='Override tickers')
    parser.add_argument('--n-regimes', type=int, help='Override number of regimes')
    parser.add_argument('--transaction-cost', type=float, help='Override transaction cost')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command-line arguments
    if args.tickers:
        config['data']['tickers'] = args.tickers
    if args.n_regimes:
        config['hmm']['n_regimes'] = args.n_regimes
    if args.transaction_cost:
        config['backtest']['transaction_cost'] = args.transaction_cost
    
    # Run pipeline
    try:
        output = main(config)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
