"""
Main execution script for Statistical Arbitrage Research Project.

This script runs the complete end-to-end pipeline:
1. Data download and preprocessing
2. Cointegration testing and pair selection
3. Kalman Filter spread modeling
4. Signal generation
5. Backtesting
6. Performance analysis
7. Sensitivity testing

Usage:
    python main.py --mode full
    python main.py --mode backtest --pair SPY IWM
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from data_pipeline import DataPipeline
from cointegration_tests import CointegrationAnalyzer
from kalman_filter import KalmanSpreadModel
from spread_metrics import SpreadAnalyzer
from signal_generation import SignalGenerator
from backtest_engine import BacktestEngine
from performance_metrics import PerformanceAnalyzer
from sensitivity_analysis import SensitivityAnalyzer
from utils import load_config, ensure_dir, setup_logging


def run_data_pipeline(config):
    """Run data download and preprocessing pipeline."""
    logger.info("="*60)
    logger.info("STEP 1: DATA PIPELINE")
    logger.info("="*60)
    
    pipeline = DataPipeline()
    
    # Download data
    tickers = config['data']['universe']
    start_date = config['data']['start_date']
    end_date = config['data']['end_date']
    
    logger.info(f"Downloading data for {len(tickers)} tickers...")
    prices = pipeline.download_data(tickers, start_date, end_date)
    
    # Preprocess
    logger.info("Preprocessing data...")
    processed_prices = pipeline.preprocess()
    
    # Train/test split
    logger.info("Splitting into train/test sets...")
    train_prices, test_prices = pipeline.train_test_split(
        train_ratio=config['data']['train_test_split']
    )
    
    logger.info(f"✓ Data pipeline complete")
    logger.info(f"  Training: {len(train_prices)} observations")
    logger.info(f"  Testing: {len(test_prices)} observations")
    
    return train_prices, test_prices


def run_cointegration_analysis(train_prices, config):
    """Run cointegration testing and pair selection."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: COINTEGRATION ANALYSIS")
    logger.info("="*60)
    
    analyzer = CointegrationAnalyzer(
        significance_level=config['cointegration']['johansen']['significance_level']
    )
    
    # Scan pairs
    logger.info("Scanning pairs for cointegration...")
    results = analyzer.scan_pairs(
        train_prices,
        method='johansen',
        min_correlation=config['cointegration']['selection']['min_correlation']
    )
    
    # Rank candidates
    logger.info("Ranking cointegrated pairs...")
    ranked_pairs = analyzer.rank_candidates(results, method='johansen', top_n=10)
    
    if len(ranked_pairs) > 0:
        logger.info(f"✓ Found {len(ranked_pairs)} cointegrated pairs")
        logger.info("\nTop 5 pairs:")
        for i, row in ranked_pairs.head().iterrows():
            logger.info(f"  {i+1}. {row['pair']}: trace={row['trace_stat']:.2f}")
        
        # Save results
        ensure_dir("results/diagnostics")
        ranked_pairs.to_csv("results/diagnostics/cointegrated_pairs.csv", index=False)
    else:
        logger.warning("No cointegrated pairs found!")
    
    return ranked_pairs


def run_backtest(pair, train_prices, test_prices, config, use_test_data=True):
    """Run backtest for a specific pair."""
    logger.info("\n" + "="*60)
    logger.info(f"STEP 3: BACKTESTING PAIR {pair}")
    logger.info("="*60)
    
    ticker_y, ticker_x = pair
    
    # Select data
    prices = test_prices if use_test_data else train_prices
    price_y = prices[ticker_y]
    price_x = prices[ticker_x]
    
    logger.info(f"Backtesting {ticker_y}-{ticker_x} on {'test' if use_test_data else 'train'} data...")
    
    # Initialize backtest engine
    engine = BacktestEngine(config_path="config/strategy_config.yaml")
    
    # Run backtest
    results = engine.run(pair, price_y, price_x)
    
    # Analyze performance
    perf_analyzer = PerformanceAnalyzer(
        risk_free_rate=config['performance']['risk_free_rate'],
        periods_per_year=config['performance']['trading_days_per_year']
    )
    
    metrics = perf_analyzer.compute_metrics(results)
    
    # Print summary
    logger.info("\n" + "-"*60)
    logger.info("BACKTEST RESULTS")
    logger.info("-"*60)
    perf_analyzer.print_summary(metrics)
    
    # Save results
    ensure_dir("results/backtests")
    engine.generate_report(results, output_dir="results/backtests")
    
    # Plot equity curve
    plot_equity_curve(results, pair)
    
    return results, metrics


def plot_equity_curve(results, pair):
    """Plot equity curve and drawdown."""
    ensure_dir("reports/figures")
    
    equity = results['equity_curve']['equity']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Equity curve
    axes[0].plot(equity.index, equity, linewidth=2, color='blue')
    axes[0].set_ylabel('Equity ($)', fontsize=12)
    axes[0].set_title(f'Equity Curve: {pair[0]}-{pair[1]}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(results['final_capital'], color='green', linestyle='--', alpha=0.5)
    
    # Drawdown
    cummax = equity.expanding().max()
    drawdown = (equity - cummax) / cummax
    
    axes[1].fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3)
    axes[1].plot(drawdown.index, drawdown, linewidth=1, color='darkred')
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('Drawdown', fontsize=12)
    axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"reports/figures/equity_curve_{pair[0]}_{pair[1]}.png", dpi=300)
    plt.close()
    
    logger.info(f"✓ Saved equity curve plot")


def run_sensitivity_analysis(pair, train_prices, test_prices, config):
    """Run sensitivity analysis."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: SENSITIVITY ANALYSIS")
    logger.info("="*60)
    
    ticker_y, ticker_x = pair
    price_y = test_prices[ticker_y]
    price_x = test_prices[ticker_x]
    
    analyzer = SensitivityAnalyzer(config_path="config/strategy_config.yaml")
    
    # Parameter sweep
    logger.info("Running parameter sweep...")
    parameter_grid = {
        'entry_threshold': [1.5, 2.0, 2.5],
        'exit_threshold': [0.0, 0.5, 1.0],
        'lookback_window': [30, 60, 90]
    }
    
    sweep_results = analyzer.parameter_sweep(
        pair, price_y, price_x, parameter_grid
    )
    
    if len(sweep_results) > 0:
        logger.info(f"✓ Parameter sweep complete: {len(sweep_results)} combinations tested")
        
        # Find best parameters
        best_idx = sweep_results['sharpe_ratio'].idxmax()
        best_params = sweep_results.loc[best_idx]
        
        logger.info("\nBest parameters:")
        logger.info(f"  Entry threshold: {best_params['entry_threshold']}")
        logger.info(f"  Exit threshold: {best_params['exit_threshold']}")
        logger.info(f"  Lookback window: {best_params['lookback_window']}")
        logger.info(f"  Sharpe ratio: {best_params['sharpe_ratio']:.2f}")
        
        # Save results
        ensure_dir("results/diagnostics")
        sweep_results.to_csv("results/diagnostics/parameter_sweep.csv", index=False)
    
    return sweep_results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Statistical Arbitrage Research Project')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'data', 'coint', 'backtest', 'sensitivity'],
                       help='Execution mode')
    parser.add_argument('--pair', nargs=2, type=str, default=None,
                       help='Specific pair to backtest (e.g., SPY IWM)')
    parser.add_argument('--config', type=str, default='config/strategy_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("STATISTICAL ARBITRAGE RESEARCH PROJECT")
    logger.info("Cointegration-Based Pairs Trading")
    logger.info("="*60)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    for dir_path in ['data/raw', 'data/processed', 'results/backtests', 
                     'results/diagnostics', 'results/signals', 'reports/figures']:
        ensure_dir(dir_path)
    
    try:
        if args.mode in ['full', 'data']:
            train_prices, test_prices = run_data_pipeline(config)
        else:
            # Load existing data
            train_prices = pd.read_parquet('data/processed/train_prices.parquet')
            test_prices = pd.read_parquet('data/processed/test_prices.parquet')
        
        if args.mode in ['full', 'coint']:
            ranked_pairs = run_cointegration_analysis(train_prices, config)
        else:
            # Load existing results
            ranked_pairs = pd.read_csv('results/diagnostics/cointegrated_pairs.csv')
        
        if args.mode in ['full', 'backtest']:
            if args.pair:
                # Backtest specific pair
                pair = tuple(args.pair)
            else:
                # Backtest top pair
                if len(ranked_pairs) > 0:
                    pair_value = ranked_pairs.iloc[0]['pair']
                    # Handle both tuple and string representations
                    if isinstance(pair_value, tuple):
                        pair = pair_value
                    elif isinstance(pair_value, str):
                        pair = eval(pair_value)
                    else:
                        logger.error(f"Unexpected pair type: {type(pair_value)}")
                        return
                else:
                    logger.error("No pairs available for backtesting")
                    return
            
            results, metrics = run_backtest(pair, train_prices, test_prices, config)
        
        if args.mode in ['full', 'sensitivity']:
            if args.pair:
                pair = tuple(args.pair)
            else:
                if len(ranked_pairs) > 0:
                    pair_value = ranked_pairs.iloc[0]['pair']
                    # Handle both tuple and string representations
                    if isinstance(pair_value, tuple):
                        pair = pair_value
                    elif isinstance(pair_value, str):
                        pair = eval(pair_value)
                    else:
                        logger.error(f"Unexpected pair type: {type(pair_value)}")
                        return
                else:
                    logger.error("No pairs available for sensitivity analysis")
                    return
            
            sweep_results = run_sensitivity_analysis(pair, train_prices, test_prices, config)
        
        logger.info("\n" + "="*60)
        logger.info("✓ EXECUTION COMPLETE")
        logger.info("="*60)
        logger.info("\nResults saved to:")
        logger.info("  - results/backtests/")
        logger.info("  - results/diagnostics/")
        logger.info("  - reports/figures/")
        logger.info("\nNext steps:")
        logger.info("  1. Review backtest results in results/backtests/")
        logger.info("  2. Examine plots in reports/figures/")
        logger.info("  3. Run Jupyter notebooks for detailed analysis")
        logger.info("  4. Read research report in reports/stat_arb_research_report.md")
        
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
