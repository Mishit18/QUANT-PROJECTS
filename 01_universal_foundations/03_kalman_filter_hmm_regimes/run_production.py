"""
PRODUCTION-GRADE EXECUTION SCRIPT

This script implements hedge-fund-level execution standards:
- Real data enforcement (no synthetic fallback)
- Comprehensive validation at every stage
- Structured logging to file and console
- Explicit error handling with context
- Graceful degradation where appropriate
- Complete audit trail

Usage:
    python run_production.py
    python run_production.py --config configs/default_config.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any

from src.data_loader import load_market_data, load_sample_data
from src.preprocessing import preprocess_data
from src.state_space_models import LocalLevelModel, LocalLinearTrendModel
from src.kalman_filter import KalmanFilter
from src.hmm_regimes import GaussianHMM
from src.signals import create_regime_aware_strategy
from src.backtest import Backtest, compare_strategies
from src.evaluation import StrategyEvaluator
from src.visualization import *


class ProductionValidator:
    """
    Validates data and model outputs at each pipeline stage.
    """
    
    @staticmethod
    def validate_data(data: Dict[str, pd.DataFrame], stage: str) -> None:
        """Validate data quality."""
        if 'returns' not in data or 'prices' not in data:
            raise ValueError(f"[{stage}] Missing required data fields")
        
        returns = data['returns']
        prices = data['prices']
        
        # Check for NaNs
        if returns.isna().any().any():
            raise ValueError(f"[{stage}] NaN values in returns")
        
        if prices.isna().any().any():
            raise ValueError(f"[{stage}] NaN values in prices")
        
        # Check for Infs
        if np.isinf(returns.values).any():
            raise ValueError(f"[{stage}] Inf values in returns")
        
        # Check for extreme values
        if (returns.abs() > 0.5).any().any():
            raise ValueError(f"[{stage}] Extreme returns detected (>50% single day)")
        
        # Check minimum length
        if len(returns) < 252:
            raise ValueError(f"[{stage}] Insufficient data: {len(returns)} < 252 days")
    
    @staticmethod
    def validate_array(arr: np.ndarray, name: str, stage: str) -> None:
        """Validate numpy array."""
        if arr is None:
            raise ValueError(f"[{stage}] {name} is None")
        
        if np.any(np.isnan(arr)):
            raise ValueError(f"[{stage}] NaN in {name}")
        
        if np.any(np.isinf(arr)):
            raise ValueError(f"[{stage}] Inf in {name}")
    
    @staticmethod
    def validate_signals(signals: np.ndarray, stage: str, max_leverage: float = 1.5) -> None:
        """Validate trading signals."""
        ProductionValidator.validate_array(signals, "signals", stage)
        
        # Allow signals up to max_leverage with small tolerance
        if not np.all((signals >= -(max_leverage + 0.1)) & (signals <= (max_leverage + 0.1))):
            raise ValueError(f"[{stage}] Signals outside [-{max_leverage}, {max_leverage}] range: [{signals.min():.3f}, {signals.max():.3f}]")
        
        if np.all(signals == 0):
            logging.warning(f"[{stage}] All signals are zero")


def setup_logging(log_file: str = 'production_run.log') -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config(config_path: str = 'configs/default_config.yaml') -> Dict[str, Any]:
    """Load configuration with validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config


def setup_directories(config: Dict[str, Any]) -> None:
    """Create output directories."""
    os.makedirs(config['visualization']['figure_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    if config['output']['save_models']:
        os.makedirs(config['output']['models_dir'], exist_ok=True)
    
    logging.info("Output directories created")


def load_and_validate_data(config: Dict[str, Any]) -> tuple:
    """
    Load market data with production-grade validation.
    
    Returns
    -------
    returns : np.ndarray
        Return series
    prices : np.ndarray
        Price series
    primary_ticker : str
        Primary ticker symbol
    """
    logging.info("="*80)
    logging.info("STAGE 1: DATA LOADING")
    logging.info("="*80)
    
    try:
        # Attempt to load real market data
        logging.info(f"Loading data for {config['data']['tickers']}...")
        
        data = load_sample_data()  # Uses real data with 3 retry attempts
        
        # Validate data structure
        ProductionValidator.validate_data(data, "DATA_LOADING")
        
        # Extract primary ticker
        returns = data['returns'].iloc[:, 0].values
        prices = data['prices'].iloc[:, 0].values
        primary_ticker = data['returns'].columns[0]
        
        # Final validation
        ProductionValidator.validate_array(returns, "returns", "DATA_LOADING")
        ProductionValidator.validate_array(prices, "prices", "DATA_LOADING")
        
        logging.info(f"[OK] Loaded {len(returns)} observations for {primary_ticker}")
        logging.info(f"  Mean return: {np.mean(returns):.6f}")
        logging.info(f"  Volatility: {np.std(returns) * np.sqrt(252):.2%}")
        logging.info(f"  Min/Max: [{np.min(returns):.4f}, {np.max(returns):.4f}]")
        
        return returns, prices, primary_ticker
        
    except Exception as e:
        logging.error(f"✗ Data loading failed: {e}")
        raise RuntimeError(f"CRITICAL: Cannot proceed without real data. Error: {e}")


def fit_kalman_filter(returns: np.ndarray, config: Dict[str, Any]) -> KalmanFilter:
    """
    Fit Kalman filter with validation.
    """
    logging.info("="*80)
    logging.info("STAGE 2: KALMAN FILTER")
    logging.info("="*80)
    
    try:
        # Initialize model
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
        
        logging.info(f"Model: {config['kalman_filter']['model_type']}")
        
        # Fit filter
        kf = KalmanFilter(model)
        filtered_states, smoothed_states = kf.filter_and_smooth(returns)
        
        # Validate outputs
        ProductionValidator.validate_array(filtered_states, "filtered_states", "KALMAN_FILTER")
        ProductionValidator.validate_array(smoothed_states, "smoothed_states", "KALMAN_FILTER")
        
        # Diagnostics
        diagnostics = kf.diagnose()
        
        logging.info(f"[OK] Kalman filter fitted successfully")
        logging.info(f"  Log-likelihood: {kf.get_log_likelihood():.2f}")
        logging.info(f"  Innovation autocorr (lag 1): {diagnostics['innovation_autocorr'][1]:.3f}")
        
        return kf
        
    except Exception as e:
        logging.error(f"✗ Kalman filter failed: {e}")
        raise RuntimeError(f"Kalman filter fitting failed: {e}")


def fit_hmm(returns: np.ndarray, config: Dict[str, Any]) -> GaussianHMM:
    """
    Fit HMM with regime collapse detection.
    """
    logging.info("="*80)
    logging.info("STAGE 3: HMM REGIME DETECTION")
    logging.info("="*80)
    
    try:
        # Initialize HMM
        hmm = GaussianHMM(
            n_regimes=config['hmm']['n_regimes'],
            n_iter=config['hmm']['n_iter'],
            tol=config['hmm']['tolerance'],
            random_state=config['hmm']['random_state']
        )
        
        logging.info(f"Fitting HMM with {config['hmm']['n_regimes']} regimes...")
        
        # Fit model
        hmm.fit(returns)
        
        # Validate convergence
        if not hmm.is_fitted:
            raise RuntimeError("HMM failed to converge")
        
        # Get regime probabilities
        regime_probs = hmm.predict_proba(returns, method='smoothed')
        regimes = hmm.predict(returns)
        
        # Validate outputs
        ProductionValidator.validate_array(regime_probs, "regime_probs", "HMM")
        ProductionValidator.validate_array(regimes, "regimes", "HMM")
        
        # Check for regime collapse
        regime_counts = np.bincount(regimes, minlength=config['hmm']['n_regimes'])
        min_regime_freq = regime_counts.min() / len(regimes)
        
        if min_regime_freq < 0.05:
            logging.warning(f"[WARNING] Potential regime collapse: min frequency = {min_regime_freq:.2%}")
        
        # Regime statistics
        stats = hmm.get_regime_statistics(returns)
        
        logging.info(f"[OK] HMM fitted successfully")
        logging.info(f"  Converged in {len(hmm.log_likelihoods)} iterations")
        logging.info(f"  Final log-likelihood: {hmm.log_likelihoods[-1]:.2f}")
        
        logging.info("\n  Regime Statistics:")
        for k in range(config['hmm']['n_regimes']):
            regime_stat = stats['regime_statistics'][k]
            logging.info(f"    Regime {k}: "
                        f"Mean={regime_stat['mean'][0]:.4f}, "
                        f"Vol={regime_stat['std'][0]:.4f}, "
                        f"Freq={regime_stat['frequency']:.2%}, "
                        f"Duration={stats['expected_duration'][k]:.1f} days")
        
        return hmm
        
    except Exception as e:
        logging.error(f"✗ HMM fitting failed: {e}")
        raise RuntimeError(f"HMM fitting failed: {e}")


def generate_signals(returns: np.ndarray, kf: KalmanFilter, 
                    hmm: GaussianHMM, config: Dict[str, Any]) -> np.ndarray:
    """
    Generate trading signals with validation.
    
    Supports both baseline and enhanced signal generation.
    """
    logging.info("="*80)
    logging.info("STAGE 4: SIGNAL GENERATION")
    logging.info("="*80)
    
    try:
        # Check if enhanced mode is enabled
        use_enhanced = config['signals'].get('enhanced', {}).get('enabled', False)
        
        if use_enhanced:
            logging.info("Using ENHANCED signal generation with:")
            logging.info(f"  - Regime-conditioned position sizing")
            logging.info(f"  - Advanced volatility targeting")
            logging.info(f"  - Regime-gated signal activation")
            
            enhanced_config = config['signals']['enhanced']
            momentum_config = enhanced_config.get('momentum', {})
            enable_momentum = momentum_config.get('enabled', False)
            
            if enable_momentum:
                logging.info(f"  - TSMOM alpha source (lookbacks: {momentum_config.get('lookbacks', [63, 126, 252])})")
                logging.info(f"    Weight: {momentum_config.get('weight', 0.4):.0%} TSMOM, {1-momentum_config.get('weight', 0.4):.0%} Kalman")
            
            from src.signals_enhanced import create_enhanced_regime_strategy
            
            signals = create_enhanced_regime_strategy(
                returns,
                kf,
                hmm,
                vol_target=enhanced_config.get('vol_target_enhanced', 0.10),
                max_leverage=enhanced_config.get('max_leverage', 1.5),
                enable_regime_gating=enhanced_config.get('enable_regime_gating', True),
                enable_momentum=enable_momentum,
                momentum_weight=momentum_config.get('weight', 0.4)
            )
            
            logging.info(f"  Max leverage: {enhanced_config.get('max_leverage', 1.5):.2f}")
            logging.info(f"  Vol target: {enhanced_config.get('vol_target_enhanced', 0.10):.2%}")
        else:
            logging.info("Using BASELINE signal generation")
            
            signals = create_regime_aware_strategy(
                returns,
                kf,
                hmm,
                vol_target=config['signals']['vol_target']
            )
        
        # Validate signals
        max_lev = enhanced_config.get('max_leverage', 1.5) if use_enhanced else 1.0
        ProductionValidator.validate_signals(signals, "SIGNAL_GENERATION", max_leverage=max_lev)
        
        # Signal statistics
        long_pct = (signals > 0.1).sum() / len(signals)
        short_pct = (signals < -0.1).sum() / len(signals)
        neutral_pct = 1 - long_pct - short_pct
        
        logging.info(f"[OK] Signals generated successfully")
        logging.info(f"  Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
        logging.info(f"  Long: {long_pct:.1%}, Short: {short_pct:.1%}, Neutral: {neutral_pct:.1%}")
        logging.info(f"  Mean absolute signal: {np.abs(signals).mean():.3f}")
        
        return signals
        
    except Exception as e:
        logging.error(f"✗ Signal generation failed: {e}")
        raise RuntimeError(f"Signal generation failed: {e}")


def run_backtest(signals: np.ndarray, returns: np.ndarray, 
                config: Dict[str, Any]) -> tuple:
    """
    Run backtest with comparison strategies.
    """
    logging.info("="*80)
    logging.info("STAGE 5: BACKTESTING")
    logging.info("="*80)
    
    try:
        # Main strategy
        bt = Backtest(
            signals=signals,
            returns=returns,
            transaction_cost=config['backtest']['transaction_cost'],
            leverage=config['backtest']['leverage'],
            initial_capital=config['backtest']['initial_capital']
        )
        
        results = bt.run()
        
        logging.info(f"[OK] Backtest completed")
        logging.info(f"\n  Performance Metrics:")
        logging.info(f"    Total Return: {results['total_return']:.2%}")
        logging.info(f"    Annualized Return: {results['annualized_return']:.2%}")
        logging.info(f"    Volatility: {results['volatility']:.2%}")
        logging.info(f"    Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logging.info(f"    Sortino Ratio: {results['sortino_ratio']:.2f}")
        logging.info(f"    Max Drawdown: {results['max_drawdown']:.2%}")
        logging.info(f"    Win Rate: {results['win_rate']:.2%}")
        logging.info(f"    Avg Turnover: {results['avg_turnover']:.2%}")
        
        # Baseline strategies
        logging.info("\n  Comparing with baselines...")
        
        # Buy and hold
        bh_signals = np.ones(len(returns))
        bt_bh = Backtest(bh_signals, returns, 
                        transaction_cost=config['backtest']['transaction_cost'])
        bt_bh.run()
        
        # Simple Kalman trend
        from src.state_space_models import LocalLevelModel
        from src.kalman_filter import KalmanFilter
        
        model = LocalLevelModel()
        kf_simple = KalmanFilter(model)
        filtered, _ = kf_simple.filter_and_smooth(returns)
        kf_signals = np.sign(filtered.flatten())
        
        bt_kf = Backtest(kf_signals, returns,
                        transaction_cost=config['backtest']['transaction_cost'])
        bt_kf.run()
        
        strategies = {
            'Buy & Hold': bt_bh,
            'Kalman Trend': bt_kf,
            'Regime-Aware': bt
        }
        
        comparison = compare_strategies(strategies)
        
        logging.info("\n  Strategy Comparison:")
        for idx, row in comparison.iterrows():
            logging.info(f"    {idx}: Sharpe={row['sharpe_ratio']:.2f}, "
                        f"Return={row['annualized_return']:.2%}, "
                        f"MaxDD={row['max_drawdown']:.2%}")
        
        return bt, strategies, comparison, results
        
    except Exception as e:
        logging.error(f"✗ Backtesting failed: {e}")
        raise RuntimeError(f"Backtesting failed: {e}")


def generate_visualizations(returns: np.ndarray, kf: KalmanFilter, 
                           hmm: GaussianHMM, bt: Backtest,
                           strategies: Dict, config: Dict[str, Any]) -> None:
    """
    Generate and save all visualizations.
    """
    if not config['visualization']['save_figures']:
        return
    
    logging.info("="*80)
    logging.info("STAGE 6: VISUALIZATION")
    logging.info("="*80)
    
    try:
        fig_dir = config['visualization']['figure_dir']
        
        # Kalman filter results
        filtered_states = kf.filtered_states
        smoothed_states, _ = kf.smooth()
        
        plot_kalman_filter_results(
            returns, filtered_states, smoothed_states,
            title="Kalman Filter: Trend Extraction",
            save_path=os.path.join(fig_dir, 'kalman_trend.png')
        )
        
        # Regime probabilities
        regime_probs = hmm.predict_proba(returns, method='smoothed')
        
        plot_regime_probabilities(
            regime_probs, returns,
            title="HMM Regime Probabilities",
            save_path=os.path.join(fig_dir, 'regime_probabilities.png')
        )
        
        # Transition matrix
        stats = hmm.get_regime_statistics(returns)
        
        plot_regime_transition_matrix(
            stats['transition_matrix'],
            title="Regime Transition Matrix",
            save_path=os.path.join(fig_dir, 'transition_matrix.png')
        )
        
        # Signals and positions
        signals = bt.signals
        positions = bt.positions
        
        plot_signals_and_positions(
            returns, signals, positions,
            title="Trading Signals and Positions",
            save_path=os.path.join(fig_dir, 'signals_positions.png')
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
        
        logging.info(f"[OK] Generated 7 visualizations in {fig_dir}/")
        
    except Exception as e:
        logging.warning(f"[WARNING] Visualization generation failed: {e}")
        # Non-critical, continue


def save_results(results: Dict, comparison: pd.DataFrame, 
                hmm: GaussianHMM, returns: np.ndarray,
                config: Dict[str, Any]) -> None:
    """
    Save results to CSV files.
    """
    if not config['output']['save_results']:
        return
    
    logging.info("="*80)
    logging.info("STAGE 7: SAVING RESULTS")
    logging.info("="*80)
    
    try:
        results_dir = config['output']['results_dir']
        
        # Performance metrics
        results_df = pd.DataFrame([results])
        results_df.to_csv(os.path.join(results_dir, 'performance_metrics.csv'), index=False)
        
        # Strategy comparison
        comparison.to_csv(os.path.join(results_dir, 'strategy_comparison.csv'))
        
        # Regime statistics
        stats = hmm.get_regime_statistics(returns)
        regime_stats_df = pd.DataFrame(stats['regime_statistics'])
        regime_stats_df.to_csv(os.path.join(results_dir, 'regime_statistics.csv'), index=False)
        
        # Regime conditional performance
        regimes = hmm.predict(returns)
        regime_returns = []
        
        for k in range(config['hmm']['n_regimes']):
            mask = regimes == k
            regime_ret = returns[mask]
            regime_returns.append({
                'regime': k,
                'mean_return': regime_ret.mean(),
                'volatility': regime_ret.std(),
                'sharpe': regime_ret.mean() / regime_ret.std() * np.sqrt(252) if regime_ret.std() > 0 else 0,
                'count': len(regime_ret)
            })
        
        regime_perf_df = pd.DataFrame(regime_returns)
        regime_perf_df.to_csv(os.path.join(results_dir, 'regime_conditional_performance.csv'), index=False)
        
        logging.info(f"[OK] Saved 4 result files to {results_dir}/")
        
    except Exception as e:
        logging.warning(f"[WARNING] Result saving failed: {e}")
        # Non-critical, continue


def main(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete production pipeline.
    """
    logging.info("="*80)
    logging.info("PRODUCTION PIPELINE START")
    logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info("="*80)
    
    # Setup
    setup_directories(config)
    
    # Stage 1: Data loading
    returns, prices, primary_ticker = load_and_validate_data(config)
    
    # Stage 2: Kalman filter
    kf = fit_kalman_filter(returns, config)
    
    # Stage 3: HMM
    hmm = fit_hmm(returns, config)
    
    # Stage 4: Signal generation
    signals = generate_signals(returns, kf, hmm, config)
    
    # Stage 5: Backtesting
    bt, strategies, comparison, results = run_backtest(signals, returns, config)
    
    # Stage 6: Visualization
    generate_visualizations(returns, kf, hmm, bt, strategies, config)
    
    # Stage 7: Save results
    save_results(results, comparison, hmm, returns, config)
    
    logging.info("="*80)
    logging.info("PRODUCTION PIPELINE COMPLETE")
    logging.info("="*80)
    
    return {
        'kf': kf,
        'hmm': hmm,
        'signals': signals,
        'backtest': bt,
        'results': results
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run production-grade Kalman Filter + HMM strategy'
    )
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-file', type=str, default='production_run.log',
                       help='Path to log file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_file)
    
    try:
        # Load config
        config = load_config(args.config)
        
        # Run pipeline
        output = main(config)
        
        logging.info(f"\n[SUCCESS] Pipeline completed without errors")
        logging.info(f"  Log file: {args.log_file}")
        logging.info(f"  Figures: {config['visualization']['figure_dir']}/")
        logging.info(f"  Results: {config['output']['results_dir']}/")
        
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"\n✗ FAILURE: {e}")
        logging.error("Pipeline terminated with errors")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
