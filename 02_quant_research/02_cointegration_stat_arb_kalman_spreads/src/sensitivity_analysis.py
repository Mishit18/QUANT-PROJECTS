"""
Sensitivity analysis and robustness testing module.

Tests strategy performance across parameter variations and market regimes
to assess robustness and identify failure modes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import itertools
import logging
from tqdm import tqdm

from utils import setup_logging
from backtest_engine import BacktestEngine
from performance_metrics import PerformanceAnalyzer


logger = setup_logging()


class SensitivityAnalyzer:
    """
    Performs sensitivity analysis on strategy parameters.
    """
    
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        """
        Initialize sensitivity analyzer.
        
        Args:
            config_path: Path to strategy configuration
        """
        self.config_path = config_path
        self.performance_analyzer = PerformanceAnalyzer()
    
    def parameter_sweep(self,
                       pair: Tuple[str, str],
                       price_y: pd.Series,
                       price_x: pd.Series,
                       parameter_grid: Dict[str, List]) -> pd.DataFrame:
        """
        Sweep through parameter combinations and evaluate performance.
        
        Args:
            pair: Tuple of (ticker_y, ticker_x)
            price_y: Price series for Y
            price_x: Price series for X
            parameter_grid: Dictionary mapping parameter names to value lists
            
        Returns:
            DataFrame with results for each parameter combination
        """
        logger.info(f"Running parameter sweep for {pair}")
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        
        for combo in tqdm(combinations, desc="Parameter sweep"):
            # Create parameter dict
            params = dict(zip(param_names, combo))
            
            try:
                # Run backtest with these parameters
                backtest_engine = BacktestEngine(self.config_path)
                
                # Update parameters
                if 'entry_threshold' in params:
                    backtest_engine.signal_generator.entry_threshold = params['entry_threshold']
                if 'exit_threshold' in params:
                    backtest_engine.signal_generator.exit_threshold = params['exit_threshold']
                if 'lookback_window' in params:
                    backtest_engine.signal_generator.lookback_window = params['lookback_window']
                if 'transaction_cost' in params:
                    backtest_engine.executor.transaction_cost = params['transaction_cost']
                    backtest_engine.executor.total_cost = (
                        params['transaction_cost'] +
                        backtest_engine.executor.slippage +
                        backtest_engine.executor.market_impact
                    )
                
                # Run backtest
                backtest_results = backtest_engine.run(pair, price_y, price_x)
                
                # Compute metrics
                metrics = self.performance_analyzer.compute_metrics(backtest_results)
                
                # Store results
                result_row = params.copy()
                result_row.update({
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'n_trades': metrics['n_trades'],
                    'profit_factor': metrics['profit_factor']
                })
                
                results.append(result_row)
            
            except Exception as e:
                logger.warning(f"Error with parameters {params}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Parameter sweep complete: {len(results_df)} successful runs")
        
        return results_df
    
    def regime_analysis(self,
                       pair: Tuple[str, str],
                       price_y: pd.Series,
                       price_x: pd.Series,
                       regime_periods: List[Tuple[str, str, str]]) -> pd.DataFrame:
        """
        Analyze strategy performance across different market regimes.
        
        Args:
            pair: Tuple of (ticker_y, ticker_x)
            price_y: Price series for Y
            price_x: Price series for X
            regime_periods: List of (regime_name, start_date, end_date) tuples
            
        Returns:
            DataFrame with performance by regime
        """
        logger.info(f"Running regime analysis for {pair}")
        
        backtest_engine = BacktestEngine(self.config_path)
        results = []
        
        for regime_name, start_date, end_date in regime_periods:
            logger.info(f"Testing regime: {regime_name} ({start_date} to {end_date})")
            
            try:
                # Filter data for regime period
                mask = (price_y.index >= start_date) & (price_y.index <= end_date)
                regime_price_y = price_y[mask]
                regime_price_x = price_x[mask]
                
                if len(regime_price_y) < 60:
                    logger.warning(f"Insufficient data for regime {regime_name}")
                    continue
                
                # Run backtest
                backtest_results = backtest_engine.run(
                    pair, regime_price_y, regime_price_x
                )
                
                # Compute metrics
                metrics = self.performance_analyzer.compute_metrics(backtest_results)
                
                # Store results
                result_row = {
                    'regime': regime_name,
                    'start_date': start_date,
                    'end_date': end_date,
                    'n_days': len(regime_price_y),
                    'total_return': metrics['total_return'],
                    'annualized_return': metrics['annualized_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'n_trades': metrics['n_trades']
                }
                
                results.append(result_row)
            
            except Exception as e:
                logger.warning(f"Error in regime {regime_name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        return results_df
    
    def rolling_window_analysis(self,
                               pair: Tuple[str, str],
                               price_y: pd.Series,
                               price_x: pd.Series,
                               window_size: int = 252,
                               step_size: int = 63) -> pd.DataFrame:
        """
        Analyze strategy performance using rolling windows.
        
        Args:
            pair: Tuple of (ticker_y, ticker_x)
            price_y: Price series for Y
            price_x: Price series for X
            window_size: Window size in days
            step_size: Step size for rolling window
            
        Returns:
            DataFrame with performance by window
        """
        logger.info(f"Running rolling window analysis for {pair}")
        
        backtest_engine = BacktestEngine(self.config_path)
        results = []
        
        start_idx = 0
        while start_idx + window_size <= len(price_y):
            end_idx = start_idx + window_size
            
            window_price_y = price_y.iloc[start_idx:end_idx]
            window_price_x = price_x.iloc[start_idx:end_idx]
            
            window_start = window_price_y.index[0]
            window_end = window_price_y.index[-1]
            
            try:
                # Run backtest
                backtest_results = backtest_engine.run(
                    pair, window_price_y, window_price_x
                )
                
                # Compute metrics
                metrics = self.performance_analyzer.compute_metrics(backtest_results)
                
                # Store results
                result_row = {
                    'window_start': window_start,
                    'window_end': window_end,
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'n_trades': metrics['n_trades']
                }
                
                results.append(result_row)
            
            except Exception as e:
                logger.warning(f"Error in window {window_start} to {window_end}: {e}")
            
            start_idx += step_size
        
        results_df = pd.DataFrame(results)
        
        logger.info(f"Rolling window analysis complete: {len(results_df)} windows")
        
        return results_df
    
    def monte_carlo_simulation(self,
                              trade_returns: pd.Series,
                              n_simulations: int = 1000,
                              n_trades: int = 100) -> Dict:
        """
        Monte Carlo simulation of strategy returns.
        
        Args:
            trade_returns: Historical trade returns
            n_simulations: Number of simulations
            n_trades: Number of trades per simulation
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running Monte Carlo simulation ({n_simulations} simulations)")
        
        if len(trade_returns) == 0:
            raise ValueError("No trade returns provided")
        
        simulated_returns = []
        simulated_sharpes = []
        simulated_max_dds = []
        
        for _ in range(n_simulations):
            # Sample trades with replacement
            sampled_returns = np.random.choice(
                trade_returns.values,
                size=n_trades,
                replace=True
            )
            
            # Compute cumulative return
            cumulative_return = (1 + sampled_returns).prod() - 1
            simulated_returns.append(cumulative_return)
            
            # Compute Sharpe ratio
            if sampled_returns.std() > 0:
                sharpe = sampled_returns.mean() / sampled_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            simulated_sharpes.append(sharpe)
            
            # Compute max drawdown
            equity_curve = (1 + sampled_returns).cumprod()
            cummax = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - cummax) / cummax
            max_dd = drawdown.min()
            simulated_max_dds.append(max_dd)
        
        # Compute statistics
        results = {
            'mean_return': np.mean(simulated_returns),
            'median_return': np.median(simulated_returns),
            'std_return': np.std(simulated_returns),
            'percentile_5': np.percentile(simulated_returns, 5),
            'percentile_95': np.percentile(simulated_returns, 95),
            'mean_sharpe': np.mean(simulated_sharpes),
            'median_sharpe': np.median(simulated_sharpes),
            'mean_max_dd': np.mean(simulated_max_dds),
            'median_max_dd': np.median(simulated_max_dds),
            'prob_positive': np.mean(np.array(simulated_returns) > 0),
            'simulated_returns': simulated_returns,
            'simulated_sharpes': simulated_sharpes,
            'simulated_max_dds': simulated_max_dds
        }
        
        logger.info(
            f"Monte Carlo complete: Mean return={results['mean_return']:.2%}, "
            f"Prob(positive)={results['prob_positive']:.2%}"
        )
        
        return results
    
    def stress_test(self,
                   pair: Tuple[str, str],
                   price_y: pd.Series,
                   price_x: pd.Series,
                   shock_scenarios: Dict[str, Dict]) -> pd.DataFrame:
        """
        Stress test strategy under extreme scenarios.
        
        Args:
            pair: Tuple of (ticker_y, ticker_x)
            price_y: Price series for Y
            price_x: Price series for X
            shock_scenarios: Dictionary of scenario definitions
            
        Returns:
            DataFrame with stress test results
        """
        logger.info(f"Running stress tests for {pair}")
        
        backtest_engine = BacktestEngine(self.config_path)
        results = []
        
        for scenario_name, scenario_params in shock_scenarios.items():
            logger.info(f"Testing scenario: {scenario_name}")
            
            try:
                # Apply scenario modifications
                modified_engine = BacktestEngine(self.config_path)
                
                if 'transaction_cost_multiplier' in scenario_params:
                    multiplier = scenario_params['transaction_cost_multiplier']
                    modified_engine.executor.transaction_cost *= multiplier
                    modified_engine.executor.total_cost = (
                        modified_engine.executor.transaction_cost +
                        modified_engine.executor.slippage +
                        modified_engine.executor.market_impact
                    )
                
                if 'volatility_shock' in scenario_params:
                    # Increase spread volatility
                    pass  # Would need to modify spread computation
                
                # Run backtest
                backtest_results = modified_engine.run(pair, price_y, price_x)
                
                # Compute metrics
                metrics = self.performance_analyzer.compute_metrics(backtest_results)
                
                # Store results
                result_row = {
                    'scenario': scenario_name,
                    'total_return': metrics['total_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'win_rate': metrics['win_rate'],
                    'n_trades': metrics['n_trades']
                }
                
                results.append(result_row)
            
            except Exception as e:
                logger.warning(f"Error in scenario {scenario_name}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        return results_df
