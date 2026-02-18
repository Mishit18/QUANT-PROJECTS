"""
Diagnostics - focused on alpha robustness, not decoration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Callable, Optional
from pathlib import Path


class Diagnostics:
    """
    Critical diagnostics only:
    1. Rolling window test (does alpha persist?)
    2. Regime performance (does HMM help?)
    3. Monte Carlo risk (tail risk assessment)
    """
    
    @staticmethod
    def rolling_window_test(backtest_func: Callable,
                           positions: pd.Series,
                           spread_returns: pd.Series,
                           window: int = 252,
                           step: int = 63) -> pd.DataFrame:
        """
        Walk-forward test: Does alpha persist out-of-sample?
        
        Args:
            backtest_func: Backtest.run method
            positions: Position series
            spread_returns: Spread returns
            window: Window size in days
            step: Step size in days
        
        Returns:
            DataFrame with rolling metrics
        """
        results = []
        
        for start_idx in range(0, len(positions) - window, step):
            end_idx = start_idx + window
            
            window_positions = positions.iloc[start_idx:end_idx]
            window_returns = spread_returns.iloc[start_idx:end_idx]
            
            # Run backtest on window
            bt_results = backtest_func(window_positions, window_returns)
            
            metrics = bt_results['net_metrics']
            
            # Build result dict with available metrics only
            result = {
                'start_date': positions.index[start_idx],
                'end_date': positions.index[end_idx-1],
                'sharpe': metrics.get('sharpe_ratio', 0)
            }
            
            # Add optional metrics if available
            if 'total_return' in metrics:
                result['total_return'] = metrics['total_return']
            if 'max_drawdown' in metrics:
                result['max_drawdown'] = metrics['max_drawdown']
            if 'win_rate' in metrics:
                result['win_rate'] = metrics['win_rate']
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def regime_performance(returns: pd.Series,
                          states: np.ndarray,
                          regime_labels: list) -> pd.DataFrame:
        """
        Performance by regime: Does HMM add value?
        
        If Sharpe is similar across regimes, HMM is useless.
        """
        results = []
        
        for regime_idx, label in enumerate(regime_labels):
            mask = states == regime_idx
            regime_returns = returns[mask]
            
            if len(regime_returns) < 10:
                continue
            
            equity = (1 + regime_returns).cumprod()
            
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0
            
            results.append({
                'regime': label,
                'n_days': mask.sum(),
                'sharpe': sharpe,
                'total_return': equity.iloc[-1] - 1 if len(equity) > 0 else 0,
                'win_rate': (regime_returns > 0).sum() / len(regime_returns) if len(regime_returns) > 0 else 0
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def monte_carlo_risk(returns: pd.Series,
                        n_simulations: int = 1000,
                        n_periods: int = 252,
                        random_seed: int = 42) -> Dict:
        """
        Monte Carlo simulation: What's the tail risk?
        
        Returns probability of loss and VaR estimates.
        
        Args:
            returns: Historical returns series
            n_simulations: Number of simulation paths
            n_periods: Number of periods to simulate
            random_seed: Random seed for reproducibility
        """
        # Set seed for reproducibility
        np.random.seed(random_seed)
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Simulate paths
        simulated_paths = np.zeros((n_periods, n_simulations))
        
        for i in range(n_simulations):
            sim_returns = np.random.normal(mean_return, std_return, n_periods)
            simulated_paths[:, i] = (1 + sim_returns).cumprod()
        
        # Final values
        final_values = simulated_paths[-1, :]
        
        return {
            'prob_loss': (final_values < 1.0).sum() / n_simulations,
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
            'median_return': np.median(final_values) - 1,
            'worst_case': final_values.min() - 1
        }
    
    @staticmethod
    def plot_diagnostics(rolling_results: pd.DataFrame,
                        regime_perf: pd.DataFrame,
                        save_path: Optional[Path] = None):
        """Plot diagnostic results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rolling Sharpe
        ax = axes[0]
        rolling_results.plot(x='start_date', y='sharpe', ax=ax, marker='o', linewidth=2)
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Target')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('Rolling Window Sharpe Ratio', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('Period Start')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Regime performance
        ax = axes[1]
        regime_perf.plot(x='regime', y='sharpe', kind='bar', ax=ax, legend=False)
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5)
        ax.set_title('Sharpe Ratio by Regime', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_xlabel('Regime')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
