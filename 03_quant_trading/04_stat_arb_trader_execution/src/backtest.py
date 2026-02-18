"""
Backtesting engine - lean and focused on alpha metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from src.execution import ExecutionModel


class Backtest:
    """
    Minimal backtest engine.
    Focus: Does the strategy have alpha?
    """
    
    def __init__(self, execution_model: Optional[ExecutionModel] = None):
        """
        Args:
            execution_model: Execution model for costs
        """
        if execution_model is None:
            execution_model = ExecutionModel()
        
        self.execution_model = execution_model
        self.results = None
    
    def run(self, positions: pd.Series, spread_returns: pd.Series) -> Dict:
        """
        Run backtest on positions and returns.
        
        Args:
            positions: Position sizes over time
            spread_returns: Spread returns
        
        Returns:
            Dict with results
        """
        # Align
        common_idx = positions.index.intersection(spread_returns.index)
        positions = positions.loc[common_idx]
        spread_returns = spread_returns.loc[common_idx]
        
        # Apply execution costs
        net_returns, cost_breakdown = self.execution_model.apply_costs(
            positions, spread_returns
        )
        
        gross_returns = cost_breakdown['gross_returns']
        
        # Equity curves (normalized to 1.0 start)
        gross_equity = (1 + gross_returns).cumprod()
        net_equity = (1 + net_returns).cumprod()
        
        # Metrics
        gross_metrics = self._calculate_metrics(gross_returns, positions)
        net_metrics = self._calculate_metrics(net_returns, positions)
        
        self.results = {
            'positions': positions,
            'gross_returns': gross_returns,
            'net_returns': net_returns,
            'gross_equity': gross_equity,
            'net_equity': net_equity,
            'cost_breakdown': cost_breakdown,
            'gross_metrics': gross_metrics,
            'net_metrics': net_metrics,
            'n_trades': (positions.diff().abs() > 0.01).sum()
        }
        
        return self.results
    
    def _calculate_metrics(self, returns: pd.Series, positions: pd.Series) -> Dict:
        """Calculate performance metrics."""
        equity = (1 + returns).cumprod()
        
        # Sharpe ratio
        if returns.std() == 0:
            sharpe = 0
        else:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            sortino = 0
        else:
            sortino = returns.mean() / downside.std() * np.sqrt(252)
        
        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min()
        
        # Calmar ratio
        if max_dd == 0:
            calmar = 0
        else:
            n_years = len(returns) / 252
            annual_return = (equity.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
            calmar = annual_return / abs(max_dd)
        
        # Total return
        total_return = equity.iloc[-1] - 1
        
        # Annual return
        n_years = len(returns) / 252
        annual_return = (equity.iloc[-1] ** (1 / n_years)) - 1 if n_years > 0 else 0
        
        # Volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Win rate
        winning_days = (returns > 0).sum()
        total_days = (returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'win_rate': win_rate
        }
    
    def get_summary(self) -> pd.DataFrame:
        """Get performance summary table."""
        if self.results is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'Gross': self.results['gross_metrics'],
            'Net': self.results['net_metrics']
        }).T
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Generate performance plots."""
        if self.results is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Equity curve
        ax = axes[0, 0]
        self.results['gross_equity'].plot(ax=ax, label='Gross', linewidth=2, alpha=0.7)
        self.results['net_equity'].plot(ax=ax, label='Net', linewidth=2)
        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[0, 1]
        equity = self.results['net_equity']
        drawdown = (equity / equity.cummax() - 1)
        drawdown.plot(ax=ax, color='red', linewidth=2)
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        # Returns distribution
        ax = axes[1, 0]
        self.results['net_returns'].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.grid(True, alpha=0.3)
        
        # Rolling Sharpe
        ax = axes[1, 1]
        rolling_sharpe = self.results['net_returns'].rolling(60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_sharpe.plot(ax=ax, linewidth=2)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Target')
        ax.set_title('Rolling 60-Day Sharpe', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class MultiPairBacktest:
    """
    Multi-pair backtest with pair-level attribution.
    
    Provides detailed breakdown of performance by pair to identify
    sources of alpha and risk.
    """
    
    def __init__(self, execution_model: Optional[ExecutionModel] = None):
        if execution_model is None:
            execution_model = ExecutionModel()
        
        self.execution_model = execution_model
        self.pair_results = {}
    
    def run_pair(self, pair_name: str, positions: pd.Series, spread_returns: pd.Series) -> Dict:
        """Run backtest for a single pair."""
        backtest = Backtest(self.execution_model)
        results = backtest.run(positions, spread_returns)
        
        self.pair_results[pair_name] = {
            'backtest': backtest,
            'results': results,
            'metrics': results['net_metrics']
        }
        
        return results
    
    def get_attribution_table(self) -> pd.DataFrame:
        """
        Pair-level attribution table.
        
        Shows:
        - PnL by pair
        - Trades by pair
        - Sharpe by pair
        - % capital allocation by pair
        """
        if not self.pair_results:
            return pd.DataFrame()
        
        attribution = []
        
        for pair_name, data in self.pair_results.items():
            metrics = data['metrics']
            results = data['results']
            
            attribution.append({
                'Pair': pair_name,
                'Total Return': f"{metrics['total_return']:.2%}",
                'Annual Return': f"{metrics['annual_return']:.2%}",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Sortino': f"{metrics['sortino_ratio']:.2f}",
                'Max DD': f"{metrics['max_drawdown']:.2%}",
                'Win Rate': f"{metrics['win_rate']:.1%}",
                'Trades': results['n_trades'],
                'Volatility': f"{metrics['annual_volatility']:.2%}"
            })
        
        df = pd.DataFrame(attribution)
        return df.set_index('Pair')
    
    def plot_pair_comparison(self, save_path: Optional[Path] = None):
        """Plot equity curves for all pairs."""
        if not self.pair_results:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for pair_name, data in self.pair_results.items():
            equity = data['results']['net_equity']
            sharpe = data['metrics']['sharpe_ratio']
            ax.plot(equity.index, equity.values, label=f"{pair_name} (Sharpe={sharpe:.2f})", linewidth=2)
        
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Pair-Level Equity Curves', fontweight='bold', fontsize=14)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
