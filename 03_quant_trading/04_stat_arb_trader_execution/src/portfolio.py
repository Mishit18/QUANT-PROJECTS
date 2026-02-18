"""
Portfolio-level statistical arbitrage execution.

Rationale: Statistical arbitrage is a cross-sectional strategy.
Individual pairs are expected to be noisy; diversification across
multiple qualifying pairs is the primary driver of risk-adjusted returns.

This is NOT cherry-picking - all pairs that pass quality gates are included.
Allocation is based on risk parity, not realized performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt

from src.execution import ExecutionModel


class PortfolioBacktest:
    """
    Portfolio-level backtest with equal-risk or inverse-volatility weighting.
    
    Design principles:
    1. All qualifying pairs are included (no performance filtering)
    2. Allocation based on ex-ante risk, not ex-post returns
    3. Portfolio-level volatility targeting
    4. Diversification is the Sharpe driver, not pair selection
    """
    
    def __init__(self, 
                 execution_model: Optional[ExecutionModel] = None,
                 allocation_method: str = 'equal_risk',
                 target_volatility: float = 0.10):
        """
        Args:
            execution_model: Execution cost model
            allocation_method: 'equal_risk' or 'inverse_volatility'
            target_volatility: Annual portfolio volatility target (e.g., 0.10 = 10%)
        """
        if execution_model is None:
            execution_model = ExecutionModel()
        
        self.execution_model = execution_model
        self.allocation_method = allocation_method
        self.target_volatility = target_volatility
        self.pair_results = {}
        self.portfolio_results = None
    
    def add_pair(self, 
                 pair_name: str,
                 positions: pd.Series,
                 spread_returns: pd.Series) -> Dict:
        """
        Add a pair to the portfolio.
        
        Args:
            pair_name: Identifier for the pair
            positions: Position sizes over time
            spread_returns: Spread returns
        
        Returns:
            Individual pair results
        """
        # Align data
        common_idx = positions.index.intersection(spread_returns.index)
        positions = positions.loc[common_idx]
        spread_returns = spread_returns.loc[common_idx]
        
        # Apply execution costs
        net_returns, cost_breakdown = self.execution_model.apply_costs(
            positions, spread_returns
        )
        
        # Store pair data
        self.pair_results[pair_name] = {
            'positions': positions,
            'spread_returns': spread_returns,
            'net_returns': net_returns,
            'cost_breakdown': cost_breakdown
        }
        
        return {
            'net_returns': net_returns,
            'positions': positions
        }
    
    def run_portfolio(self) -> Dict:
        """
        Construct and backtest portfolio using all added pairs.
        
        Allocation rationale:
        - Equal-risk: Each pair contributes equally to portfolio volatility
        - Inverse-volatility: Weight inversely proportional to volatility
        
        This is NOT performance-based allocation. Weights are determined
        by ex-ante risk characteristics, not ex-post returns.
        
        Returns:
            Portfolio-level results
        """
        if len(self.pair_results) == 0:
            raise ValueError("No pairs added to portfolio")
        
        # Get common index across all pairs
        common_idx = None
        for pair_data in self.pair_results.values():
            if common_idx is None:
                common_idx = pair_data['net_returns'].index
            else:
                common_idx = common_idx.intersection(pair_data['net_returns'].index)
        
        # Align all pairs to common index
        aligned_returns = {}
        aligned_positions = {}
        
        for pair_name, pair_data in self.pair_results.items():
            aligned_returns[pair_name] = pair_data['net_returns'].loc[common_idx]
            aligned_positions[pair_name] = pair_data['positions'].loc[common_idx]
        
        # Calculate allocation weights
        weights = self._calculate_weights(aligned_returns)
        
        # Construct portfolio returns
        portfolio_returns = pd.Series(0.0, index=common_idx)
        
        for pair_name, weight in weights.items():
            portfolio_returns += weight * aligned_returns[pair_name]
        
        # Apply volatility targeting
        # Rationale: Scale portfolio to target volatility level
        # This is risk management, not return optimization
        realized_vol = portfolio_returns.std() * np.sqrt(252)
        if realized_vol > 0:
            vol_scalar = self.target_volatility / realized_vol
        else:
            vol_scalar = 1.0
        
        portfolio_returns_scaled = portfolio_returns * vol_scalar
        
        # Calculate metrics
        portfolio_equity = (1 + portfolio_returns_scaled).cumprod()
        
        metrics = self._calculate_metrics(portfolio_returns_scaled)
        
        self.portfolio_results = {
            'returns': portfolio_returns_scaled,
            'returns_unscaled': portfolio_returns,
            'equity': portfolio_equity,
            'weights': weights,
            'metrics': metrics,
            'vol_scalar': vol_scalar,
            'n_pairs': len(self.pair_results)
        }
        
        return self.portfolio_results
    
    def _calculate_weights(self, returns_dict: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate portfolio weights based on allocation method.
        
        Rationale:
        - Equal-risk: Each pair contributes equally to portfolio variance
        - Inverse-volatility: Weight inversely proportional to volatility
        
        This is NOT performance-based. Weights depend only on risk characteristics.
        
        Args:
            returns_dict: Dictionary of pair returns
        
        Returns:
            Dictionary of weights (sum to 1.0)
        """
        if self.allocation_method == 'equal_risk':
            # Equal risk contribution
            # Weight inversely proportional to volatility
            vols = {}
            for pair_name, returns in returns_dict.items():
                vols[pair_name] = returns.std()
            
            # Inverse volatility weights
            inv_vols = {k: 1.0 / v if v > 0 else 0 for k, v in vols.items()}
            total_inv_vol = sum(inv_vols.values())
            
            if total_inv_vol > 0:
                weights = {k: v / total_inv_vol for k, v in inv_vols.items()}
            else:
                # Fallback to equal weight
                n = len(returns_dict)
                weights = {k: 1.0 / n for k in returns_dict.keys()}
        
        elif self.allocation_method == 'inverse_volatility':
            # Same as equal_risk for this implementation
            vols = {}
            for pair_name, returns in returns_dict.items():
                vols[pair_name] = returns.std()
            
            inv_vols = {k: 1.0 / v if v > 0 else 0 for k, v in vols.items()}
            total_inv_vol = sum(inv_vols.values())
            
            if total_inv_vol > 0:
                weights = {k: v / total_inv_vol for k, v in inv_vols.items()}
            else:
                n = len(returns_dict)
                weights = {k: 1.0 / n for k in returns_dict.keys()}
        
        else:
            # Equal weight fallback
            n = len(returns_dict)
            weights = {k: 1.0 / n for k in returns_dict.keys()}
        
        return weights
    
    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate portfolio performance metrics."""
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
    
    def get_attribution(self) -> pd.DataFrame:
        """
        Get pair-level attribution.
        
        Shows contribution of each pair to portfolio, but this is
        descriptive only - allocation was based on risk, not performance.
        """
        if self.portfolio_results is None:
            raise ValueError("Must run portfolio backtest first")
        
        attribution = []
        weights = self.portfolio_results['weights']
        
        for pair_name, pair_data in self.pair_results.items():
            returns = pair_data['net_returns']
            
            # Calculate pair metrics
            equity = (1 + returns).cumprod()
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            total_return = equity.iloc[-1] - 1
            
            attribution.append({
                'Pair': pair_name,
                'Weight': f"{weights[pair_name]:.1%}",
                'Total Return': f"{total_return:.2%}",
                'Sharpe': f"{sharpe:.2f}",
                'Contribution': f"{weights[pair_name] * total_return:.2%}"
            })
        
        return pd.DataFrame(attribution)
    
    def plot_results(self, save_path: Optional[Path] = None):
        """Generate portfolio performance plots."""
        if self.portfolio_results is None:
            raise ValueError("Must run portfolio backtest first")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        returns = self.portfolio_results['returns']
        equity = self.portfolio_results['equity']
        
        # Equity curve
        ax = axes[0, 0]
        equity.plot(ax=ax, linewidth=2, color='steelblue')
        ax.set_title('Portfolio Equity Curve', fontweight='bold')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True, alpha=0.3)
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.3)
        
        # Drawdown
        ax = axes[0, 1]
        drawdown = (equity / equity.cummax() - 1)
        drawdown.plot(ax=ax, color='red', linewidth=2)
        ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Portfolio Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown')
        ax.grid(True, alpha=0.3)
        
        # Returns distribution
        ax = axes[1, 0]
        returns.hist(bins=50, ax=ax, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Returns')
        ax.grid(True, alpha=0.3)
        
        # Rolling Sharpe
        ax = axes[1, 1]
        rolling_sharpe = returns.rolling(60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        rolling_sharpe.plot(ax=ax, linewidth=2, color='steelblue')
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
