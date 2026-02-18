"""
PnL analysis and performance metrics for backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: float = 252 * 6.5 * 3600) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        periods_per_year: Number of periods in a year (for annualization)
        
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std()
    
    sharpe = mean_return / std_return * np.sqrt(periods_per_year)
    
    return sharpe


def compute_max_drawdown(pnl_series: pd.Series) -> Dict[str, float]:
    """
    Compute maximum drawdown and related metrics.
    
    Args:
        pnl_series: Cumulative PnL series
        
    Returns:
        Dictionary with max_drawdown, max_drawdown_pct, drawdown_duration
    """
    # Running maximum
    running_max = pnl_series.expanding().max()
    
    # Drawdown
    drawdown = pnl_series - running_max
    
    # Max drawdown
    max_dd = drawdown.min()
    max_dd_pct = (max_dd / running_max.max() * 100) if running_max.max() > 0 else 0
    
    # Drawdown duration (number of periods in drawdown)
    in_drawdown = (drawdown < 0).astype(int)
    dd_duration = in_drawdown.sum()
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'drawdown_duration': dd_duration
    }


def compute_win_rate(trades_df: pd.DataFrame) -> float:
    """
    Compute win rate from trade history.
    
    Requires trades_df to have 'pnl' column.
    """
    if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
        return 0.0
    
    winning_trades = (trades_df['pnl'] > 0).sum()
    total_trades = len(trades_df)
    
    return winning_trades / total_trades if total_trades > 0 else 0.0


def compute_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Compute profit factor (gross profit / gross loss).
    
    Requires trades_df to have 'pnl' column.
    """
    if len(trades_df) == 0 or 'pnl' not in trades_df.columns:
        return 0.0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = -trades_df[trades_df['pnl'] < 0]['pnl'].sum()
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def analyze_backtest_results(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Comprehensive analysis of backtest results.
    
    Args:
        results_df: DataFrame from EventSimulator with PnL history
        
    Returns:
        Dictionary of performance metrics
    """
    metrics = {}
    
    # Total PnL
    metrics['total_pnl'] = results_df['total_pnl'].iloc[-1] if len(results_df) > 0 else 0
    metrics['realized_pnl'] = results_df['realized_pnl'].iloc[-1] if len(results_df) > 0 else 0
    
    # Returns
    returns = results_df['total_pnl'].diff()
    metrics['mean_return'] = returns.mean()
    metrics['std_return'] = returns.std()
    
    # Sharpe ratio
    metrics['sharpe_ratio'] = compute_sharpe_ratio(returns)
    
    # Drawdown
    dd_metrics = compute_max_drawdown(results_df['total_pnl'])
    metrics.update(dd_metrics)
    
    # Number of trades (approximate from position changes)
    position_changes = results_df['position'].diff().abs()
    metrics['num_trades'] = (position_changes > 0).sum()
    
    # Average position
    metrics['avg_position'] = results_df['position'].abs().mean()
    metrics['max_position'] = results_df['position'].abs().max()
    
    return metrics


def print_performance_summary(metrics: Dict[str, float]):
    """Print formatted performance summary."""
    print("\n" + "="*60)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\nPnL Metrics:")
    print(f"  Total PnL:              ${metrics.get('total_pnl', 0):,.2f}")
    print(f"  Realized PnL:           ${metrics.get('realized_pnl', 0):,.2f}")
    
    print(f"\nRisk-Adjusted Returns:")
    print(f"  Sharpe Ratio:           {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"  Max Drawdown:           ${metrics.get('max_drawdown', 0):,.2f}")
    print(f"  Max Drawdown %:         {metrics.get('max_drawdown_pct', 0):.2f}%")
    
    print(f"\nTrading Activity:")
    print(f"  Number of Trades:       {int(metrics.get('num_trades', 0))}")
    print(f"  Avg Position Size:      {metrics.get('avg_position', 0):.0f}")
    print(f"  Max Position Size:      {metrics.get('max_position', 0):.0f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    
    # Simulate PnL series
    returns = np.random.normal(0.001, 0.01, 1000)
    pnl = np.cumsum(returns)
    
    results_df = pd.DataFrame({
        'timestamp': range(1000),
        'total_pnl': pnl,
        'realized_pnl': pnl * 0.8,
        'position': np.random.randint(-100, 100, 1000)
    })
    
    # Analyze
    metrics = analyze_backtest_results(results_df)
    print_performance_summary(metrics)
