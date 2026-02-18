"""
Performance metrics and analytics module.

Computes comprehensive performance statistics including Sharpe ratio,
maximum drawdown, win rate, and risk-adjusted returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

from utils import (
    setup_logging,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    annualize_return,
    annualize_volatility
)


logger = setup_logging()


class PerformanceAnalyzer:
    """
    Analyzes backtest performance and computes risk metrics.
    """
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 periods_per_year: int = 252):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            periods_per_year: Trading periods per year
        """
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def compute_metrics(self, results: Dict) -> Dict:
        """
        Compute comprehensive performance metrics.
        
        Args:
            results: Backtest results dictionary
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info("Computing performance metrics")
        
        equity_curve = results['equity_curve']['equity']
        returns = results['equity_curve']['returns'].dropna()
        trade_log = results['trade_log']
        
        # Basic metrics
        initial_capital = equity_curve.iloc[0]
        final_capital = equity_curve.iloc[-1]
        total_return = (final_capital - initial_capital) / initial_capital
        
        # Annualized metrics
        n_years = len(returns) / self.periods_per_year
        annualized_ret = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annualized_vol = annualize_volatility(returns, self.periods_per_year)
        
        # Risk-adjusted returns
        sharpe = compute_sharpe_ratio(returns, self.risk_free_rate, self.periods_per_year)
        sortino = compute_sortino_ratio(returns, self.risk_free_rate, self.periods_per_year)
        calmar = compute_calmar_ratio(returns, self.periods_per_year)
        
        # Drawdown analysis
        max_dd, peak_date, trough_date = compute_max_drawdown(equity_curve)
        
        # Compute drawdown duration
        if pd.notna(peak_date) and pd.notna(trough_date):
            dd_duration = (trough_date - peak_date).days
        else:
            dd_duration = 0
        
        # Trade statistics
        if len(trade_log) > 0:
            winning_trades = trade_log[trade_log['net_pnl'] > 0]
            losing_trades = trade_log[trade_log['net_pnl'] < 0]
            
            n_trades = len(trade_log)
            n_wins = len(winning_trades)
            n_losses = len(losing_trades)
            win_rate = n_wins / n_trades if n_trades > 0 else 0
            
            avg_win = winning_trades['net_pnl'].mean() if n_wins > 0 else 0
            avg_loss = losing_trades['net_pnl'].mean() if n_losses > 0 else 0
            
            profit_factor = (
                abs(winning_trades['net_pnl'].sum() / losing_trades['net_pnl'].sum())
                if n_losses > 0 and losing_trades['net_pnl'].sum() != 0
                else np.inf if n_wins > 0 else 0
            )
            
            avg_holding_period = trade_log['holding_period'].mean()
            
            # Return statistics
            avg_return = trade_log['return_pct'].mean()
            std_return = trade_log['return_pct'].std()
            
        else:
            n_trades = 0
            n_wins = 0
            n_losses = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_period = 0
            avg_return = 0
            std_return = 0
        
        # Compile metrics
        metrics = {
            # Return metrics
            'total_return': total_return,
            'annualized_return': annualized_ret,
            'annualized_volatility': annualized_vol,
            
            # Risk-adjusted metrics
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            
            # Drawdown metrics
            'max_drawdown': max_dd,
            'max_drawdown_duration': dd_duration,
            'peak_date': peak_date,
            'trough_date': trough_date,
            
            # Trade metrics
            'n_trades': n_trades,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_period': avg_holding_period,
            'avg_return_per_trade': avg_return,
            'std_return_per_trade': std_return,
            
            # Cost metrics
            'total_costs': results.get('cumulative_costs', 0),
            'cost_per_trade': results.get('cumulative_costs', 0) / n_trades if n_trades > 0 else 0,
            
            # Capital metrics
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'peak_capital': equity_curve.max(),
            'trough_capital': equity_curve.min()
        }
        
        return metrics
    
    def print_summary(self, metrics: Dict) -> None:
        """
        Print formatted performance summary.
        
        Args:
            metrics: Performance metrics dictionary
        """
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        print("\nReturn Metrics:")
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['annualized_return']:>10.2%}")
        print(f"  Annualized Vol:      {metrics['annualized_volatility']:>10.2%}")
        
        print("\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
        
        print("\nDrawdown Metrics:")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"  DD Duration (days):  {metrics['max_drawdown_duration']:>10.0f}")
        
        print("\nTrade Metrics:")
        print(f"  Number of Trades:    {metrics['n_trades']:>10.0f}")
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
        print(f"  Avg Holding Period:  {metrics['avg_holding_period']:>10.1f} days")
        
        print("\nP&L Metrics:")
        print(f"  Average Win:         ${metrics['avg_win']:>10,.2f}")
        print(f"  Average Loss:        ${metrics['avg_loss']:>10,.2f}")
        print(f"  Total Costs:         ${metrics['total_costs']:>10,.2f}")
        
        print("\nCapital Metrics:")
        print(f"  Initial Capital:     ${metrics['initial_capital']:>10,.2f}")
        print(f"  Final Capital:       ${metrics['final_capital']:>10,.2f}")
        print(f"  Peak Capital:        ${metrics['peak_capital']:>10,.2f}")
        
        print("\n" + "="*60 + "\n")
    
    def compute_rolling_metrics(self,
                               equity_curve: pd.Series,
                               window: int = 252) -> pd.DataFrame:
        """
        Compute rolling performance metrics.
        
        Args:
            equity_curve: Equity curve series
            window: Rolling window size
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling return
        rolling_metrics['rolling_return'] = (
            (1 + returns).rolling(window=window).apply(lambda x: x.prod() - 1, raw=True)
        )
        
        # Rolling volatility
        rolling_metrics['rolling_vol'] = (
            returns.rolling(window=window).std() * np.sqrt(self.periods_per_year)
        )
        
        # Rolling Sharpe
        excess_returns = returns - self.risk_free_rate / self.periods_per_year
        rolling_metrics['rolling_sharpe'] = (
            np.sqrt(self.periods_per_year) *
            excess_returns.rolling(window=window).mean() /
            returns.rolling(window=window).std()
        )
        
        # Rolling max drawdown
        rolling_max = equity_curve.rolling(window=window, min_periods=1).max()
        rolling_dd = (equity_curve - rolling_max) / rolling_max
        rolling_metrics['rolling_max_dd'] = rolling_dd.rolling(window=window).min()
        
        return rolling_metrics
    
    def compute_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """
        Compute monthly return statistics.
        
        Args:
            equity_curve: Equity curve series
            
        Returns:
            DataFrame with monthly returns
        """
        monthly_equity = equity_curve.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        monthly_stats = pd.DataFrame({
            'return': monthly_returns,
            'cumulative': (1 + monthly_returns).cumprod() - 1
        })
        
        return monthly_stats
    
    def compare_to_benchmark(self,
                           strategy_returns: pd.Series,
                           benchmark_returns: pd.Series) -> Dict:
        """
        Compare strategy performance to benchmark.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with comparison metrics
        """
        # Align returns
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strat_ret = strategy_returns.loc[common_idx]
        bench_ret = benchmark_returns.loc[common_idx]
        
        # Excess returns
        excess_returns = strat_ret - bench_ret
        
        # Information ratio
        tracking_error = excess_returns.std() * np.sqrt(self.periods_per_year)
        if tracking_error > 0:
            information_ratio = (
                excess_returns.mean() * self.periods_per_year / tracking_error
            )
        else:
            information_ratio = 0
        
        # Beta
        covariance = np.cov(strat_ret, bench_ret)[0, 1]
        benchmark_var = bench_ret.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 0
        
        # Alpha
        rf_daily = self.risk_free_rate / self.periods_per_year
        alpha = (
            strat_ret.mean() - rf_daily -
            beta * (bench_ret.mean() - rf_daily)
        ) * self.periods_per_year
        
        comparison = {
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'beta': beta,
            'alpha': alpha,
            'correlation': strat_ret.corr(bench_ret)
        }
        
        return comparison
