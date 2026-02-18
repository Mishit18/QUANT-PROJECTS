"""
Display project statistics summary.
Run this after executing the main pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def show_statistics():
    print('='*60)
    print('PROJECT STATISTICS SUMMARY')
    print('='*60)
    
    # Check if results exist
    if not Path('results/backtests').exists():
        print('\n⚠ No results found. Run: python main.py --mode full')
        return
    
    # Load results
    try:
        equity = pd.read_csv('results/backtests/XLK_XLV_equity_curve.csv')
        trades = pd.read_csv('results/backtests/XLK_XLV_trade_log.csv')
        pairs = pd.read_csv('results/diagnostics/cointegrated_pairs.csv')
        
        print('\nCointegration Analysis:')
        print(f'  Pairs tested: 28')
        print(f'  Cointegrated pairs: {len(pairs)}')
        print(f'  Best trace statistic: {pairs["trace_stat"].max():.2f}')
        print(f'  Average correlation: {pairs["correlation"].mean():.2f}')
        
        print('\nBacktest Performance:')
        print(f'  Total trades: {len(trades)}')
        winning_trades = (trades["pnl"] > 0).sum()
        print(f'  Winning trades: {winning_trades}')
        print(f'  Losing trades: {len(trades) - winning_trades}')
        print(f'  Win rate: {winning_trades / len(trades) * 100:.1f}%')
        print(f'  Average PnL per trade: ${trades["pnl"].mean():.2f}')
        print(f'  Best trade: ${trades["pnl"].max():.2f}')
        print(f'  Worst trade: ${trades["pnl"].min():.2f}')
        
        # Calculate holding period if dates are available
        if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            trades['entry_date'] = pd.to_datetime(trades['entry_date'])
            trades['exit_date'] = pd.to_datetime(trades['exit_date'])
            trades['days_held'] = (trades['exit_date'] - trades['entry_date']).dt.days
            print(f'  Average holding period: {trades["days_held"].mean():.1f} days')
        elif 'days_held' in trades.columns:
            print(f'  Average holding period: {trades["days_held"].mean():.1f} days')
        
        print(f'  Total PnL: ${trades["pnl"].sum():.2f}')
        
        print('\nRisk Metrics:')
        returns = equity['equity'].pct_change().dropna()
        print(f'  Daily volatility: {returns.std():.4f}')
        print(f'  Annualized volatility: {returns.std() * np.sqrt(252):.2%}')
        
        # Calculate max drawdown
        cummax = equity['equity'].cummax()
        drawdown = (equity['equity'] - cummax) / cummax
        max_dd = drawdown.min()
        print(f'  Max drawdown: {max_dd:.2%}')
        
        # Calculate Sharpe ratio
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
            print(f'  Sharpe ratio: {sharpe:.2f}')
        
        print('\nCapital Evolution:')
        print(f'  Initial capital: ${equity["equity"].iloc[0]:,.2f}')
        print(f'  Final capital: ${equity["equity"].iloc[-1]:,.2f}')
        print(f'  Total return: {(equity["equity"].iloc[-1] / equity["equity"].iloc[0] - 1) * 100:.3f}%')
        
        print('\n' + '='*60)
        
        # Show top pairs
        print('\nTop Cointegrated Pairs:')
        print(pairs[['pair', 'trace_stat', 'correlation']].to_string(index=False))
        
        print('\n' + '='*60)
        
        # Show trade summary
        print('\nTrade Summary:')
        # Select available columns
        display_cols = []
        for col in ['entry_date', 'exit_date', 'direction', 'pnl', 'days_held']:
            if col in trades.columns:
                display_cols.append(col)
        
        if display_cols:
            print(trades[display_cols].to_string(index=False))
        else:
            print(trades.to_string(index=False))
        
        print('\n' + '='*60)
        
    except FileNotFoundError as e:
        print(f'\n⚠ Error: {e}')
        print('Run the backtest first: python main.py --mode full')

if __name__ == '__main__':
    show_statistics()
