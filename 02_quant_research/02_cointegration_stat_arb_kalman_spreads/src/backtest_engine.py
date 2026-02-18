"""
Backtesting engine for statistical arbitrage strategies.

Implements event-driven backtest with realistic execution, position tracking,
and comprehensive performance analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

from utils import setup_logging, load_config, ensure_dir
from execution import ExecutionSimulator, PositionSizer
from signal_generation import SignalGenerator
from kalman_filter import KalmanSpreadModel
from spread_metrics import SpreadAnalyzer


logger = setup_logging()


class BacktestEngine:
    """
    Event-driven backtesting engine for pairs trading strategies.
    """
    
    def __init__(self, config_path: str = "config/strategy_config.yaml"):
        """
        Initialize backtest engine.
        
        Args:
            config_path: Path to strategy configuration file
        """
        self.config = load_config(config_path)
        
        # Initialize components
        self.signal_generator = SignalGenerator(
            entry_threshold=self.config['signals']['entry_threshold'],
            exit_threshold=self.config['signals']['exit_threshold'],
            stop_loss_threshold=self.config['signals']['stop_loss_threshold'],
            lookback_window=self.config['spread']['lookback_window'],
            min_holding_period=self.config['signals']['min_holding_period'],
            max_holding_period=self.config['signals']['max_holding_period'],
            volatility_filter=self.config['signals']['volatility_filter']['enabled'],
            vol_lookback=self.config['signals']['volatility_filter']['lookback'],
            vol_percentile=self.config['signals']['volatility_filter']['max_percentile']
        )
        
        self.executor = ExecutionSimulator(
            transaction_cost=self.config['execution']['transaction_cost'],
            slippage=self.config['execution']['slippage'],
            market_impact=self.config['execution']['market_impact']
        )
        
        self.position_sizer = PositionSizer(
            method=self.config['position_sizing']['method'],
            volatility_target=self.config['position_sizing']['volatility_target'],
            max_leverage=self.config['position_sizing']['max_leverage'],
            max_position_size=self.config['position_sizing']['max_position_size']
        )
        
        self.initial_capital = self.config['backtest']['initial_capital']
        
    def run(self,
           pair: Tuple[str, str],
           price_y: pd.Series,
           price_x: pd.Series,
           hedge_ratio: Optional[pd.Series] = None,
           spread: Optional[pd.Series] = None) -> Dict:
        """
        Run backtest for a single pair.
        
        Args:
            pair: Tuple of (ticker_y, ticker_x)
            price_y: Price series for Y
            price_x: Price series for X
            hedge_ratio: Pre-computed hedge ratio (uses Kalman if None)
            spread: Pre-computed spread (computed if None)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for pair {pair}")
        
        # Align price series
        df = pd.DataFrame({'y': price_y, 'x': price_x}).dropna()
        price_y = df['y']
        price_x = df['x']
        
        # Compute spread and hedge ratio if not provided
        if spread is None or hedge_ratio is None:
            logger.info("Computing spread using Kalman Filter")
            try:
                kf_model = KalmanSpreadModel(
                    transition_covariance=self.config['kalman']['transition_covariance'],
                    observation_covariance=self.config['kalman']['observation_covariance']
                )
                spread, hedge_ratio = kf_model.fit_transform(
                    price_y, price_x,
                    em_iterations=self.config['kalman']['em_iterations']
                )
            except Exception as e:
                logger.warning(f"Kalman Filter failed ({e}), using OLS hedge ratio")
                # Fallback to OLS
                from statsmodels.regression.linear_model import OLS
                from statsmodels.tools import add_constant
                X = add_constant(price_x)
                model = OLS(price_y, X).fit()
                beta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
                alpha = model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0]
                spread = price_y - beta * price_x - alpha
                hedge_ratio = pd.Series(beta, index=price_y.index)
                logger.info(f"Using OLS hedge ratio: {beta:.4f}")
        
        # Generate signals
        signals = self.signal_generator.generate_signals(spread)
        
        # Initialize tracking variables
        capital = self.initial_capital
        equity_curve = []
        cash_curve = []
        position_value_curve = []
        trade_log = []
        
        current_position = 0
        shares_y = 0
        shares_x = 0
        entry_price_y = 0
        entry_price_x = 0
        entry_date = None
        cumulative_pnl = 0
        cumulative_costs = 0
        
        # Event-driven backtest loop
        for date in signals.index:
            position_signal = signals.loc[date, 'position']
            position_change = signals.loc[date, 'position_change']
            
            current_price_y = price_y.loc[date]
            current_price_x = price_x.loc[date]
            current_hedge_ratio = hedge_ratio.loc[date] if date in hedge_ratio.index else 1.0
            
            # Compute current position value
            if current_position != 0:
                position_value = (
                    shares_y * current_price_y +
                    shares_x * current_price_x
                )
            else:
                position_value = 0
            
            # Handle position changes
            if position_change != 0:
                # Close existing position
                if current_position != 0:
                    # Compute P&L
                    pnl_details = self.executor.compute_pnl(
                        shares_y, shares_x,
                        entry_price_y, entry_price_x,
                        current_price_y, current_price_x
                    )
                    
                    # Execution cost for closing
                    close_cost = self.executor.compute_execution_cost(
                        abs(shares_y * current_price_y) + abs(shares_x * current_price_x)
                    )
                    
                    # Update capital
                    net_pnl = pnl_details['total_pnl'] - close_cost
                    capital += net_pnl
                    cumulative_pnl += pnl_details['total_pnl']
                    cumulative_costs += close_cost
                    
                    # Log trade
                    trade_log.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'direction': 'long' if current_position == 1 else 'short',
                        'entry_price_y': entry_price_y,
                        'entry_price_x': entry_price_x,
                        'exit_price_y': current_price_y,
                        'exit_price_x': current_price_x,
                        'shares_y': shares_y,
                        'shares_x': shares_x,
                        'pnl': pnl_details['total_pnl'],
                        'cost': close_cost,
                        'net_pnl': net_pnl,
                        'return_pct': pnl_details['return_pct'],
                        'holding_period': (date - entry_date).days
                    })
                    
                    # Reset position
                    current_position = 0
                    shares_y = 0
                    shares_x = 0
                
                # Open new position
                if position_signal != 0:
                    # Compute position size
                    spread_vol = spread.rolling(window=60).std().loc[date]
                    if pd.isna(spread_vol) or spread_vol == 0:
                        spread_vol = spread.std()
                    
                    position_size = self.position_sizer.compute_position_size(
                        capital, spread_vol * np.sqrt(252)
                    )
                    
                    # Execute trade
                    execution = self.executor.execute_spread_trade(
                        position_signal,
                        current_price_y,
                        current_price_x,
                        current_hedge_ratio,
                        position_size
                    )
                    
                    if execution['executed']:
                        # Update position
                        current_position = position_signal
                        shares_y = execution['shares_y']
                        shares_x = execution['shares_x']
                        entry_price_y = current_price_y
                        entry_price_x = current_price_x
                        entry_date = date
                        
                        # Update capital
                        capital -= execution['cost']
                        cumulative_costs += execution['cost']
            
            # Track equity
            total_equity = capital + position_value
            equity_curve.append(total_equity)
            cash_curve.append(capital)
            position_value_curve.append(position_value)
        
        # Close any remaining position at end
        if current_position != 0:
            final_price_y = price_y.iloc[-1]
            final_price_x = price_x.iloc[-1]
            
            pnl_details = self.executor.compute_pnl(
                shares_y, shares_x,
                entry_price_y, entry_price_x,
                final_price_y, final_price_x
            )
            
            close_cost = self.executor.compute_execution_cost(
                abs(shares_y * final_price_y) + abs(shares_x * final_price_x)
            )
            
            net_pnl = pnl_details['total_pnl'] - close_cost
            capital += net_pnl
            cumulative_pnl += pnl_details['total_pnl']
            cumulative_costs += close_cost
            
            trade_log.append({
                'entry_date': entry_date,
                'exit_date': signals.index[-1],
                'direction': 'long' if current_position == 1 else 'short',
                'entry_price_y': entry_price_y,
                'entry_price_x': entry_price_x,
                'exit_price_y': final_price_y,
                'exit_price_x': final_price_x,
                'shares_y': shares_y,
                'shares_x': shares_x,
                'pnl': pnl_details['total_pnl'],
                'cost': close_cost,
                'net_pnl': net_pnl,
                'return_pct': pnl_details['return_pct'],
                'holding_period': (signals.index[-1] - entry_date).days
            })
            
            equity_curve[-1] = capital
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'equity': equity_curve,
            'cash': cash_curve,
            'position_value': position_value_curve
        }, index=signals.index)
        
        # Compute returns
        results_df['returns'] = results_df['equity'].pct_change()
        
        # Compile results
        results = {
            'pair': pair,
            'equity_curve': results_df,
            'trade_log': pd.DataFrame(trade_log) if trade_log else pd.DataFrame(),
            'signals': signals,
            'spread': spread,
            'hedge_ratio': hedge_ratio,
            'final_capital': capital,
            'total_return': (capital - self.initial_capital) / self.initial_capital,
            'cumulative_pnl': cumulative_pnl,
            'cumulative_costs': cumulative_costs,
            'n_trades': len(trade_log)
        }
        
        logger.info(
            f"Backtest complete: Final capital=${capital:,.2f}, "
            f"Return={results['total_return']:.2%}, Trades={results['n_trades']}"
        )
        
        return results
    
    def generate_report(self,
                       results: Dict,
                       output_dir: str = "results/backtests") -> None:
        """
        Generate comprehensive backtest report.
        
        Args:
            results: Backtest results dictionary
            output_dir: Output directory for results
        """
        ensure_dir(output_dir)
        
        pair_str = f"{results['pair'][0]}_{results['pair'][1]}"
        
        # Save equity curve
        results['equity_curve'].to_csv(
            f"{output_dir}/{pair_str}_equity_curve.csv"
        )
        
        # Save trade log
        if len(results['trade_log']) > 0:
            results['trade_log'].to_csv(
                f"{output_dir}/{pair_str}_trade_log.csv",
                index=False
            )
        
        # Save signals
        results['signals'].to_csv(
            f"{output_dir}/{pair_str}_signals.csv"
        )
        
        logger.info(f"Saved backtest results to {output_dir}")
