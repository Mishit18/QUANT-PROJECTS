"""
Execution simulation module.

Models realistic order execution including transaction costs, slippage,
and market impact for backtesting purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import logging

from utils import setup_logging


logger = setup_logging()


class ExecutionSimulator:
    """
    Simulates realistic order execution for backtesting.
    
    Includes:
    - Transaction costs (commissions/fees)
    - Slippage
    - Market impact
    """
    
    def __init__(self,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005,
                 market_impact: float = 0.0002):
        """
        Initialize execution simulator.
        
        Args:
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 10 bps)
            slippage: Slippage as fraction
            market_impact: Market impact as fraction
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.market_impact = market_impact
        
        self.total_cost = transaction_cost + slippage + market_impact
    
    def compute_execution_cost(self,
                              notional: float,
                              side: str = 'buy') -> float:
        """
        Compute total execution cost for a trade.
        
        Args:
            notional: Notional value of trade
            side: 'buy' or 'sell'
            
        Returns:
            Total execution cost
        """
        cost = abs(notional) * self.total_cost
        return cost
    
    def execute_spread_trade(self,
                            position_change: float,
                            price_y: float,
                            price_x: float,
                            hedge_ratio: float,
                            capital: float) -> Dict:
        """
        Execute spread trade (simultaneous trades in both legs).
        
        For long spread: buy Y, sell X
        For short spread: sell Y, buy X
        
        Args:
            position_change: Change in position (-1, 0, or 1)
            price_y: Current price of Y
            price_x: Current price of X
            hedge_ratio: Hedge ratio (units of X per unit of Y)
            capital: Available capital
            
        Returns:
            Dictionary with execution details
        """
        if position_change == 0:
            return {
                'executed': False,
                'cost': 0,
                'notional_y': 0,
                'notional_x': 0,
                'shares_y': 0,
                'shares_x': 0
            }
        
        # Determine position size based on capital
        # Allocate capital to both legs
        notional_y = capital * 0.5  # 50% to Y
        notional_x = capital * 0.5  # 50% to X
        
        # Compute shares
        shares_y = notional_y / price_y
        shares_x = notional_x / price_x
        
        # Adjust for hedge ratio
        # We want: shares_y = hedge_ratio * shares_x
        # Given capital constraint, solve for optimal allocation
        total_value = price_y + hedge_ratio * price_x
        shares_y = capital / total_value
        shares_x = hedge_ratio * shares_y
        
        # Apply position direction
        shares_y *= position_change
        shares_x *= -position_change  # Opposite direction
        
        # Compute notionals
        notional_y = shares_y * price_y
        notional_x = shares_x * price_x
        
        # Compute execution costs
        cost_y = self.compute_execution_cost(notional_y)
        cost_x = self.compute_execution_cost(notional_x)
        total_cost = cost_y + cost_x
        
        execution_details = {
            'executed': True,
            'cost': total_cost,
            'notional_y': notional_y,
            'notional_x': notional_x,
            'shares_y': shares_y,
            'shares_x': shares_x,
            'price_y': price_y,
            'price_x': price_x,
            'hedge_ratio': hedge_ratio
        }
        
        return execution_details
    
    def compute_pnl(self,
                   shares_y: float,
                   shares_x: float,
                   entry_price_y: float,
                   entry_price_x: float,
                   exit_price_y: float,
                   exit_price_x: float) -> Dict:
        """
        Compute P&L for a closed position.
        
        Args:
            shares_y: Shares of Y held
            shares_x: Shares of X held
            entry_price_y: Entry price of Y
            entry_price_x: Entry price of X
            exit_price_y: Exit price of Y
            exit_price_x: Exit price of X
            
        Returns:
            Dictionary with P&L details
        """
        # P&L from Y
        pnl_y = shares_y * (exit_price_y - entry_price_y)
        
        # P&L from X
        pnl_x = shares_x * (exit_price_x - entry_price_x)
        
        # Total P&L
        total_pnl = pnl_y + pnl_x
        
        # Return on notional
        entry_notional = abs(shares_y * entry_price_y) + abs(shares_x * entry_price_x)
        if entry_notional > 0:
            return_pct = total_pnl / entry_notional
        else:
            return_pct = 0
        
        pnl_details = {
            'pnl_y': pnl_y,
            'pnl_x': pnl_x,
            'total_pnl': total_pnl,
            'entry_notional': entry_notional,
            'return_pct': return_pct
        }
        
        return pnl_details


class PositionSizer:
    """
    Determines position sizes based on risk management rules.
    """
    
    def __init__(self,
                 method: str = 'volatility_target',
                 volatility_target: float = 0.10,
                 max_leverage: float = 2.0,
                 max_position_size: float = 0.2):
        """
        Initialize position sizer.
        
        Args:
            method: 'equal_weight' or 'volatility_target'
            volatility_target: Target volatility (annualized)
            max_leverage: Maximum leverage allowed
            max_position_size: Maximum position size as fraction of capital
        """
        self.method = method
        self.volatility_target = volatility_target
        self.max_leverage = max_leverage
        self.max_position_size = max_position_size
    
    def compute_position_size(self,
                            capital: float,
                            spread_volatility: float,
                            n_positions: int = 1) -> float:
        """
        Compute position size based on risk management rules.
        
        Args:
            capital: Available capital
            spread_volatility: Spread volatility (annualized)
            n_positions: Number of concurrent positions
            
        Returns:
            Position size (capital to allocate)
        """
        if self.method == 'equal_weight':
            # Equal weight across positions
            position_size = capital * self.max_position_size / n_positions
        
        elif self.method == 'volatility_target':
            # Size position to target volatility
            if spread_volatility > 0:
                position_size = (self.volatility_target / spread_volatility) * capital
            else:
                position_size = capital * self.max_position_size
            
            # Apply maximum position size constraint
            position_size = min(position_size, capital * self.max_position_size)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Apply leverage constraint
        position_size = min(position_size, capital * self.max_leverage)
        
        return position_size
