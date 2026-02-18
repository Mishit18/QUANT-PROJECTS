"""
Portfolio Manager for Multi-Pair Statistical Arbitrage.

Implements:
- Volatility-targeted position sizing
- Risk-parity allocation across pairs
- Portfolio-level risk controls
- Capital recycling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

from utils import setup_logging

logger = setup_logging()


@dataclass
class PairPosition:
    """Track position for a single pair."""
    pair: Tuple[str, str]
    direction: int  # 1 for long spread, -1 for short spread
    entry_date: pd.Timestamp
    entry_zscore: float
    shares_y: float
    shares_x: float
    notional: float
    target_vol: float
    

class PortfolioManager:
    """
    Manages portfolio of cointegrated pairs with volatility targeting.
    
    Key Features:
    - Dynamic position sizing based on spread volatility
    - Portfolio-level volatility targeting
    - Risk-parity allocation
    - Capital recycling
    """
    
    def __init__(self,
                 initial_capital: float,
                 target_portfolio_vol: float = 0.10,
                 max_pairs: int = 5,
                 max_pair_weight: float = 0.30,
                 vol_lookback: int = 60):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            target_portfolio_vol: Target annualized portfolio volatility (e.g., 0.10 = 10%)
            max_pairs: Maximum concurrent positions
            max_pair_weight: Maximum weight per pair (e.g., 0.30 = 30%)
            vol_lookback: Lookback window for volatility estimation
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.target_portfolio_vol = target_portfolio_vol
        self.max_pairs = max_pairs
        self.max_pair_weight = max_pair_weight
        self.vol_lookback = vol_lookback
        
        self.positions: Dict[Tuple[str, str], PairPosition] = {}
        self.pair_allocations: Dict[Tuple[str, str], float] = {}
        
        logger.info(
            f"Portfolio Manager initialized: "
            f"capital=${initial_capital:,.0f}, "
            f"target_vol={target_portfolio_vol:.1%}, "
            f"max_pairs={max_pairs}"
        )
    
    def compute_volatility_adjusted_size(self,
                                        spread: pd.Series,
                                        price_y: float,
                                        price_x: float,
                                        hedge_ratio: float) -> float:
        """
        Compute position size using volatility targeting.
        
        Logic:
        1. Estimate spread volatility (rolling std)
        2. Scale position inversely with volatility
        3. Target fixed dollar volatility contribution
        
        Formula:
            position_size = (target_vol * capital) / (spread_vol * sqrt(252))
        
        Args:
            spread: Spread time series
            price_y: Current price of Y
            price_x: Current price of X
            hedge_ratio: Current hedge ratio
            
        Returns:
            Notional position size in dollars
        """
        # Estimate spread volatility
        spread_returns = spread.pct_change().dropna()
        
        if len(spread_returns) < self.vol_lookback:
            logger.warning(f"Insufficient data for vol estimation: {len(spread_returns)} < {self.vol_lookback}")
            return 0.0
        
        # Rolling volatility (annualized)
        recent_returns = spread_returns.tail(self.vol_lookback)
        spread_vol = recent_returns.std() * np.sqrt(252)
        
        if spread_vol == 0 or np.isnan(spread_vol):
            logger.warning("Zero or NaN spread volatility")
            return 0.0
        
        # Available capital (considering existing positions)
        available_capital = self.capital * (1 - sum(self.pair_allocations.values()))
        
        # Target dollar volatility for this pair
        # Allocate risk equally across max_pairs
        pair_target_vol = self.target_portfolio_vol / np.sqrt(self.max_pairs)
        target_dollar_vol = available_capital * pair_target_vol
        
        # Position size = target_vol / spread_vol
        # This ensures each pair contributes equally to portfolio risk
        notional = target_dollar_vol / spread_vol
        
        # Cap at max_pair_weight
        max_notional = self.capital * self.max_pair_weight
        notional = min(notional, max_notional)
        
        logger.info(
            f"Vol-adjusted sizing: spread_vol={spread_vol:.2%}, "
            f"target_vol={pair_target_vol:.2%}, notional=${notional:,.0f}"
        )
        
        return notional
    
    def compute_shares(self,
                      notional: float,
                      price_y: float,
                      price_x: float,
                      hedge_ratio: float,
                      direction: int) -> Tuple[float, float]:
        """
        Convert notional to share quantities.
        
        For long spread (direction=1): Long Y, Short X
        For short spread (direction=-1): Short Y, Long X
        
        Args:
            notional: Target notional exposure
            price_y: Price of Y
            price_x: Price of X
            hedge_ratio: Hedge ratio (beta)
            direction: 1 for long spread, -1 for short
            
        Returns:
            (shares_y, shares_x)
        """
        # Notional split between Y and X based on hedge ratio
        # Total notional = |shares_y * price_y| + |shares_x * price_x|
        # Constraint: shares_y / shares_x = hedge_ratio
        
        # Solve: shares_y = notional / (price_y + hedge_ratio * price_x)
        shares_y = notional / (price_y + hedge_ratio * price_x)
        shares_x = shares_y * hedge_ratio
        
        # Apply direction
        shares_y *= direction
        shares_x *= -direction  # Opposite side
        
        return shares_y, shares_x
    
    def can_open_position(self, pair: Tuple[str, str]) -> bool:
        """
        Check if we can open a new position.
        
        Conditions:
        - Not already in this pair
        - Haven't hit max_pairs limit
        - Have available capital
        """
        if pair in self.positions:
            return False
        
        if len(self.positions) >= self.max_pairs:
            logger.info(f"Max pairs limit reached: {len(self.positions)}/{self.max_pairs}")
            return False
        
        allocated = sum(self.pair_allocations.values())
        if allocated >= 0.95:  # Keep 5% buffer
            logger.info(f"Capital fully allocated: {allocated:.1%}")
            return False
        
        return True
    
    def open_position(self,
                     pair: Tuple[str, str],
                     direction: int,
                     entry_date: pd.Timestamp,
                     entry_zscore: float,
                     spread: pd.Series,
                     price_y: float,
                     price_x: float,
                     hedge_ratio: float) -> Optional[PairPosition]:
        """
        Open a new position with volatility-adjusted sizing.
        
        Args:
            pair: (ticker_y, ticker_x)
            direction: 1 for long spread, -1 for short
            entry_date: Entry timestamp
            entry_zscore: Entry z-score
            spread: Spread time series
            price_y: Current price of Y
            price_x: Current price of X
            hedge_ratio: Hedge ratio
            
        Returns:
            PairPosition if opened, None otherwise
        """
        if not self.can_open_position(pair):
            return None
        
        # Compute volatility-adjusted size
        notional = self.compute_volatility_adjusted_size(
            spread, price_y, price_x, hedge_ratio
        )
        
        if notional < 1000:  # Minimum position size
            logger.info(f"Position too small: ${notional:.0f}")
            return None
        
        # Convert to shares
        shares_y, shares_x = self.compute_shares(
            notional, price_y, price_x, hedge_ratio, direction
        )
        
        # Create position
        position = PairPosition(
            pair=pair,
            direction=direction,
            entry_date=entry_date,
            entry_zscore=entry_zscore,
            shares_y=shares_y,
            shares_x=shares_x,
            notional=notional,
            target_vol=self.target_portfolio_vol / np.sqrt(self.max_pairs)
        )
        
        # Update tracking
        self.positions[pair] = position
        self.pair_allocations[pair] = notional / self.capital
        
        logger.info(
            f"OPENED {pair}: direction={direction}, "
            f"notional=${notional:,.0f}, z={entry_zscore:.2f}"
        )
        
        return position
    
    def close_position(self,
                      pair: Tuple[str, str],
                      exit_date: pd.Timestamp,
                      exit_zscore: float,
                      price_y: float,
                      price_x: float,
                      reason: str = "signal") -> Optional[float]:
        """
        Close an existing position and realize PnL.
        
        Args:
            pair: Pair to close
            exit_date: Exit timestamp
            exit_zscore: Exit z-score
            price_y: Current price of Y
            price_x: Current price of X
            reason: Reason for exit
            
        Returns:
            Realized PnL
        """
        if pair not in self.positions:
            return None
        
        position = self.positions[pair]
        
        # Calculate PnL
        # Long spread: profit when spread narrows (Y outperforms X)
        # Short spread: profit when spread widens (X outperforms Y)
        pnl_y = position.shares_y * price_y
        pnl_x = position.shares_x * price_x
        pnl = pnl_y + pnl_x
        
        # Update capital
        self.capital += pnl
        
        # Remove position
        del self.positions[pair]
        del self.pair_allocations[pair]
        
        days_held = (exit_date - position.entry_date).days
        
        logger.info(
            f"CLOSED {pair}: reason={reason}, "
            f"pnl=${pnl:,.2f}, days={days_held}, "
            f"entry_z={position.entry_zscore:.2f}, exit_z={exit_zscore:.2f}"
        )
        
        return pnl
    
    def get_portfolio_statistics(self) -> Dict:
        """
        Compute current portfolio statistics.
        
        Returns:
            Dictionary with portfolio metrics
        """
        stats = {
            'capital': self.capital,
            'return': (self.capital / self.initial_capital - 1),
            'num_positions': len(self.positions),
            'capital_allocated': sum(self.pair_allocations.values()),
            'capital_available': 1 - sum(self.pair_allocations.values()),
        }
        
        if self.positions:
            stats['pairs'] = list(self.positions.keys())
            stats['notionals'] = {
                pair: pos.notional for pair, pos in self.positions.items()
            }
        
        return stats
    
    def check_regime_stability(self,
                              spread: pd.Series,
                              vol_threshold: float = 3.0) -> bool:
        """
        Check if spread is in stable regime.
        
        Filters:
        1. Volatility explosion (vol > threshold * historical vol)
        2. Extreme z-scores (potential cointegration breakdown)
        
        Args:
            spread: Spread time series
            vol_threshold: Multiplier for vol explosion detection
            
        Returns:
            True if stable, False if unstable
        """
        if len(spread) < self.vol_lookback * 2:
            return True  # Not enough data to judge
        
        # Recent vs historical volatility
        recent_vol = spread.tail(self.vol_lookback // 2).std()
        historical_vol = spread.tail(self.vol_lookback).std()
        
        if recent_vol > vol_threshold * historical_vol:
            logger.warning(
                f"Volatility explosion detected: "
                f"recent={recent_vol:.4f}, historical={historical_vol:.4f}"
            )
            return False
        
        return True
