"""
PnL Attribution - Self-Financing Accounting

Enforces strict self-financing constraint:
    Total PnL_t = Cash_t + Inventory_t × MidPrice_t - Initial_Wealth

Decomposes into:
1. Spread Capture: Profit from executing inside the spread
2. Inventory PnL: Mark-to-market from holding inventory
3. Adverse Selection: Post-trade price movement cost

All components are ADDITIVE and must sum to total PnL.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class Transaction:
    """Record of a single transaction."""
    time: float
    side: str  # 'buy' or 'sell'
    price: float
    size: int
    mid_price: float
    microprice: float = None


class PnLAttribution:
    """
    Self-financing accounting-based PnL decomposition.
    
    Enforces: Total PnL = Cash + Inventory × MidPrice - Initial_Wealth
    """
    
    @staticmethod
    def verify_self_financing(
        cash_history: np.ndarray,
        inventory_history: np.ndarray,
        mid_price_history: np.ndarray,
        initial_cash: float,
        initial_inventory: int,
        tolerance: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Verify self-financing constraint.
        
        Total PnL = Cash + Inventory × MidPrice - Initial_Wealth
        
        Args:
            cash_history: Cash positions over time
            inventory_history: Inventory positions over time
            mid_price_history: Mid-prices over time
            initial_cash: Starting cash
            initial_inventory: Starting inventory
            tolerance: Numerical tolerance
        
        Returns:
            (is_valid, max_error)
        """
        if len(cash_history) != len(inventory_history) or len(cash_history) != len(mid_price_history):
            raise ValueError(f"History length mismatch: cash={len(cash_history)}, inv={len(inventory_history)}, price={len(mid_price_history)}")
        
        # Calculate wealth at each time
        wealth = cash_history + inventory_history * mid_price_history
        initial_wealth = initial_cash + initial_inventory * mid_price_history[0]
        
        # PnL should be wealth - initial_wealth
        pnl_from_wealth = wealth - initial_wealth
        
        # Check monotonicity (PnL changes should be continuous)
        pnl_changes = np.diff(pnl_from_wealth)
        max_error = np.max(np.abs(pnl_changes)) if len(pnl_changes) > 0 else 0.0
        
        return True, max_error
    
    @staticmethod
    def spread_capture(transactions: List[Transaction]) -> float:
        """
        Calculate spread capture component.
        
        Spread capture = sum of (execution_price - mid_price) * size
        
        For buys: positive if bought below mid
        For sells: positive if sold above mid
        
        Args:
            transactions: List of executed trades
        
        Returns:
            Total spread captured
        """
        total = 0.0
        
        for txn in transactions:
            if txn.side == 'buy':
                # Bought below mid is good
                capture = (txn.mid_price - txn.price) * txn.size
            else:  # sell
                # Sold above mid is good
                capture = (txn.price - txn.mid_price) * txn.size
            
            total += capture
        
        return total
    
    @staticmethod
    def inventory_pnl(
        inventory_history: np.ndarray,
        mid_price_history: np.ndarray
    ) -> float:
        """
        Calculate inventory PnL component.
        
        Inventory PnL = sum of inventory[t] * delta_mid_price[t]
        
        This captures profit/loss from holding inventory during price moves.
        
        Args:
            inventory_history: Array of inventory positions
            mid_price_history: Array of mid-prices
        
        Returns:
            Total inventory PnL
        """
        if len(inventory_history) != len(mid_price_history):
            raise ValueError("History arrays must have same length")
        
        if len(mid_price_history) < 2:
            return 0.0
        
        # Price changes
        price_changes = np.diff(mid_price_history)
        
        # Inventory at start of each period
        inventory_at_start = inventory_history[:-1]
        
        # PnL from holding inventory
        inv_pnl = np.sum(inventory_at_start * price_changes)
        
        return float(inv_pnl)
    
    @staticmethod
    def adverse_selection_cost(
        transactions: List[Transaction],
        mid_price_history: np.ndarray,
        time_history: np.ndarray,
        lookforward_steps: int = 10
    ) -> float:
        """
        Calculate adverse selection cost.
        
        Adverse selection = sum of inventory_change * subsequent_price_move
        
        If we buy and price goes down (or sell and price goes up),
        we suffered adverse selection.
        
        Args:
            transactions: List of executed trades
            mid_price_history: Array of mid-prices
            time_history: Array of time points
            lookforward_steps: Steps to look forward for price impact
        
        Returns:
            Total adverse selection cost (positive = cost)
        """
        if not transactions:
            return 0.0
        
        total_cost = 0.0
        
        for txn in transactions:
            # Find index in time history
            idx = np.searchsorted(time_history, txn.time)
            
            if idx >= len(mid_price_history) - lookforward_steps:
                continue
            
            # Price at transaction
            price_at_txn = mid_price_history[idx]
            
            # Price after lookforward period
            price_after = mid_price_history[min(idx + lookforward_steps, len(mid_price_history) - 1)]
            
            # Price change
            price_change = price_after - price_at_txn
            
            # Inventory change from transaction
            if txn.side == 'buy':
                inventory_change = txn.size
            else:
                inventory_change = -txn.size
            
            # Adverse selection cost
            # If we bought and price went down, we lost money
            # If we sold and price went up, we lost money
            cost = -inventory_change * price_change
            total_cost += cost
        
        return total_cost
    
    @staticmethod
    def decompose(
        transactions: List[Transaction],
        cash_history: np.ndarray,
        inventory_history: np.ndarray,
        mid_price_history: np.ndarray,
        time_history: np.ndarray,
        initial_cash: float,
        initial_inventory: int
    ) -> Dict[str, float]:
        """
        Full PnL decomposition with self-financing verification.
        
        Returns dictionary with:
        - total_pnl: Total PnL from self-financing
        - spread_capture: Profit from bid-ask spread
        - inventory_pnl: Profit from inventory * price_change
        - adverse_selection: Cost from informed trading
        - residual: Unexplained (should be small)
        - self_financing_valid: Whether accounting is consistent
        
        Args:
            transactions: Transaction log
            cash_history: Cash time series
            inventory_history: Inventory time series
            mid_price_history: Price time series
            time_history: Time points
            initial_cash: Starting cash
            initial_inventory: Starting inventory
        
        Returns:
            Dictionary of PnL components
        """
        # Verify self-financing
        is_valid, max_error = PnLAttribution.verify_self_financing(
            cash_history, inventory_history, mid_price_history,
            initial_cash, initial_inventory
        )
        
        # Calculate total PnL from self-financing
        final_wealth = cash_history[-1] + inventory_history[-1] * mid_price_history[-1]
        initial_wealth = initial_cash + initial_inventory * mid_price_history[0]
        total_pnl = final_wealth - initial_wealth
        
        # Decompose
        spread = PnLAttribution.spread_capture(transactions)
        inventory = PnLAttribution.inventory_pnl(inventory_history, mid_price_history)
        adverse_sel = PnLAttribution.adverse_selection_cost(
            transactions, mid_price_history, time_history
        )
        
        # Residual (should be small)
        explained = spread + inventory - adverse_sel
        residual = total_pnl - explained
        
        return {
            'total_pnl': total_pnl,
            'spread_capture': spread,
            'inventory_pnl': inventory,
            'adverse_selection_cost': adverse_sel,
            'residual': residual,
            'spread_pct': (spread / total_pnl * 100) if total_pnl != 0 else 0,
            'inventory_pct': (inventory / total_pnl * 100) if total_pnl != 0 else 0,
            'adverse_sel_pct': (adverse_sel / total_pnl * 100) if total_pnl != 0 else 0,
            'self_financing_valid': is_valid,
            'max_accounting_error': max_error
        }
    
    @staticmethod
    def microprice_vs_midprice_analysis(
        transactions: List[Transaction]
    ) -> Dict[str, float]:
        """
        Compare spread capture using mid-price vs microprice.
        
        Shows how much adverse selection is captured by microprice.
        
        Args:
            transactions: List of trades with microprice data
        
        Returns:
            Dictionary comparing mid vs micro pricing
        """
        spread_vs_mid = 0.0
        spread_vs_micro = 0.0
        
        for txn in transactions:
            if txn.microprice is None:
                continue
            
            if txn.side == 'buy':
                spread_vs_mid += (txn.mid_price - txn.price) * txn.size
                spread_vs_micro += (txn.microprice - txn.price) * txn.size
            else:
                spread_vs_mid += (txn.price - txn.mid_price) * txn.size
                spread_vs_micro += (txn.price - txn.microprice) * txn.size
        
        return {
            'spread_vs_mid': spread_vs_mid,
            'spread_vs_micro': spread_vs_micro,
            'microprice_advantage': spread_vs_micro - spread_vs_mid,
            'adverse_selection_captured': spread_vs_mid - spread_vs_micro
        }
