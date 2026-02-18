"""
Execution model - realistic but not alpha-killing.

Transaction costs matter but shouldn't dominate edge.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class ExecutionModel:
    """
    Realistic execution with costs that don't kill alpha.
    """
    
    def __init__(self, 
                 transaction_cost_bps: float = 3.0,
                 slippage_bps: float = 1.5):
        """
        Args:
            transaction_cost_bps: Commission + fees (3 bps = 0.03%)
            slippage_bps: Market impact (1.5 bps = 0.015%)
        """
        # PRODUCTION FIX: Defensive type casting
        try:
            self.tc_bps = float(transaction_cost_bps)
            self.slippage_bps = float(slippage_bps)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"ExecutionModel parameters must be numeric. Error: {e}"
            )
        
        # Validation
        if self.tc_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Transaction costs must be non-negative")
        
        # Convert to decimal
        self.tc_rate = self.tc_bps / 10000
        self.slippage_rate = self.slippage_bps / 10000
    
    def apply_costs(self, 
                    positions: pd.Series,
                    returns: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Apply transaction costs to returns.
        
        Args:
            positions: Position sizes (lagged by 1)
            returns: Spread returns
        
        Returns:
            (net_returns, cost_breakdown)
        """
        # Gross returns
        gross_returns = positions.shift(1) * returns
        gross_returns = gross_returns.fillna(0)
        
        # Calculate trades (position changes)
        trades = positions.diff().abs()
        trades = trades.fillna(0)
        
        # Transaction costs (both legs of spread)
        tc_cost = trades * self.tc_rate * 2  # 2x for both legs
        
        # Slippage (proportional to trade size)
        slippage_cost = trades * self.slippage_rate * 2
        
        # Total costs as return drag
        total_costs = tc_cost + slippage_cost
        
        # Net returns
        net_returns = gross_returns - total_costs
        
        # Cost breakdown
        breakdown = pd.DataFrame({
            'gross_returns': gross_returns,
            'transaction_costs': -tc_cost,
            'slippage': -slippage_cost,
            'total_costs': -total_costs,
            'net_returns': net_returns
        })
        
        return net_returns, breakdown
    
    def estimate_capacity(self, 
                         avg_daily_volume: float,
                         avg_position_size: float) -> float:
        """
        Estimate strategy capacity.
        
        Rule of thumb: Don't trade more than 1% of daily volume.
        
        Args:
            avg_daily_volume: Average daily dollar volume
            avg_position_size: Average position size
        
        Returns:
            Estimated capacity in dollars
        """
        # Conservative: 0.5% of daily volume
        max_position = avg_daily_volume * 0.005
        
        if avg_position_size == 0:
            return 0
        
        capacity = max_position / avg_position_size
        
        return capacity



class OUCollapseMonitor:
    """
    Mid-trade OU collapse detection.
    
    Production safety rule: If OU half-life collapses below threshold
    mid-trade, immediately exit to prevent:
    - Regime breakdown
    - Silent alpha decay
    - Holding positions in broken relationships
    """
    
    def __init__(self, min_half_life: float = 5.0, lookback_window: int = 60):
        """
        Args:
            min_half_life: Minimum acceptable half-life (days)
            lookback_window: Rolling window for OU estimation (days)
        """
        self.min_half_life = float(min_half_life)
        self.lookback_window = int(lookback_window)
        self.collapse_events = []
    
    def check_collapse(self, spread: pd.Series, current_idx: int) -> Tuple[bool, Optional[float]]:
        """
        Check if OU process has collapsed at current time.
        
        Args:
            spread: Full spread series
            current_idx: Current position in series
        
        Returns:
            (has_collapsed, current_half_life)
        """
        # Get recent window
        start_idx = max(0, current_idx - self.lookback_window)
        window = spread.iloc[start_idx:current_idx+1]
        
        if len(window) < 20:  # Need minimum data
            return False, None
        
        # Calculate rolling half-life
        half_life = self._calculate_half_life(window)
        
        if half_life is None or np.isinf(half_life):
            return True, None  # Collapse detected
        
        if half_life < self.min_half_life:
            self.collapse_events.append({
                'timestamp': spread.index[current_idx],
                'half_life': half_life,
                'reason': f'HL={half_life:.1f}d < {self.min_half_life}d'
            })
            return True, half_life
        
        return False, half_life
    
    def _calculate_half_life(self, spread: pd.Series) -> Optional[float]:
        """Calculate half-life using AR(1) model."""
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        df = pd.DataFrame({'spread_diff': spread_diff, 'spread_lag': spread_lag}).dropna()
        
        if len(df) < 10:
            return None
        
        X = df['spread_lag'].values.reshape(-1, 1)
        y = df['spread_diff'].values
        
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            lambda_param = beta[1]
            
            if lambda_param >= 0:
                return np.inf
            
            half_life = -np.log(2) / lambda_param
            return half_life
        except:
            return None
    
    def get_collapse_summary(self) -> pd.DataFrame:
        """Get summary of all collapse events."""
        if not self.collapse_events:
            return pd.DataFrame()
        
        return pd.DataFrame(self.collapse_events)
