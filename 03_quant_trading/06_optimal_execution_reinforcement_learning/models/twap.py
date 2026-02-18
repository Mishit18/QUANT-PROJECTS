"""Time-Weighted Average Price (TWAP) execution strategy."""

import numpy as np


class TWAP:
    """
    TWAP: Uniform liquidation over time horizon.
    
    Trade schedule: v_t = X_0 / N for all t
    """
    
    def __init__(self, initial_inventory: float, num_steps: int):
        """
        Args:
            initial_inventory: Total inventory to liquidate
            num_steps: Number of execution periods
        """
        self.initial_inventory = initial_inventory
        self.num_steps = num_steps
        self.trade_size = initial_inventory / num_steps
        self.current_step = 0
    
    def reset(self):
        """Reset strategy."""
        self.current_step = 0
    
    def get_action(self, state: np.ndarray) -> float:
        """
        Get next trade size.
        
        Args:
            state: Environment state (unused for TWAP)
        
        Returns:
            Trade size as fraction of remaining inventory
        """
        remaining_inventory = state[0] * self.initial_inventory
        
        if remaining_inventory < 1e-6:
            return 0.0
        
        # Trade uniform amount
        trade_fraction = self.trade_size / remaining_inventory
        trade_fraction = min(trade_fraction, 1.0)
        
        self.current_step += 1
        return trade_fraction
