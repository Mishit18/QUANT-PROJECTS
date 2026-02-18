"""Volume-Weighted Average Price (VWAP) execution strategy."""

import numpy as np


class VWAP:
    """
    VWAP: Trade proportional to expected volume profile.
    
    Simplified: assumes U-shaped intraday volume pattern.
    """
    
    def __init__(self, initial_inventory: float, num_steps: int):
        """
        Args:
            initial_inventory: Total inventory to liquidate
            num_steps: Number of execution periods
        """
        self.initial_inventory = initial_inventory
        self.num_steps = num_steps
        
        # Generate U-shaped volume profile
        self.volume_profile = self._generate_volume_profile()
        self.trade_schedule = initial_inventory * self.volume_profile
        self.current_step = 0
    
    def _generate_volume_profile(self) -> np.ndarray:
        """
        Generate U-shaped volume profile.
        
        Higher volume at open and close, lower in middle.
        """
        t = np.linspace(0, 1, self.num_steps)
        
        # U-shape: high at 0 and 1, low at 0.5
        profile = 1.0 + 0.5 * (2 * t - 1) ** 2
        
        # Normalize to sum to 1
        profile /= np.sum(profile)
        
        return profile
    
    def reset(self):
        """Reset strategy."""
        self.current_step = 0
    
    def get_action(self, state: np.ndarray) -> float:
        """
        Get next trade size based on volume profile.
        
        Args:
            state: Environment state
        
        Returns:
            Trade size as fraction of remaining inventory
        """
        remaining_inventory = state[0] * self.initial_inventory
        
        if remaining_inventory < 1e-6 or self.current_step >= self.num_steps:
            return 0.0
        
        # Trade according to volume profile
        target_trade = self.trade_schedule[self.current_step]
        trade_fraction = target_trade / remaining_inventory
        trade_fraction = min(trade_fraction, 1.0)
        
        self.current_step += 1
        return trade_fraction
