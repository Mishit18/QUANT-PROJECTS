"""
Market Environment

Combines price process and order flow into unified market simulation.
"""

import numpy as np
from typing import Optional, Dict, Any
from .price_process import ArithmeticBrownianMotion
from .order_flow import OrderFlow


class MarketEnvironment:
    """
    Complete market environment for market making simulation.
    
    Combines:
    - Mid-price dynamics (stochastic process)
    - Order flow (Poisson arrivals)
    - Adverse selection effects
    """
    
    def __init__(
        self,
        initial_price: float,
        volatility: float,
        dt: float,
        intensity_model,
        adverse_selection_coef: float = 0.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize market environment.
        
        Args:
            initial_price: Starting mid-price
            volatility: Price volatility
            dt: Time step
            intensity_model: Order arrival intensity model
            adverse_selection_coef: Adverse selection coefficient
            random_seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(random_seed)
        
        # Price process
        self.price_process = ArithmeticBrownianMotion(
            initial_price=initial_price,
            volatility=volatility,
            dt=dt
        )
        
        # Order flow
        self.order_flow = OrderFlow(
            intensity_model=intensity_model,
            adverse_selection_coef=adverse_selection_coef
        )
        
        self.dt = dt
        self.current_time = 0.0
        
    @property
    def mid_price(self) -> float:
        """Get current mid-price."""
        return self.price_process.current_price
    
    def step(
        self,
        bid_price: float,
        ask_price: float
    ) -> Dict[str, Any]:
        """
        Simulate one time step of market dynamics.
        
        Args:
            bid_price: Market maker's bid quote
            ask_price: Market maker's ask quote
        
        Returns:
            Dictionary with step results:
                - mid_price: New mid-price
                - bid_filled: Whether bid was filled
                - ask_filled: Whether ask was filled
                - bid_fill_price: Execution price for bid (if filled)
                - ask_fill_price: Execution price for ask (if filled)
                - adverse_selection_impact: Price impact from fills
        """
        # Update mid-price
        old_mid = self.mid_price
        new_mid = self.price_process.step(self.rng)
        
        # Calculate spreads
        bid_spread = old_mid - bid_price
        ask_spread = ask_price - old_mid
        
        # Simulate order arrivals
        bid_filled, ask_filled = self.order_flow.simulate_arrivals(
            bid_spread=bid_spread,
            ask_spread=ask_spread,
            dt=self.dt,
            rng=self.rng
        )
        
        # Apply adverse selection
        total_impact = 0.0
        
        if bid_filled:
            impact = self.order_flow.adverse_selection_impact('buy', old_mid, self.rng)
            total_impact += impact
            self.order_flow.record_order(self.current_time, 'buy', bid_price)
        
        if ask_filled:
            impact = self.order_flow.adverse_selection_impact('sell', old_mid, self.rng)
            total_impact += impact
            self.order_flow.record_order(self.current_time, 'sell', ask_price)
        
        # Update price with adverse selection
        if total_impact != 0.0:
            self.price_process.current_price += total_impact
            new_mid = self.price_process.current_price
        
        self.current_time += self.dt
        
        return {
            'time': self.current_time,
            'mid_price': new_mid,
            'bid_filled': bid_filled,
            'ask_filled': ask_filled,
            'bid_fill_price': bid_price if bid_filled else None,
            'ask_fill_price': ask_price if ask_filled else None,
            'adverse_selection_impact': total_impact
        }
    
    def reset(self):
        """Reset environment to initial state."""
        self.price_process.reset()
        self.order_flow.reset()
        self.current_time = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current market state.
        
        Returns:
            Dictionary with current state information
        """
        return {
            'time': self.current_time,
            'mid_price': self.mid_price,
            'dt': self.dt
        }
