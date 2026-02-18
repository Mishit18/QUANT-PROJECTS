"""
Order Flow Simulation

Models for simulating order arrivals and executions.
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class Order:
    """Represents a market order."""
    time: float
    side: str  # 'buy' or 'sell'
    price: float
    size: int = 1


class OrderFlow:
    """
    Simulates order arrivals using Poisson processes.
    
    Orders arrive according to quote-dependent intensities:
        - Buy orders arrive at rate λ^bid(δ^bid)
        - Sell orders arrive at rate λ^ask(δ^ask)
    
    where δ^bid, δ^ask are the spreads from mid-price.
    """
    
    def __init__(self, intensity_model, adverse_selection_coef: float = 0.0):
        """
        Initialize order flow simulator.
        
        Args:
            intensity_model: Model for arrival intensities (e.g., ExponentialIntensity)
            adverse_selection_coef: Coefficient for adverse selection impact
        """
        self.intensity_model = intensity_model
        self.adverse_selection_coef = adverse_selection_coef
        self.order_history: List[Order] = []
    
    def simulate_arrivals(
        self,
        bid_spread: float,
        ask_spread: float,
        dt: float,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[bool, bool]:
        """
        Simulate order arrivals in time interval dt.
        
        Uses thinning method for Poisson process:
        1. Generate potential arrival from max intensity
        2. Accept with probability λ(δ)/λ_max
        
        Args:
            bid_spread: Current bid spread from mid
            ask_spread: Current ask spread from mid
            dt: Time interval
            rng: Random number generator (optional)
        
        Returns:
            (bid_filled, ask_filled) - boolean indicators
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Calculate intensities
        lambda_bid = self.intensity_model.intensity(bid_spread)
        lambda_ask = self.intensity_model.intensity(ask_spread)
        
        # Probability of arrival in dt (Poisson approximation)
        prob_bid = 1.0 - np.exp(-lambda_bid * dt)
        prob_ask = 1.0 - np.exp(-lambda_ask * dt)
        
        # Sample arrivals
        bid_filled = rng.random() < prob_bid
        ask_filled = rng.random() < prob_ask
        
        return bid_filled, ask_filled
    
    def adverse_selection_impact(
        self,
        side: str,
        mid_price: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Calculate adverse selection price impact.
        
        When an order fills, it may contain information about future price movement.
        Model: price jumps in direction of the trade.
        
        Args:
            side: 'buy' or 'sell' from market maker perspective
            mid_price: Current mid-price
            rng: Random number generator (optional)
        
        Returns:
            Price impact (signed)
        """
        if self.adverse_selection_coef == 0.0:
            return 0.0
        
        if rng is None:
            rng = np.random.default_rng()
        
        # Adverse selection: price moves against market maker
        # If MM sells (buy order arrives), price tends to go up
        # If MM buys (sell order arrives), price tends to go down
        
        direction = 1.0 if side == 'sell' else -1.0
        impact = direction * self.adverse_selection_coef * mid_price * rng.normal(1.0, 0.5)
        
        return impact
    
    def record_order(self, time: float, side: str, price: float, size: int = 1):
        """
        Record an executed order.
        
        Args:
            time: Execution time
            side: 'buy' or 'sell'
            price: Execution price
            size: Order size
        """
        order = Order(time=time, side=side, price=price, size=size)
        self.order_history.append(order)
    
    def get_order_history(self) -> List[Order]:
        """Get history of executed orders."""
        return self.order_history
    
    def reset(self):
        """Reset order history."""
        self.order_history = []


class MultiAgentOrderFlow:
    """
    Order flow for multi-agent market making.
    
    Orders are allocated to the agent with the best quote.
    If multiple agents have the same best quote, allocation is random.
    """
    
    def __init__(self, intensity_model, adverse_selection_coef: float = 0.0):
        """
        Initialize multi-agent order flow.
        
        Args:
            intensity_model: Model for arrival intensities
            adverse_selection_coef: Adverse selection coefficient
        """
        self.intensity_model = intensity_model
        self.adverse_selection_coef = adverse_selection_coef
    
    def allocate_orders(
        self,
        bid_quotes: List[Tuple[int, float]],  # List of (agent_id, bid_price)
        ask_quotes: List[Tuple[int, float]],  # List of (agent_id, ask_price)
        dt: float,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Allocate orders to agents based on best quotes with proper competition.
        
        Price-time priority:
        1. Best price gets priority
        2. If tied, random allocation (pro-rata in real markets)
        3. Multiple agents can fill if order flow is sufficient
        
        Args:
            bid_quotes: List of (agent_id, bid_price) tuples
            ask_quotes: List of (agent_id, ask_price) tuples
            dt: Time interval
            rng: Random number generator (optional)
        
        Returns:
            (bid_winner_id, ask_winner_id) - IDs of agents who got filled (None if no fill)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        if not bid_quotes or not ask_quotes:
            return None, None
        
        # Find best bid (highest) and best ask (lowest)
        best_bid_price = max(price for _, price in bid_quotes)
        best_ask_price = min(price for _, price in ask_quotes)
        
        # Calculate spreads from mid-price
        mid_price = (best_bid_price + best_ask_price) / 2.0
        bid_spread = mid_price - best_bid_price
        ask_spread = best_ask_price - mid_price
        
        # Simulate arrivals based on BEST quotes (competition effect)
        # Tighter spreads from competition increase fill probability
        lambda_bid = self.intensity_model.intensity(max(bid_spread, 0.001))
        lambda_ask = self.intensity_model.intensity(max(ask_spread, 0.001))
        
        prob_bid = 1.0 - np.exp(-lambda_bid * dt)
        prob_ask = 1.0 - np.exp(-lambda_ask * dt)
        
        bid_arrival = rng.random() < prob_bid
        ask_arrival = rng.random() < prob_ask
        
        # Allocate to agents at best price
        bid_winner = None
        ask_winner = None
        
        if bid_arrival:
            # Find all agents at best bid (price-time priority)
            best_bidders = [agent_id for agent_id, price in bid_quotes 
                           if abs(price - best_bid_price) < 1e-10]
            if best_bidders:
                # Random allocation among tied agents (pro-rata approximation)
                bid_winner = rng.choice(best_bidders)
        
        if ask_arrival:
            # Find all agents at best ask
            best_askers = [agent_id for agent_id, price in ask_quotes 
                          if abs(price - best_ask_price) < 1e-10]
            if best_askers:
                ask_winner = rng.choice(best_askers)
        
        return bid_winner, ask_winner
    
    def adverse_selection_impact(
        self,
        side: str,
        mid_price: float,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """Calculate adverse selection impact (same as single-agent)."""
        if self.adverse_selection_coef == 0.0:
            return 0.0
        
        if rng is None:
            rng = np.random.default_rng()
        
        direction = 1.0 if side == 'sell' else -1.0
        impact = direction * self.adverse_selection_coef * mid_price * rng.normal(1.0, 0.5)
        
        return impact
