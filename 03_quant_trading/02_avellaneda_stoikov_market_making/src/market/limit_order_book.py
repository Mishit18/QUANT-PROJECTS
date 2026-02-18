"""
Limit Order Book with Queue Position Dynamics

Implements realistic queue position tracking and fill probability.
No calibration - purely structural model.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Quote:
    """Represents a limit order quote."""
    price: float
    size: int
    queue_position: int = 0  # Position in queue at this price level


class SimpleLOB:
    """
    Simplified limit order book with queue dynamics.
    
    Key features:
    - Queue position affects fill probability
    - Orders at better prices fill first
    - Queue position degrades with market activity
    
    No calibration - uses structural assumptions only.
    """
    
    def __init__(self, tick_size: float = 0.01):
        """
        Initialize LOB.
        
        Args:
            tick_size: Minimum price increment
        """
        self.tick_size = tick_size
        self.bid_depth = {}  # price -> total size at level
        self.ask_depth = {}  # price -> total size at level
        
    def compute_microprice(self, mid_price: float, bid_size: float, ask_size: float) -> float:
        """
        Compute microprice from order book imbalance.
        
        Microprice = bid * (ask_size / (bid_size + ask_size)) + 
                     ask * (bid_size / (bid_size + ask_size))
        
        This is a better estimate of "fair value" than mid-price.
        
        Args:
            mid_price: Current mid-price
            bid_size: Size at best bid
            ask_size: Size at best ask
        
        Returns:
            Microprice estimate
        """
        if bid_size + ask_size == 0:
            return mid_price
        
        # Imbalance-weighted price
        bid_weight = ask_size / (bid_size + ask_size)
        ask_weight = bid_size / (bid_size + ask_size)
        
        # Approximate bid/ask from mid
        bid_price = mid_price - self.tick_size / 2
        ask_price = mid_price + self.tick_size / 2
        
        microprice = bid_price * bid_weight + ask_price * ask_weight
        return microprice
    
    def queue_position_fill_probability(
        self,
        quote_price: float,
        best_price: float,
        queue_position: int,
        base_intensity: float,
        dt: float,
        is_bid: bool = True
    ) -> float:
        """
        Calculate fill probability accounting for queue position.
        
        Logic:
        - Orders at better prices fill first
        - Queue position reduces fill probability
        - No calibration - uses structural model
        
        Args:
            quote_price: Our quote price
            best_price: Best price in market
            queue_position: Our position in queue (0 = front)
            base_intensity: Base arrival rate
            dt: Time step
            is_bid: Whether this is a bid (vs ask)
        
        Returns:
            Fill probability in [0, 1]
        """
        # If we're not at best, probability is lower
        if is_bid:
            price_diff = best_price - quote_price
        else:
            price_diff = quote_price - best_price
        
        if price_diff > self.tick_size / 2:
            # Not at best price - much lower probability
            base_prob = 1.0 - np.exp(-base_intensity * dt * 0.1)
        else:
            # At best price - use base intensity
            base_prob = 1.0 - np.exp(-base_intensity * dt)
        
        # Queue position penalty: exponential decay
        # Front of queue (pos=0) gets full probability
        # Back of queue gets reduced probability
        queue_penalty = np.exp(-0.1 * queue_position)
        
        return base_prob * queue_penalty
    
    def update_queue_position(
        self,
        current_position: int,
        market_activity: float
    ) -> int:
        """
        Update queue position based on market activity.
        
        Queue position improves (moves forward) as orders ahead get filled.
        
        Args:
            current_position: Current position in queue
            market_activity: Measure of recent fills (0-1)
        
        Returns:
            Updated queue position
        """
        # Stochastic improvement based on market activity
        if current_position > 0 and np.random.random() < market_activity:
            return max(0, current_position - 1)
        return current_position
    
    def get_depth_at_level(self, price: float, is_bid: bool) -> int:
        """
        Get total size at a price level.
        
        Args:
            price: Price level
            is_bid: Whether bid side
        
        Returns:
            Total size at level
        """
        depth_dict = self.bid_depth if is_bid else self.ask_depth
        return depth_dict.get(price, 0)
    
    def add_order(self, price: float, size: int, is_bid: bool) -> int:
        """
        Add order to book and return queue position.
        
        Args:
            price: Order price
            size: Order size
            is_bid: Whether bid order
        
        Returns:
            Queue position (0 = front)
        """
        depth_dict = self.bid_depth if is_bid else self.ask_depth
        
        # Queue position is current depth at this level
        queue_pos = depth_dict.get(price, 0)
        
        # Add to depth
        depth_dict[price] = queue_pos + size
        
        return queue_pos
    
    def remove_order(self, price: float, size: int, is_bid: bool):
        """
        Remove order from book.
        
        Args:
            price: Order price
            size: Order size
            is_bid: Whether bid order
        """
        depth_dict = self.bid_depth if is_bid else self.ask_depth
        
        if price in depth_dict:
            depth_dict[price] = max(0, depth_dict[price] - size)
            if depth_dict[price] == 0:
                del depth_dict[price]
