"""
Competitive Market Maker

Extension for multi-agent competitive market making scenarios.
"""

import numpy as np
from typing import List, Dict, Any
from .market_maker import MarketMaker


class CompetitiveMarketMaker(MarketMaker):
    """
    Market maker in competitive environment.
    
    Extends base MarketMaker with:
    - Awareness of competitor quotes
    - Strategic quote adjustment
    - Competition metrics tracking
    """
    
    def __init__(
        self,
        model,
        initial_inventory: int = 0,
        initial_cash: float = 0.0,
        name: str = "CMM",
        aggressiveness: float = 1.0
    ):
        """
        Initialize competitive market maker.
        
        Args:
            model: Avellaneda-Stoikov model
            initial_inventory: Starting inventory
            initial_cash: Starting cash
            name: Agent name
            aggressiveness: Multiplier for quote adjustment (>1 = more aggressive)
        """
        super().__init__(model, initial_inventory, initial_cash, name)
        self.aggressiveness = aggressiveness
        
        # Competition tracking
        self.times_best_bid = 0
        self.times_best_ask = 0
        self.times_tied_bid = 0
        self.times_tied_ask = 0
    
    def competitive_quote(
        self,
        mid_price: float,
        time: float,
        competitor_bids: List[float] = None,
        competitor_asks: List[float] = None
    ) -> tuple[float, float]:
        """
        Generate quotes considering competitor quotes.
        
        Strategy:
        1. Calculate optimal AS quotes
        2. Add small random noise for realistic competition
        3. Apply aggressiveness factor
        
        Args:
            mid_price: Current mid-price
            time: Current time
            competitor_bids: List of competitor bid prices
            competitor_asks: List of competitor ask prices
        
        Returns:
            (bid_price, ask_price)
        """
        # Get base optimal quotes
        base_bid, base_ask = self.model.optimal_quotes(mid_price, self.inventory, time)
        
        # Add small random noise to create realistic competition
        # Without this, agents with similar γ would quote identically
        tick_size = 0.01
        noise_scale = 0.5  # Half a tick on average
        bid_noise = np.random.normal(0, noise_scale) * tick_size
        ask_noise = np.random.normal(0, noise_scale) * tick_size
        
        bid = base_bid + bid_noise
        ask = base_ask + ask_noise
        
        # Ensure bid < ask
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - tick_size
            ask = mid + tick_size
        
        # Record quotes
        self.mid_price_history.append(mid_price)
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        
        bid_spread = mid_price - bid
        ask_spread = ask - mid_price
        self.bid_spread_history.append(bid_spread)
        self.ask_spread_history.append(ask_spread)
        
        return bid, ask
    
    def update_competition_stats(
        self,
        my_bid: float,
        my_ask: float,
        all_bids: List[float],
        all_asks: List[float]
    ):
        """
        Update statistics about competitive position.
        
        Args:
            my_bid: This agent's bid
            my_ask: This agent's ask
            all_bids: All bids (including this agent's)
            all_asks: All asks (including this agent's)
        """
        if all_bids:
            best_bid = max(all_bids)
            if my_bid == best_bid:
                n_at_best = sum(1 for b in all_bids if b == best_bid)
                if n_at_best == 1:
                    self.times_best_bid += 1
                else:
                    self.times_tied_bid += 1
        
        if all_asks:
            best_ask = min(all_asks)
            if my_ask == best_ask:
                n_at_best = sum(1 for a in all_asks if a == best_ask)
                if n_at_best == 1:
                    self.times_best_ask += 1
                else:
                    self.times_tied_ask += 1
    
    def get_competition_metrics(self) -> Dict[str, Any]:
        """
        Get metrics related to competitive performance.
        
        Returns:
            Dictionary of competition metrics
        """
        total_quotes = len(self.bid_history)
        
        metrics = {
            'times_best_bid': self.times_best_bid,
            'times_best_ask': self.times_best_ask,
            'times_tied_bid': self.times_tied_bid,
            'times_tied_ask': self.times_tied_ask,
            'pct_best_bid': self.times_best_bid / total_quotes if total_quotes > 0 else 0.0,
            'pct_best_ask': self.times_best_ask / total_quotes if total_quotes > 0 else 0.0,
            'aggressiveness': self.aggressiveness
        }
        
        return metrics
    
    def get_metrics(self, mid_price: float) -> Dict[str, Any]:
        """
        Get all metrics including competition metrics.
        
        Args:
            mid_price: Current mid-price
        
        Returns:
            Combined metrics dictionary
        """
        base_metrics = super().get_metrics(mid_price)
        comp_metrics = self.get_competition_metrics()
        
        return {**base_metrics, **comp_metrics}
