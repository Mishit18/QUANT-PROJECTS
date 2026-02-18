"""
Inventory control for market-making agents.
"""
import numpy as np


class InventoryController:
    """Inventory-aware pricing using reservation price framework."""
    
    def __init__(self, risk_aversion, inventory_limit, price_volatility=1.0):
        """
        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter γ
        inventory_limit : int
            Maximum absolute inventory
        price_volatility : float
            Estimated price volatility σ
        """
        self.gamma = risk_aversion
        self.inventory_limit = inventory_limit
        self.sigma = price_volatility
        
        self.inventory = 0
        self.inventory_history = []
    
    def compute_reservation_price(self, mid_price, timestamp=None):
        """
        Compute reservation price with inventory penalty.
        
        r(t) = m(t) - γ * q(t) * σ²
        
        Parameters
        ----------
        mid_price : float
            Current mid price
        timestamp : float, optional
            Current time
        
        Returns
        -------
        reservation_price : float
        """
        penalty = self.gamma * self.inventory * (self.sigma ** 2)
        reservation_price = mid_price - penalty
        
        if timestamp is not None:
            self.inventory_history.append((timestamp, self.inventory, reservation_price))
        
        return reservation_price
    
    def compute_spread_adjustment(self):
        """
        Adjust spread based on inventory position.
        
        Returns
        -------
        bid_adjustment : float
            Adjustment to bid spread (negative = tighter)
        ask_adjustment : float
            Adjustment to ask spread (negative = tighter)
        """
        # Inventory ratio
        inv_ratio = self.inventory / self.inventory_limit if self.inventory_limit > 0 else 0.0
        
        # If long, widen ask and tighten bid to encourage selling
        # If short, widen bid and tighten ask to encourage buying
        if self.inventory > 0:
            bid_adjustment = -0.5 * inv_ratio  # Tighter bid
            ask_adjustment = 0.5 * inv_ratio    # Wider ask
        elif self.inventory < 0:
            bid_adjustment = -0.5 * inv_ratio   # Wider bid
            ask_adjustment = 0.5 * inv_ratio    # Tighter ask
        else:
            bid_adjustment = 0.0
            ask_adjustment = 0.0
        
        return bid_adjustment, ask_adjustment
    
    def update_inventory(self, trade_side, trade_size):
        """
        Update inventory after trade.
        
        Parameters
        ----------
        trade_side : Side
            Side of trade (BUY or SELL from agent perspective)
        trade_size : int
            Trade size
        """
        from lob.order import Side
        
        if trade_side == Side.BUY:
            self.inventory += trade_size
        else:
            self.inventory -= trade_size
    
    def check_inventory_limit(self):
        """
        Check if inventory is within limits.
        
        Returns
        -------
        can_buy : bool
            Whether agent can buy more
        can_sell : bool
            Whether agent can sell more
        """
        can_buy = self.inventory < self.inventory_limit
        can_sell = self.inventory > -self.inventory_limit
        
        return can_buy, can_sell
    
    def get_inventory_stats(self):
        """Get inventory statistics."""
        if len(self.inventory_history) == 0:
            return {}
        
        inventories = [inv for _, inv, _ in self.inventory_history]
        
        return {
            'current': self.inventory,
            'max': max(inventories),
            'min': min(inventories),
            'mean': np.mean(inventories),
            'std': np.std(inventories)
        }
