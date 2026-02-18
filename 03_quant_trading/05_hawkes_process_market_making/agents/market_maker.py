"""
Market-making agent with inventory control and adverse selection filtering.
"""
import numpy as np
from lob.order import Side
from agents.inventory_control import InventoryController


class MarketMaker:
    """
    Market-making agent that quotes bid/ask with inventory awareness.
    """
    
    def __init__(self, lob, tick_size, config):
        """
        Parameters
        ----------
        lob : LimitOrderBook
            Order book instance
        tick_size : float
            Price tick size
        config : dict
            Agent configuration
        """
        self.lob = lob
        self.tick_size = tick_size
        
        # Configuration
        self.inventory_limit = config.get('inventory_limit', 100)
        self.risk_aversion = config.get('risk_aversion', 0.01)
        self.target_spread_ticks = config.get('target_spread_ticks', 2)
        self.min_spread_ticks = config.get('min_spread_ticks', 1)
        self.max_spread_ticks = config.get('max_spread_ticks', 5)
        self.quote_size = config.get('quote_size', 10)
        self.latency_ms = config.get('latency_ms', 1.0)
        self.adverse_selection_threshold = config.get('adverse_selection_threshold', 0.7)
        
        # Inventory controller
        self.inventory_controller = InventoryController(
            risk_aversion=self.risk_aversion,
            inventory_limit=self.inventory_limit,
            price_volatility=tick_size * 5  # Rough estimate
        )
        
        # Active quotes
        self.active_bid_id = None
        self.active_ask_id = None
        
        # PnL tracking
        self.cash = 0.0
        self.pnl_history = []
        self.trades = []
    
    def update_quotes(self, timestamp):
        """
        Update bid/ask quotes based on current market state.
        
        Parameters
        ----------
        timestamp : float
            Current time
        """
        # Cancel existing quotes
        self._cancel_quotes()
        
        # Get market state
        mid_price = self.lob.mid_price()
        if mid_price is None:
            return
        
        imbalance = self.lob.imbalance()
        
        # Compute reservation price
        reservation_price = self.inventory_controller.compute_reservation_price(
            mid_price, timestamp
        )
        
        # Determine spread
        spread_ticks = self._compute_spread(imbalance)
        half_spread = spread_ticks * self.tick_size / 2.0
        
        # Inventory adjustments
        bid_adj, ask_adj = self.inventory_controller.compute_spread_adjustment()
        
        # Quote prices
        bid_price = reservation_price - half_spread + bid_adj * self.tick_size
        ask_price = reservation_price + half_spread + ask_adj * self.tick_size
        
        # Round to tick
        bid_price = self._round_to_tick(bid_price)
        ask_price = self._round_to_tick(ask_price)
        
        # Check inventory limits
        can_buy, can_sell = self.inventory_controller.check_inventory_limit()
        
        # Place quotes
        if can_buy and bid_price > 0:
            self.active_bid_id, _ = self.lob.add_limit_order(
                Side.BUY, bid_price, self.quote_size, timestamp
            )
        
        if can_sell and ask_price > 0:
            self.active_ask_id, _ = self.lob.add_limit_order(
                Side.SELL, ask_price, self.quote_size, timestamp
            )
    
    def _compute_spread(self, imbalance):
        """
        Compute spread in ticks based on adverse selection risk.
        
        Parameters
        ----------
        imbalance : float
            Order book imbalance
        
        Returns
        -------
        spread_ticks : int
        """
        spread = self.target_spread_ticks
        
        # Widen spread if imbalance is high (adverse selection risk)
        if abs(imbalance) > self.adverse_selection_threshold:
            spread += 1
        
        # Clamp to limits
        spread = max(self.min_spread_ticks, min(self.max_spread_ticks, spread))
        
        return spread
    
    def _cancel_quotes(self):
        """Cancel active quotes."""
        if self.active_bid_id is not None:
            self.lob.cancel_order(self.active_bid_id)
            self.active_bid_id = None
        
        if self.active_ask_id is not None:
            self.lob.cancel_order(self.active_ask_id)
            self.active_ask_id = None
    
    def process_fill(self, order_id, fill_price, fill_size, timestamp):
        """
        Process order fill and update PnL.
        
        Parameters
        ----------
        order_id : int
            Filled order ID
        fill_price : float
            Fill price
        fill_size : int
            Fill size
        timestamp : float
            Fill time
        """
        # Determine side
        if order_id == self.active_bid_id:
            side = Side.BUY
            self.active_bid_id = None
        elif order_id == self.active_ask_id:
            side = Side.SELL
            self.active_ask_id = None
        else:
            return
        
        # Update cash
        if side == Side.BUY:
            self.cash -= fill_price * fill_size
        else:
            self.cash += fill_price * fill_size
        
        # Update inventory
        self.inventory_controller.update_inventory(side, fill_size)
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'side': side,
            'price': fill_price,
            'size': fill_size
        })
        
        # Update PnL
        mid_price = self.lob.mid_price()
        if mid_price is not None:
            inventory = self.inventory_controller.inventory
            pnl = self.cash + inventory * mid_price
            self.pnl_history.append((timestamp, pnl, inventory))
    
    def mark_to_market(self, timestamp):
        """
        Mark position to market and record PnL.
        
        Parameters
        ----------
        timestamp : float
            Current time
        """
        mid_price = self.lob.mid_price()
        if mid_price is not None:
            inventory = self.inventory_controller.inventory
            pnl = self.cash + inventory * mid_price
            self.pnl_history.append((timestamp, pnl, inventory))
    
    def _round_to_tick(self, price):
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size
    
    def get_performance_summary(self):
        """
        Get performance summary.
        
        Returns
        -------
        summary : dict
            Performance metrics
        """
        if len(self.pnl_history) == 0:
            return {}
        
        times, pnls, inventories = zip(*self.pnl_history)
        pnls = np.array(pnls)
        
        # PnL metrics
        final_pnl = pnls[-1]
        max_pnl = np.max(pnls)
        min_pnl = np.min(pnls)
        
        # Drawdown
        running_max = np.maximum.accumulate(pnls)
        drawdowns = running_max - pnls
        max_drawdown = np.max(drawdowns)
        
        # Returns
        if len(pnls) > 1:
            returns = np.diff(pnls)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            sharpe_annualized = sharpe * np.sqrt(252 * 6.5 * 3600)  # Assuming second resolution
        else:
            sharpe_annualized = 0.0
        
        # Inventory stats
        inv_stats = self.inventory_controller.get_inventory_stats()
        
        # Trade stats
        num_trades = len(self.trades)
        
        return {
            'final_pnl': final_pnl,
            'max_pnl': max_pnl,
            'min_pnl': min_pnl,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_annualized,
            'num_trades': num_trades,
            'inventory_stats': inv_stats
        }
