"""
Event processor that converts Hawkes events to LOB operations.
"""
import numpy as np
from lob.order import OrderType, Side
from lob.limit_order_book import LimitOrderBook


class EventProcessor:
    """Process Hawkes events and update limit order book."""
    
    def __init__(self, lob, tick_size, initial_mid_price, order_size_dist='constant'):
        """
        Parameters
        ----------
        lob : LimitOrderBook
            Order book instance
        tick_size : float
            Price tick size
        initial_mid_price : float
            Starting mid price
        order_size_dist : str
            Order size distribution ('constant', 'uniform', 'exponential')
        """
        self.lob = lob
        self.tick_size = tick_size
        self.mid_price = initial_mid_price
        self.order_size_dist = order_size_dist
        
        # Track active orders for cancellation
        self.active_buy_orders = []
        self.active_sell_orders = []
        
        self.rng = np.random.RandomState(42)
    
    def process_event(self, timestamp, event_type):
        """
        Process a Hawkes event and update LOB.
        
        Parameters
        ----------
        timestamp : float
            Event time
        event_type : int
            OrderType enum value
        
        Returns
        -------
        trades : list of Trade
            Executed trades (if any)
        """
        trades = []
        
        if event_type == OrderType.LIMIT_BUY:
            trades = self._process_limit_buy(timestamp)
        elif event_type == OrderType.LIMIT_SELL:
            trades = self._process_limit_sell(timestamp)
        elif event_type == OrderType.MARKET_BUY:
            trades = self._process_market_buy(timestamp)
        elif event_type == OrderType.MARKET_SELL:
            trades = self._process_market_sell(timestamp)
        elif event_type == OrderType.CANCEL_BUY:
            self._process_cancel_buy()
        elif event_type == OrderType.CANCEL_SELL:
            self._process_cancel_sell()
        
        # Update mid price estimate
        mid = self.lob.mid_price()
        if mid is not None:
            self.mid_price = mid
        
        return trades
    
    def _process_limit_buy(self, timestamp):
        """Add limit buy order."""
        price = self._sample_limit_price(Side.BUY)
        size = self._sample_order_size()
        
        order_id, trades = self.lob.add_limit_order(Side.BUY, price, size, timestamp)
        
        if order_id in self.lob.orders:
            self.active_buy_orders.append(order_id)
        
        return trades
    
    def _process_limit_sell(self, timestamp):
        """Add limit sell order."""
        price = self._sample_limit_price(Side.SELL)
        size = self._sample_order_size()
        
        order_id, trades = self.lob.add_limit_order(Side.SELL, price, size, timestamp)
        
        if order_id in self.lob.orders:
            self.active_sell_orders.append(order_id)
        
        return trades
    
    def _process_market_buy(self, timestamp):
        """Execute market buy order."""
        size = self._sample_order_size()
        return self.lob.add_market_order(Side.BUY, size, timestamp)
    
    def _process_market_sell(self, timestamp):
        """Execute market sell order."""
        size = self._sample_order_size()
        return self.lob.add_market_order(Side.SELL, size, timestamp)
    
    def _process_cancel_buy(self):
        """Cancel random buy order."""
        if len(self.active_buy_orders) > 0:
            idx = self.rng.randint(len(self.active_buy_orders))
            order_id = self.active_buy_orders[idx]
            
            if self.lob.cancel_order(order_id):
                self.active_buy_orders.pop(idx)
    
    def _process_cancel_sell(self):
        """Cancel random sell order."""
        if len(self.active_sell_orders) > 0:
            idx = self.rng.randint(len(self.active_sell_orders))
            order_id = self.active_sell_orders[idx]
            
            if self.lob.cancel_order(order_id):
                self.active_sell_orders.pop(idx)
    
    def _sample_limit_price(self, side):
        """Sample limit order price relative to mid."""
        # Place orders within a few ticks of mid
        offset_ticks = self.rng.geometric(0.3)  # Geometric distribution
        offset = offset_ticks * self.tick_size
        
        if side == Side.BUY:
            # Buy orders below mid
            return self.mid_price - offset
        else:
            # Sell orders above mid
            return self.mid_price + offset
    
    def _sample_order_size(self):
        """Sample order size."""
        if self.order_size_dist == 'constant':
            return 10
        elif self.order_size_dist == 'uniform':
            return self.rng.randint(5, 20)
        elif self.order_size_dist == 'exponential':
            return max(1, int(self.rng.exponential(10)))
        else:
            return 10
    
    def initialize_book(self, num_levels=10, volume_per_level=100):
        """
        Initialize book with symmetric depth.
        
        Parameters
        ----------
        num_levels : int
            Number of price levels on each side
        volume_per_level : int
            Volume at each level
        """
        for i in range(1, num_levels + 1):
            bid_price = self.mid_price - i * self.tick_size
            ask_price = self.mid_price + i * self.tick_size
            
            bid_id, _ = self.lob.add_limit_order(Side.BUY, bid_price, volume_per_level, 0.0)
            ask_id, _ = self.lob.add_limit_order(Side.SELL, ask_price, volume_per_level, 0.0)
            
            self.active_buy_orders.append(bid_id)
            self.active_sell_orders.append(ask_id)
