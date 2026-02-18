"""
Limit order book implementation with price-time priority.
"""
import numpy as np
from collections import defaultdict, deque
from lob.order import Order, Trade, Side


class LimitOrderBook:
    """Event-driven limit order book with price-time priority."""
    
    def __init__(self, tick_size=0.01):
        """
        Parameters
        ----------
        tick_size : float
            Minimum price increment
        """
        self.tick_size = tick_size
        
        # Price levels: price -> deque of orders
        self.bids = defaultdict(deque)  # Buy orders
        self.asks = defaultdict(deque)  # Sell orders
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.next_order_id = 1
        
        # Trade history
        self.trades = []
        
        # Book state cache
        self._best_bid = None
        self._best_ask = None
        self._dirty = True
    
    def add_limit_order(self, side, price, size, timestamp):
        """
        Add limit order to book.
        
        Parameters
        ----------
        side : Side
            Order side
        price : float
            Limit price
        size : int
            Order size
        timestamp : float
            Order timestamp
        
        Returns
        -------
        order_id : int
            Assigned order ID
        trades : list of Trade
            Immediate executions (if any)
        """
        price = self._round_price(price)
        order_id = self.next_order_id
        self.next_order_id += 1
        
        order = Order(order_id, side, price, size, timestamp)
        
        # Try to match immediately
        trades = self._match_order(order, timestamp)
        
        # If not fully filled, add to book
        if order.size > 0:
            self.orders[order_id] = order
            if side == Side.BUY:
                self.bids[price].append(order)
            else:
                self.asks[price].append(order)
            self._dirty = True
        
        return order_id, trades
    
    def add_market_order(self, side, size, timestamp):
        """
        Execute market order (immediate or cancel).
        
        Parameters
        ----------
        side : Side
            Order side
        size : int
            Order size
        timestamp : float
            Order timestamp
        
        Returns
        -------
        trades : list of Trade
            Executed trades
        """
        trades = []
        remaining = size
        
        if side == Side.BUY:
            # Buy at best ask
            while remaining > 0 and len(self.asks) > 0:
                best_ask = self.best_ask()
                if best_ask is None:
                    break
                
                level = self.asks[best_ask]
                if len(level) == 0:
                    del self.asks[best_ask]
                    self._dirty = True
                    continue
                
                order = level[0]
                fill_size = min(remaining, order.size)
                
                trades.append(Trade(timestamp, order.price, fill_size, Side.BUY))
                
                order.size -= fill_size
                remaining -= fill_size
                
                if order.size == 0:
                    level.popleft()
                    del self.orders[order.order_id]
                    if len(level) == 0:
                        del self.asks[best_ask]
                
                self._dirty = True
        
        else:
            # Sell at best bid
            while remaining > 0 and len(self.bids) > 0:
                best_bid = self.best_bid()
                if best_bid is None:
                    break
                
                level = self.bids[best_bid]
                if len(level) == 0:
                    del self.bids[best_bid]
                    self._dirty = True
                    continue
                
                order = level[0]
                fill_size = min(remaining, order.size)
                
                trades.append(Trade(timestamp, order.price, fill_size, Side.SELL))
                
                order.size -= fill_size
                remaining -= fill_size
                
                if order.size == 0:
                    level.popleft()
                    del self.orders[order.order_id]
                    if len(level) == 0:
                        del self.bids[best_bid]
                
                self._dirty = True
        
        self.trades.extend(trades)
        return trades
    
    def cancel_order(self, order_id):
        """
        Cancel order by ID.
        
        Parameters
        ----------
        order_id : int
            Order ID to cancel
        
        Returns
        -------
        success : bool
            Whether cancellation succeeded
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.side == Side.BUY:
            level = self.bids[order.price]
        else:
            level = self.asks[order.price]
        
        try:
            level.remove(order)
            del self.orders[order_id]
            
            if len(level) == 0:
                if order.side == Side.BUY:
                    del self.bids[order.price]
                else:
                    del self.asks[order.price]
            
            self._dirty = True
            return True
        except ValueError:
            return False
    
    def _match_order(self, order, timestamp):
        """Match order against book, return trades."""
        trades = []
        
        if order.side == Side.BUY:
            # Match against asks
            while order.size > 0 and len(self.asks) > 0:
                best_ask = self.best_ask()
                if best_ask is None or order.price < best_ask:
                    break
                
                level = self.asks[best_ask]
                if len(level) == 0:
                    del self.asks[best_ask]
                    self._dirty = True
                    continue
                
                contra_order = level[0]
                fill_size = min(order.size, contra_order.size)
                
                trades.append(Trade(timestamp, contra_order.price, fill_size, Side.BUY))
                
                order.size -= fill_size
                contra_order.size -= fill_size
                
                if contra_order.size == 0:
                    level.popleft()
                    del self.orders[contra_order.order_id]
                    if len(level) == 0:
                        del self.asks[best_ask]
                
                self._dirty = True
        
        else:
            # Match against bids
            while order.size > 0 and len(self.bids) > 0:
                best_bid = self.best_bid()
                if best_bid is None or order.price > best_bid:
                    break
                
                level = self.bids[best_bid]
                if len(level) == 0:
                    del self.bids[best_bid]
                    self._dirty = True
                    continue
                
                contra_order = level[0]
                fill_size = min(order.size, contra_order.size)
                
                trades.append(Trade(timestamp, contra_order.price, fill_size, Side.SELL))
                
                order.size -= fill_size
                contra_order.size -= fill_size
                
                if contra_order.size == 0:
                    level.popleft()
                    del self.orders[contra_order.order_id]
                    if len(level) == 0:
                        del self.bids[best_bid]
                
                self._dirty = True
        
        self.trades.extend(trades)
        return trades
    
    def best_bid(self):
        """Get best bid price."""
        if self._dirty or self._best_bid is None:
            self._update_best_prices()
        return self._best_bid
    
    def best_ask(self):
        """Get best ask price."""
        if self._dirty or self._best_ask is None:
            self._update_best_prices()
        return self._best_ask
    
    def mid_price(self):
        """Get mid price."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        elif bid is not None:
            return bid
        elif ask is not None:
            return ask
        else:
            return None
    
    def spread(self):
        """Get bid-ask spread."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    def imbalance(self):
        """
        Compute order book imbalance at best levels.
        
        Returns
        -------
        imbalance : float
            (bid_volume - ask_volume) / (bid_volume + ask_volume)
        """
        bid = self.best_bid()
        ask = self.best_ask()
        
        if bid is None or ask is None:
            return 0.0
        
        bid_vol = sum(o.size for o in self.bids[bid])
        ask_vol = sum(o.size for o in self.asks[ask])
        
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        
        return (bid_vol - ask_vol) / total
    
    def _update_best_prices(self):
        """Update cached best prices."""
        self._best_bid = max(self.bids.keys()) if self.bids else None
        self._best_ask = min(self.asks.keys()) if self.asks else None
        self._dirty = False
    
    def _round_price(self, price):
        """Round price to tick size."""
        return round(price / self.tick_size) * self.tick_size
    
    def get_depth(self, num_levels=5):
        """
        Get order book depth.
        
        Returns
        -------
        bids : list of (price, volume)
        asks : list of (price, volume)
        """
        bid_prices = sorted(self.bids.keys(), reverse=True)[:num_levels]
        ask_prices = sorted(self.asks.keys())[:num_levels]
        
        bids = [(p, sum(o.size for o in self.bids[p])) for p in bid_prices]
        asks = [(p, sum(o.size for o in self.asks[p])) for p in ask_prices]
        
        return bids, asks
