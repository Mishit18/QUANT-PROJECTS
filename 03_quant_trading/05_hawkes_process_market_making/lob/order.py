"""
Order types and data structures.
"""
from enum import IntEnum
from dataclasses import dataclass


class OrderType(IntEnum):
    """Order type enumeration matching Hawkes event types."""
    LIMIT_BUY = 0
    LIMIT_SELL = 1
    MARKET_BUY = 2
    MARKET_SELL = 3
    CANCEL_BUY = 4
    CANCEL_SELL = 5


class Side(IntEnum):
    """Order side."""
    BUY = 0
    SELL = 1


@dataclass
class Order:
    """Limit order representation."""
    order_id: int
    side: Side
    price: float
    size: int
    timestamp: float
    
    def __repr__(self):
        side_str = "BUY" if self.side == Side.BUY else "SELL"
        return f"Order({self.order_id}, {side_str}, {self.price:.2f}, {self.size}@{self.timestamp:.3f})"


@dataclass
class Trade:
    """Executed trade representation."""
    timestamp: float
    price: float
    size: int
    aggressor_side: Side
    
    def __repr__(self):
        side_str = "BUY" if self.aggressor_side == Side.BUY else "SELL"
        return f"Trade({self.price:.2f}, {self.size}, {side_str}@{self.timestamp:.3f})"
