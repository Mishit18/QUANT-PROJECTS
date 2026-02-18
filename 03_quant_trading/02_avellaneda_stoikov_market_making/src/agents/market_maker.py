"""
Market Maker Agent

Implements optimal market-making strategy using Avellaneda-Stoikov model.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from ..models.avellaneda_stoikov import AvellanedaStoikov


class MarketMaker:
    """
    Market maker agent using Avellaneda-Stoikov optimal quoting.
    
    The agent:
    1. Observes current mid-price, inventory, and time
    2. Computes optimal bid/ask quotes using AS model
    3. Tracks cash, inventory, and PnL
    4. Records all transactions for analysis
    """
    
    def __init__(
        self,
        model: AvellanedaStoikov,
        initial_inventory: int = 0,
        initial_cash: float = 0.0,
        name: str = "MM"
    ):
        """
        Initialize market maker agent.
        
        Args:
            model: Avellaneda-Stoikov model for optimal quoting
            initial_inventory: Starting inventory
            initial_cash: Starting cash
            name: Agent name (for identification)
        """
        self.model = model
        self.name = name
        
        # State variables
        self.inventory = initial_inventory
        self.cash = initial_cash
        self.initial_inventory = initial_inventory
        self.initial_cash = initial_cash
        
        # History tracking
        self.inventory_history: List[int] = [initial_inventory]
        self.cash_history: List[float] = [initial_cash]
        self.pnl_history: List[float] = [0.0]
        self.time_history: List[float] = [0.0]
        self.mid_price_history: List[float] = []
        self.bid_history: List[float] = []
        self.ask_history: List[float] = []
        self.bid_spread_history: List[float] = []
        self.ask_spread_history: List[float] = []
        
        # Transaction log
        self.transactions: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.n_bid_fills = 0
        self.n_ask_fills = 0
        self.total_spread_captured = 0.0
        self.adverse_selection_cost = 0.0
    
    def quote(self, mid_price: float, time: float) -> Tuple[float, float]:
        """
        Generate optimal bid/ask quotes.
        
        Args:
            mid_price: Current mid-price
            time: Current time
        
        Returns:
            (bid_price, ask_price)
        """
        bid, ask = self.model.optimal_quotes(mid_price, self.inventory, time)
        
        # Record quotes
        self.mid_price_history.append(mid_price)
        self.bid_history.append(bid)
        self.ask_history.append(ask)
        
        # Record spreads
        bid_spread = mid_price - bid
        ask_spread = ask - mid_price
        self.bid_spread_history.append(bid_spread)
        self.ask_spread_history.append(ask_spread)
        
        return bid, ask
    
    def process_fill(
        self,
        side: str,
        price: float,
        size: int,
        time: float,
        mid_price: float
    ):
        """
        Process an order fill.
        
        Args:
            side: 'buy' or 'sell' from market maker perspective
            price: Execution price
            size: Order size
            time: Execution time
            mid_price: Mid-price at execution
        """
        if side == 'buy':
            # Market maker buys (bid filled)
            self.inventory += size
            self.cash -= price * size
            self.n_bid_fills += 1
            
            # Spread captured (bought below mid)
            spread_captured = (mid_price - price) * size
            self.total_spread_captured += spread_captured
            
        elif side == 'sell':
            # Market maker sells (ask filled)
            self.inventory -= size
            self.cash += price * size
            self.n_ask_fills += 1
            
            # Spread captured (sold above mid)
            spread_captured = (price - mid_price) * size
            self.total_spread_captured += spread_captured
        
        else:
            raise ValueError(f"Invalid side: {side}")
        
        # Record transaction
        transaction = {
            'time': time,
            'side': side,
            'price': price,
            'size': size,
            'mid_price': mid_price,
            'inventory_after': self.inventory,
            'cash_after': self.cash
        }
        self.transactions.append(transaction)
    
    def update_state(self, time: float, mid_price: float):
        """
        Update agent state and history.
        
        Args:
            time: Current time
            mid_price: Current mid-price
        """
        self.time_history.append(time)
        self.inventory_history.append(self.inventory)
        self.cash_history.append(self.cash)
        
        # Calculate PnL (mark-to-market)
        mtm_value = self.cash + self.inventory * mid_price
        initial_value = self.initial_cash + self.initial_inventory * mid_price
        pnl = mtm_value - initial_value
        self.pnl_history.append(pnl)
    
    def get_pnl(self, mid_price: float) -> float:
        """
        Calculate current PnL (mark-to-market).
        
        Args:
            mid_price: Current mid-price for marking inventory
        
        Returns:
            Current PnL
        """
        current_value = self.cash + self.inventory * mid_price
        initial_value = self.initial_cash + self.initial_inventory * mid_price
        return current_value - initial_value
    
    def get_metrics(self, mid_price: float) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            mid_price: Current mid-price
        
        Returns:
            Dictionary of performance metrics
        """
        pnl = self.get_pnl(mid_price)
        
        # Calculate returns
        if len(self.pnl_history) > 1:
            returns = np.diff(self.pnl_history)
            sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
            max_dd = self._calculate_max_drawdown()
        else:
            sharpe = 0.0
            max_dd = 0.0
        
        metrics = {
            'name': self.name,
            'final_pnl': pnl,
            'final_inventory': self.inventory,
            'final_cash': self.cash,
            'n_bid_fills': self.n_bid_fills,
            'n_ask_fills': self.n_ask_fills,
            'total_fills': self.n_bid_fills + self.n_ask_fills,
            'spread_captured': self.total_spread_captured,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'inventory_std': np.std(self.inventory_history),
            'avg_inventory': np.mean(self.inventory_history)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from PnL history."""
        if len(self.pnl_history) < 2:
            return 0.0
        
        pnl_array = np.array(self.pnl_history)
        running_max = np.maximum.accumulate(pnl_array)
        drawdown = running_max - pnl_array
        return np.max(drawdown)
    
    def reset(self):
        """Reset agent to initial state."""
        self.inventory = self.initial_inventory
        self.cash = self.initial_cash
        
        self.inventory_history = [self.initial_inventory]
        self.cash_history = [self.initial_cash]
        self.pnl_history = [0.0]
        self.time_history = [0.0]
        self.mid_price_history = []
        self.bid_history = []
        self.ask_history = []
        self.bid_spread_history = []
        self.ask_spread_history = []
        
        self.transactions = []
        self.n_bid_fills = 0
        self.n_ask_fills = 0
        self.total_spread_captured = 0.0
        self.adverse_selection_cost = 0.0
    
    def get_history(self) -> Dict[str, np.ndarray]:
        """
        Get complete history as numpy arrays.
        
        Returns:
            Dictionary of historical data
        """
        return {
            'time': np.array(self.time_history),
            'inventory': np.array(self.inventory_history),
            'cash': np.array(self.cash_history),
            'pnl': np.array(self.pnl_history),
            'mid_price': np.array(self.mid_price_history),
            'bid': np.array(self.bid_history),
            'ask': np.array(self.ask_history),
            'bid_spread': np.array(self.bid_spread_history),
            'ask_spread': np.array(self.ask_spread_history)
        }
