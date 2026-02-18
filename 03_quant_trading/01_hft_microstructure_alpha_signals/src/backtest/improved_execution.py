"""
Improved execution logic for HFT backtesting.

Implements execution improvements without model retraining:
1. Confidence-based trade filtering
2. Single-horizon execution
3. Passive limit orders
4. Volatility-conditioned execution
5. Position scaling by confidence
6. Comprehensive diagnostics

All thresholds are FIXED (not optimized on test data).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Position:
    """Current position state."""
    size: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class LimitOrder:
    """Pending limit order."""
    timestamp: int
    side: str  # 'buy' or 'sell'
    price: float
    size: int
    event_idx: int
    events_alive: int = 0


@dataclass
class Trade:
    """Executed trade record."""
    timestamp: int
    side: str
    price: float
    size: int
    cost: float
    confidence: float
    volatility: float


@dataclass
class ExecutionDiagnostics:
    """Diagnostics for execution analysis."""
    signals_generated: int = 0
    signals_filtered_confidence: int = 0
    signals_filtered_volatility: int = 0
    signals_executed: int = 0
    limit_orders_posted: int = 0
    limit_orders_filled: int = 0
    limit_orders_cancelled: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    holding_times: List[int] = field(default_factory=list)
    pnl_per_trade: List[float] = field(default_factory=list)


class ImprovedExecutor:
    """
    Improved execution engine with fixed thresholds.
    
    Key improvements:
    - Confidence filtering (fixed threshold)
    - Passive limit orders
    - Volatility regime filtering
    - Position scaling by confidence
    - Comprehensive diagnostics
    """
    
    # FIXED THRESHOLDS (not optimized on test data)
    CONFIDENCE_THRESHOLD = 0.20  # |P(up) - P(down)| must exceed this
    VOLATILITY_PERCENTILE = 50  # Only trade below median volatility
    MAX_ORDER_LIFETIME = 10  # Cancel limit orders after N events
    MAX_POSITION = 500  # Maximum inventory
    BASE_SIZE = 50  # Base order size
    
    def __init__(self, config: Dict):
        self.config = config
        self.position = Position()
        self.trades: List[Trade] = []
        self.pnl_history: List[Dict] = []
        self.diagnostics = ExecutionDiagnostics()
        self.pending_orders: List[LimitOrder] = []
        
        # Config parameters
        self.latency_events = config.get('latency_events', 1)
        
        # Use fixed thresholds
        self.confidence_threshold = self.CONFIDENCE_THRESHOLD
        self.max_order_lifetime = self.MAX_ORDER_LIFETIME
        self.max_position = self.MAX_POSITION
        self.base_size = self.BASE_SIZE
        
    def run(self, df: pd.DataFrame, predictions: pd.DataFrame, 
            features: pd.DataFrame) -> pd.DataFrame:
        """
        Run improved backtest.
        
        Args:
            df: LOB event data
            predictions: Model prediction probabilities
            features: Feature data (for volatility)
            
        Returns:
            DataFrame with backtest results
        """
        print(f"Running improved backtest on {len(df)} events...")
        
        # Reset state
        self.position = Position()
        self.trades = []
        self.pnl_history = []
        self.diagnostics = ExecutionDiagnostics()
        self.pending_orders = []
        
        # Compute volatility threshold (FIXED at median)
        volatility = features['volatility'].values
        self.volatility_threshold = np.nanpercentile(
            volatility[~np.isnan(volatility)], 
            self.VOLATILITY_PERCENTILE
        )
        
        print(f"Fixed volatility threshold: {self.volatility_threshold:.6f}")
        print(f"Fixed confidence threshold: {self.confidence_threshold:.2f}")
        
        # Process events sequentially
        for i in range(self.latency_events, len(df)):
            current_event = df.iloc[i]
            signal_idx = i - self.latency_events
            
            if signal_idx < 0 or signal_idx >= len(predictions):
                continue
            
            # Check pending limit orders
            self._check_limit_orders(current_event, i)
            
            # Process new signal
            signal = predictions.iloc[signal_idx]
            feature = features.iloc[signal_idx]
            self._process_signal(current_event, signal, feature, i)
            
            # Update PnL
            self._update_pnl(current_event, i)
        
        # Cancel remaining orders
        for order in self.pending_orders:
            self.diagnostics.limit_orders_cancelled += 1
        
        results_df = pd.DataFrame(self.pnl_history)
        
        self._print_diagnostics()
        
        return results_df
    
    def _compute_confidence(self, signal: pd.Series) -> float:
        """
        Compute confidence score from probabilities.
        
        Confidence = P(up) - P(down)
        Range: [-1, 1]
        """
        prob_up = signal.get('prob_up', signal[2] if len(signal) > 2 else 0)
        prob_down = signal.get('prob_down', signal[0] if len(signal) > 0 else 0)
        
        confidence = prob_up - prob_down
        return confidence
    
    def _should_trade(self, confidence: float, volatility: float) -> bool:
        """
        FIX 1 & 4: Filter trades by confidence and volatility.
        
        Returns True only if:
        - |confidence| > threshold (FIXED)
        - volatility < threshold (FIXED at median)
        """
        self.diagnostics.signals_generated += 1
        
        # Filter by confidence
        if abs(confidence) < self.confidence_threshold:
            self.diagnostics.signals_filtered_confidence += 1
            return False
        
        # Filter by volatility
        if volatility > self.volatility_threshold:
            self.diagnostics.signals_filtered_volatility += 1
            return False
        
        return True
    
    def _process_signal(self, event: pd.Series, signal: pd.Series, 
                       feature: pd.Series, event_idx: int):
        """
        Process trading signal with improved execution logic.
        """
        # Compute confidence (FIX 1)
        confidence = self._compute_confidence(signal)
        self.diagnostics.confidence_scores.append(confidence)
        
        # Get volatility (FIX 4)
        volatility = feature.get('volatility', 0.0)
        if np.isnan(volatility):
            volatility = 0.0
        
        # Filter trades (FIX 1 & 4)
        if not self._should_trade(confidence, volatility):
            return
        
        # Current prices
        bid_price = event['bid_price_1']
        ask_price = event['ask_price_1']
        
        # Entry logic (FIX 2: single direction based on confidence)
        if self.position.size == 0:
            if confidence > 0:
                # Bullish signal - post buy limit order (FIX 3)
                size = self._compute_order_size(confidence)
                self._post_limit_buy(event['timestamp'], bid_price, size, 
                                    event_idx, confidence, volatility)
                
            elif confidence < 0:
                # Bearish signal - post sell limit order (FIX 3)
                size = self._compute_order_size(abs(confidence))
                self._post_limit_sell(event['timestamp'], ask_price, size, 
                                     event_idx, confidence, volatility)
        
        # Exit logic (close position if confidence reverses)
        elif self.position.size > 0 and confidence < -self.confidence_threshold:
            # Exit long with limit order
            self._post_limit_sell(event['timestamp'], ask_price, 
                                 self.position.size, event_idx, confidence, volatility)
            
        elif self.position.size < 0 and confidence > self.confidence_threshold:
            # Exit short with limit order
            self._post_limit_buy(event['timestamp'], bid_price, 
                                -self.position.size, event_idx, confidence, volatility)
    
    def _compute_order_size(self, confidence: float) -> int:
        """
        FIX 5: Scale position size by confidence.
        
        Size = base_size * confidence
        Clipped at max_position.
        """
        size = int(self.base_size * (1 + confidence))
        size = min(size, self.max_position - abs(self.position.size))
        return max(size, 10)  # Minimum size
    
    def _post_limit_buy(self, timestamp: int, price: float, size: int, 
                       event_idx: int, confidence: float, volatility: float):
        """
        FIX 3: Post passive buy limit order at best bid.
        """
        if size <= 0:
            return
        
        order = LimitOrder(timestamp, 'buy', price, size, event_idx)
        self.pending_orders.append(order)
        self.diagnostics.limit_orders_posted += 1
    
    def _post_limit_sell(self, timestamp: int, price: float, size: int, 
                        event_idx: int, confidence: float, volatility: float):
        """
        FIX 3: Post passive sell limit order at best ask.
        """
        if size <= 0:
            return
        
        order = LimitOrder(timestamp, 'sell', price, size, event_idx)
        self.pending_orders.append(order)
        self.diagnostics.limit_orders_posted += 1
    
    def _check_limit_orders(self, event: pd.Series, event_idx: int):
        """
        FIX 3: Check if limit orders are filled or should be cancelled.
        
        Fill logic:
        - Buy order fills if market trades at or below our bid
        - Sell order fills if market trades at or above our ask
        
        Cancel after MAX_ORDER_LIFETIME events.
        """
        filled_orders = []
        
        for order in self.pending_orders:
            order.events_alive += 1
            
            # Check for fill
            filled = False
            
            if order.side == 'buy':
                # Buy fills if someone sells at our price or lower
                if event['ask_price_1'] <= order.price:
                    self._execute_fill(order, event['timestamp'])
                    filled = True
                    
            elif order.side == 'sell':
                # Sell fills if someone buys at our price or higher
                if event['bid_price_1'] >= order.price:
                    self._execute_fill(order, event['timestamp'])
                    filled = True
            
            # Cancel if too old (FIX 3)
            if order.events_alive > self.max_order_lifetime:
                self.diagnostics.limit_orders_cancelled += 1
                filled_orders.append(order)
            elif filled:
                filled_orders.append(order)
        
        # Remove filled/cancelled orders
        for order in filled_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
    
    def _execute_fill(self, order: LimitOrder, timestamp: int):
        """
        Execute limit order fill.
        
        No spread crossing cost (we provide liquidity).
        """
        self.diagnostics.limit_orders_filled += 1
        self.diagnostics.signals_executed += 1
        
        # Minimal transaction cost (maker rebate in reality)
        cost = order.size * order.price * 0.00001  # 0.1 bps
        
        # Track entry time for holding period
        entry_event = order.event_idx
        
        if order.side == 'buy':
            # Update position
            if self.position.size < 0:
                # Closing short
                pnl = -self.position.size * (order.price - self.position.avg_price)
                self.position.realized_pnl += pnl - cost
                self.diagnostics.pnl_per_trade.append(pnl - cost)
                self.position.size += order.size
                
                if self.position.size > 0:
                    self.position.avg_price = order.price
            else:
                # Opening/adding long
                total_cost = self.position.size * self.position.avg_price + order.size * order.price
                self.position.size += order.size
                self.position.avg_price = total_cost / self.position.size
                self.position.realized_pnl -= cost
        
        elif order.side == 'sell':
            # Update position
            if self.position.size > 0:
                # Closing long
                pnl = order.size * (order.price - self.position.avg_price)
                self.position.realized_pnl += pnl - cost
                self.diagnostics.pnl_per_trade.append(pnl - cost)
                self.position.size -= order.size
                
                if self.position.size < 0:
                    self.position.avg_price = order.price
            else:
                # Opening/adding short
                total_cost = -self.position.size * self.position.avg_price + order.size * order.price
                self.position.size -= order.size
                self.position.avg_price = total_cost / (-self.position.size) if self.position.size < 0 else 0
                self.position.realized_pnl -= cost
        
        # Record trade
        trade = Trade(timestamp, order.side, order.price, order.size, cost, 0.0, 0.0)
        self.trades.append(trade)
    
    def _update_pnl(self, event: pd.Series, event_idx: int):
        """Update PnL tracking."""
        mid_price = (event['bid_price_1'] + event['ask_price_1']) / 2
        
        if self.position.size != 0:
            self.position.unrealized_pnl = self.position.size * (mid_price - self.position.avg_price)
        else:
            self.position.unrealized_pnl = 0.0
        
        total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
        
        self.pnl_history.append({
            'timestamp': event['timestamp'],
            'event_idx': event_idx,
            'position': self.position.size,
            'mid_price': mid_price,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl,
            'total_pnl': total_pnl
        })
    
    def _print_diagnostics(self):
        """
        FIX 6: Print comprehensive execution diagnostics.
        """
        print("\n" + "="*60)
        print("EXECUTION DIAGNOSTICS")
        print("="*60)
        
        print(f"\nSignal Filtering:")
        print(f"  Total signals generated:        {self.diagnostics.signals_generated}")
        print(f"  Filtered by confidence:         {self.diagnostics.signals_filtered_confidence}")
        print(f"  Filtered by volatility:         {self.diagnostics.signals_filtered_volatility}")
        print(f"  Signals executed:               {self.diagnostics.signals_executed}")
        
        filter_rate = (self.diagnostics.signals_filtered_confidence + 
                      self.diagnostics.signals_filtered_volatility) / max(self.diagnostics.signals_generated, 1)
        print(f"  Filter rate:                    {filter_rate*100:.1f}%")
        
        print(f"\nLimit Order Execution:")
        print(f"  Orders posted:                  {self.diagnostics.limit_orders_posted}")
        print(f"  Orders filled:                  {self.diagnostics.limit_orders_filled}")
        print(f"  Orders cancelled:               {self.diagnostics.limit_orders_cancelled}")
        
        fill_rate = self.diagnostics.limit_orders_filled / max(self.diagnostics.limit_orders_posted, 1)
        print(f"  Fill rate:                      {fill_rate*100:.1f}%")
        
        print(f"\nConfidence Distribution:")
        if self.diagnostics.confidence_scores:
            conf_array = np.array(self.diagnostics.confidence_scores)
            print(f"  Mean:                           {conf_array.mean():.4f}")
            print(f"  Std:                            {conf_array.std():.4f}")
            print(f"  Min:                            {conf_array.min():.4f}")
            print(f"  Max:                            {conf_array.max():.4f}")
        
        print(f"\nPnL per Trade:")
        if self.diagnostics.pnl_per_trade:
            pnl_array = np.array(self.diagnostics.pnl_per_trade)
            print(f"  Mean:                           ${pnl_array.mean():.2f}")
            print(f"  Std:                            ${pnl_array.std():.2f}")
            print(f"  Win rate:                       {(pnl_array > 0).mean()*100:.1f}%")
        
        print(f"\nFinal Results:")
        print(f"  Total trades:                   {len(self.trades)}")
        print(f"  Realized PnL:                   ${self.position.realized_pnl:.2f}")
        print("="*60)
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostics as dictionary."""
        return {
            'signals_generated': self.diagnostics.signals_generated,
            'signals_filtered_confidence': self.diagnostics.signals_filtered_confidence,
            'signals_filtered_volatility': self.diagnostics.signals_filtered_volatility,
            'signals_executed': self.diagnostics.signals_executed,
            'limit_orders_posted': self.diagnostics.limit_orders_posted,
            'limit_orders_filled': self.diagnostics.limit_orders_filled,
            'limit_orders_cancelled': self.diagnostics.limit_orders_cancelled,
            'fill_rate': self.diagnostics.limit_orders_filled / max(self.diagnostics.limit_orders_posted, 1),
            'filter_rate': (self.diagnostics.signals_filtered_confidence + 
                          self.diagnostics.signals_filtered_volatility) / max(self.diagnostics.signals_generated, 1),
            'confidence_mean': np.mean(self.diagnostics.confidence_scores) if self.diagnostics.confidence_scores else 0,
            'pnl_per_trade_mean': np.mean(self.diagnostics.pnl_per_trade) if self.diagnostics.pnl_per_trade else 0,
            'win_rate': (np.array(self.diagnostics.pnl_per_trade) > 0).mean() if self.diagnostics.pnl_per_trade else 0,
        }
