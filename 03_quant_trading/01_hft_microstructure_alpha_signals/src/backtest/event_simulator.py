"""
Event-driven backtesting simulator for HFT strategies.

Processes LOB events sequentially with realistic latency and execution assumptions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class Position:
    """Current position state."""
    size: int = 0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    """Executed trade record."""
    timestamp: int
    side: str  # 'buy' or 'sell'
    price: float
    size: int
    cost: float


class EventSimulator:
    """
    Event-driven HFT backtest simulator.
    
    Key features:
    - Event-time processing (no clock-time resampling)
    - Latency modeling (minimum delay between signal and execution)
    - Realistic execution (spread crossing, queue dynamics)
    - Transaction costs
    - Position tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.position = Position()
        self.trades: List[Trade] = []
        self.pnl_history: List[Dict] = []
        
        # Config parameters
        self.latency_events = config.get('latency_events', 1)
        self.max_position = config.get('max_position', 1000)
        self.entry_threshold = config.get('entry_threshold', 0.6)
        self.exit_threshold = config.get('exit_threshold', 0.4)
        self.base_size = config.get('base_size', 100)
        
    def run(self, df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on LOB data with model predictions.
        
        Args:
            df: LOB event data
            predictions: DataFrame with prediction probabilities
            
        Returns:
            DataFrame with backtest results
        """
        print(f"Running backtest on {len(df)} events...")
        
        # Reset state
        self.position = Position()
        self.trades = []
        self.pnl_history = []
        
        # Process events sequentially
        for i in range(self.latency_events, len(df)):
            # Current market state (what we observe now)
            current_event = df.iloc[i]
            
            # Signal from latency_events ago (realistic delay)
            signal_idx = i - self.latency_events
            
            if signal_idx < 0 or signal_idx >= len(predictions):
                continue
            
            signal = predictions.iloc[signal_idx]
            
            # Make trading decision
            self._process_signal(current_event, signal, i)
            
            # Update PnL
            self._update_pnl(current_event, i)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.pnl_history)
        
        print(f"Backtest complete. Total trades: {len(self.trades)}")
        print(f"Final PnL: ${self.position.realized_pnl:.2f}")
        
        return results_df
    
    def _process_signal(self, event: pd.Series, signal: pd.Series, event_idx: int):
        """
        Process trading signal and execute if conditions met.
        
        Args:
            event: Current market state
            signal: Prediction signal (probabilities)
            event_idx: Current event index
        """
        # Extract probabilities (assume columns: prob_down, prob_flat, prob_up)
        if 'prob_up' in signal:
            prob_up = signal['prob_up']
            prob_down = signal['prob_down']
        else:
            # If not available, skip
            return
        
        # Current prices
        bid_price = event['bid_price_1']
        ask_price = event['ask_price_1']
        mid_price = (bid_price + ask_price) / 2
        
        # Entry logic
        if self.position.size == 0:
            # No position - look for entry
            if prob_up > self.entry_threshold:
                # Buy signal
                size = self._compute_order_size(event, prob_up)
                self._execute_buy(event['timestamp'], ask_price, size, event_idx)
                
            elif prob_down > self.entry_threshold:
                # Sell signal
                size = self._compute_order_size(event, prob_down)
                self._execute_sell(event['timestamp'], bid_price, size, event_idx)
        
        # Exit logic
        elif self.position.size > 0:
            # Long position - check exit
            if prob_down > self.exit_threshold or prob_up < 0.4:
                # Exit long
                self._execute_sell(event['timestamp'], bid_price, self.position.size, event_idx)
                
        elif self.position.size < 0:
            # Short position - check exit
            if prob_up > self.exit_threshold or prob_down < 0.4:
                # Exit short
                self._execute_buy(event['timestamp'], ask_price, -self.position.size, event_idx)
    
    def _compute_order_size(self, event: pd.Series, confidence: float) -> int:
        """
        Compute order size based on confidence and market conditions.
        
        Scale by:
        - Prediction confidence
        - Spread (reduce size when spread is wide)
        - Position limits
        """
        size = self.base_size
        
        # Scale by confidence
        if self.config.get('scale_by_confidence', True):
            size = int(size * confidence)
        
        # Scale by spread (reduce size when spread is wide)
        if self.config.get('scale_by_spread', True):
            spread = event['ask_price_1'] - event['bid_price_1']
            mid_price = (event['bid_price_1'] + event['ask_price_1']) / 2
            relative_spread = spread / mid_price
            
            # Reduce size if spread > 2 ticks
            if relative_spread > 0.0002:
                size = int(size * 0.5)
        
        # Respect position limits
        size = min(size, self.max_position - abs(self.position.size))
        
        return max(size, 0)
    
    def _execute_buy(self, timestamp: int, price: float, size: int, event_idx: int):
        """Execute buy order (cross the spread)."""
        if size <= 0:
            return
        
        # Transaction cost (half spread)
        cost = 0.0
        if self.config.get('half_spread_cost', True):
            # Assume we pay half spread as cost
            cost = size * price * 0.0001  # simplified
        
        # Update position
        if self.position.size < 0:
            # Closing short position
            pnl = -self.position.size * (price - self.position.avg_price)
            self.position.realized_pnl += pnl - cost
            self.position.size += size
            
            if self.position.size > 0:
                self.position.avg_price = price
        else:
            # Opening or adding to long
            total_cost = self.position.size * self.position.avg_price + size * price
            self.position.size += size
            self.position.avg_price = total_cost / self.position.size if self.position.size > 0 else 0
            self.position.realized_pnl -= cost
        
        # Record trade
        trade = Trade(timestamp, 'buy', price, size, cost)
        self.trades.append(trade)
    
    def _execute_sell(self, timestamp: int, price: float, size: int, event_idx: int):
        """Execute sell order (cross the spread)."""
        if size <= 0:
            return
        
        # Transaction cost
        cost = 0.0
        if self.config.get('half_spread_cost', True):
            cost = size * price * 0.0001
        
        # Update position
        if self.position.size > 0:
            # Closing long position
            pnl = size * (price - self.position.avg_price)
            self.position.realized_pnl += pnl - cost
            self.position.size -= size
            
            if self.position.size < 0:
                self.position.avg_price = price
        else:
            # Opening or adding to short
            total_cost = -self.position.size * self.position.avg_price + size * price
            self.position.size -= size
            self.position.avg_price = total_cost / (-self.position.size) if self.position.size < 0 else 0
            self.position.realized_pnl -= cost
        
        # Record trade
        trade = Trade(timestamp, 'sell', price, size, cost)
        self.trades.append(trade)
    
    def _update_pnl(self, event: pd.Series, event_idx: int):
        """Update unrealized PnL and record history."""
        mid_price = (event['bid_price_1'] + event['ask_price_1']) / 2
        
        # Unrealized PnL
        if self.position.size != 0:
            self.position.unrealized_pnl = self.position.size * (mid_price - self.position.avg_price)
        else:
            self.position.unrealized_pnl = 0.0
        
        # Total PnL
        total_pnl = self.position.realized_pnl + self.position.unrealized_pnl
        
        # Record
        self.pnl_history.append({
            'timestamp': event['timestamp'],
            'event_idx': event_idx,
            'position': self.position.size,
            'mid_price': mid_price,
            'realized_pnl': self.position.realized_pnl,
            'unrealized_pnl': self.position.unrealized_pnl,
            'total_pnl': total_pnl
        })


if __name__ == "__main__":
    # Test simulator
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    # Create dummy predictions
    predictions = pd.DataFrame({
        'prob_down': np.random.random(len(df)) * 0.3,
        'prob_flat': np.random.random(len(df)) * 0.4,
        'prob_up': np.random.random(len(df)) * 0.3,
    })
    
    # Normalize probabilities
    prob_sum = predictions.sum(axis=1)
    predictions = predictions.div(prob_sum, axis=0)
    
    # Run backtest
    config = {
        'latency_events': 1,
        'max_position': 1000,
        'entry_threshold': 0.6,
        'base_size': 100,
    }
    
    simulator = EventSimulator(config)
    results = simulator.run(df, predictions)
    
    print("\nBacktest results:")
    print(results.tail())
    print(f"\nFinal PnL: ${results['total_pnl'].iloc[-1]:.2f}")
