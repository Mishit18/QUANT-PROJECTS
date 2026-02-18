"""
Expected Value (EV) based execution with queue awareness and PnL attribution.

Implements economic filtering without overfitting:
1. EV-based trade filtering (confidence × expected_move - transaction_cost)
2. Spread-conditioned execution
3. Queue-aware order management
4. Maker rebate modeling (FIXED)
5. Comprehensive PnL attribution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Position:
    """Current position state."""
    size: int = 0
    avg_price: float = 0.0
    entry_spread: float = 0.0
    entry_event: int = 0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class LimitOrder:
    """Pending limit order with entry conditions."""
    timestamp: int
    side: str
    price: float
    size: int
    event_idx: int
    events_alive: int = 0
    entry_spread: float = 0.0
    entry_ofi: float = 0.0
    entry_queue_imb: float = 0.0
    confidence: float = 0.0


@dataclass
class Trade:
    """Executed trade with attribution."""
    timestamp: int
    side: str
    price: float
    size: int
    confidence: float
    spread_at_entry: float
    spread_at_fill: float
    cost: float
    rebate: float
    holding_time: int = 0


@dataclass
class PnLAttribution:
    """Detailed PnL attribution."""
    directional_pnl: float = 0.0
    spread_capture: float = 0.0
    transaction_costs: float = 0.0
    maker_rebates: float = 0.0
    adverse_selection: float = 0.0
    
    # Tracking lists
    pnl_by_confidence: List[Tuple[float, float]] = field(default_factory=list)
    pnl_by_spread: List[Tuple[float, float]] = field(default_factory=list)
    filled_trades: List[Trade] = field(default_factory=list)
    cancelled_orders: List[LimitOrder] = field(default_factory=list)
    missed_opportunities: int = 0



@dataclass
class ExecutionDiagnostics:
    """Comprehensive execution diagnostics."""
    signals_generated: int = 0
    signals_filtered_ev: int = 0
    signals_filtered_spread: int = 0
    signals_executed: int = 0
    
    limit_orders_posted: int = 0
    limit_orders_filled: int = 0
    limit_orders_cancelled_ofi: int = 0
    limit_orders_cancelled_queue: int = 0
    limit_orders_cancelled_spread: int = 0
    limit_orders_cancelled_timeout: int = 0
    
    ev_scores: List[float] = field(default_factory=list)
    confidence_scores: List[float] = field(default_factory=list)
    spread_at_signal: List[float] = field(default_factory=list)
    holding_times: List[int] = field(default_factory=list)


class EVExecutor:
    """
    Expected Value based execution engine.
    
    Key features:
    - EV-based trade filtering (economic, not statistical)
    - Spread-conditioned execution
    - Queue-aware order cancellation
    - Fixed maker rebate modeling
    - Comprehensive PnL attribution
    """

    
    # FIXED PARAMETERS (not optimized)
    TICK_SIZE = 0.01
    EXPECTED_MOVE_TICKS = 1.5  # Historical average move for 5-tick horizon (more realistic)
    MAKER_REBATE_TICKS = 0.2  # Fixed maker rebate
    MAX_ORDER_LIFETIME = 10  # Cancel after N events
    MAX_POSITION = 500
    BASE_SIZE = 50
    
    def __init__(self, config: Dict):
        self.config = config
        self.position = Position()
        self.trades: List[Trade] = []
        self.pnl_history: List[Dict] = []
        self.diagnostics = ExecutionDiagnostics()
        self.attribution = PnLAttribution()
        self.pending_orders: List[LimitOrder] = []
        
        self.latency_events = config.get('latency_events', 1)
        
        # Compute spread statistics (FIXED at median)
        self.spread_threshold = None

    
    def run(self, df: pd.DataFrame, predictions: pd.DataFrame, 
            features: pd.DataFrame) -> pd.DataFrame:
        """Run EV-based backtest with full attribution."""
        print(f"Running EV-based backtest on {len(df)} events...")
        
        # Reset state
        self.position = Position()
        self.trades = []
        self.pnl_history = []
        self.diagnostics = ExecutionDiagnostics()
        self.attribution = PnLAttribution()
        self.pending_orders = []
        
        # Compute spread threshold (FIXED at median)
        spreads = (df['ask_price_1'] - df['bid_price_1']).values
        self.spread_threshold = np.nanmedian(spreads)
        
        print(f"Fixed spread threshold (median): {self.spread_threshold:.4f}")
        print(f"Fixed expected move: {self.EXPECTED_MOVE_TICKS} ticks")
        print(f"Fixed maker rebate: {self.MAKER_REBATE_TICKS} ticks")
        
        # Process events
        for i in range(self.latency_events, len(df)):
            if i % 5000 == 0:
                print(f"  Processing event {i}/{len(df)}...")
            
            current_event = df.iloc[i]
            signal_idx = i - self.latency_events
            
            if signal_idx < 0 or signal_idx >= len(predictions):
                continue
            
            # Check and manage pending orders
            self._manage_pending_orders(current_event, df.iloc[i], 
                                        features.iloc[i], i)
            
            # Process new signal
            signal = predictions.iloc[signal_idx]
            feature = features.iloc[signal_idx]
            self._process_signal(current_event, signal, feature, i)
            
            # Update PnL
            self._update_pnl(current_event, i)
        
        # Final cleanup
        for order in self.pending_orders:
            self.diagnostics.limit_orders_cancelled_timeout += 1
            self.attribution.cancelled_orders.append(order)
        
        results_df = pd.DataFrame(self.pnl_history)
        self._print_diagnostics()
        self._print_attribution()
        
        return results_df

    
    def _compute_ev(self, confidence: float, spread: float) -> float:
        """
        TASK 1: Compute Expected Value for trade.
        
        EV = confidence × expected_move - transaction_cost
        
        Where:
        - expected_move = historical average (FIXED)
        - transaction_cost = spread / 2 (for market orders)
        
        For limit orders, we get maker rebate instead of paying spread.
        """
        expected_move = confidence * self.EXPECTED_MOVE_TICKS * self.TICK_SIZE
        transaction_cost = spread / 2.0
        maker_rebate = self.MAKER_REBATE_TICKS * self.TICK_SIZE
        
        # Net EV with maker rebate
        ev = expected_move - transaction_cost + maker_rebate
        
        return ev

    
    def _should_enter_trade(self, confidence: float, spread: float, 
                           spread_history: List[float]) -> Tuple[bool, str]:
        """
        TASK 1 & 2: Combined EV and spread filtering.
        
        Returns (should_trade, reason)
        """
        self.diagnostics.signals_generated += 1
        
        # TASK 1: EV filter
        ev = self._compute_ev(confidence, spread)
        self.diagnostics.ev_scores.append(ev)
        
        if ev <= 0:
            self.diagnostics.signals_filtered_ev += 1
            return False, "negative_ev"
        
        # TASK 2: Spread conditioning
        # Only enter when spread is tight (below median)
        if spread > self.spread_threshold:
            self.diagnostics.signals_filtered_spread += 1
            return False, "wide_spread"
        
        return True, "passed"

    
    def _should_cancel_order(self, order: LimitOrder, current_event: pd.Series,
                            current_features: pd.Series) -> Tuple[bool, str]:
        """
        TASK 3: Queue-aware order cancellation.
        
        Cancel if:
        - OFI flips sign
        - Queue imbalance deteriorates
        - Spread widens
        - Timeout
        """
        # Timeout
        if order.events_alive > self.MAX_ORDER_LIFETIME:
            return True, "timeout"
        
        # Get current market state
        current_spread = current_event['ask_price_1'] - current_event['bid_price_1']
        
        # TASK 3a: Cancel if spread widens significantly
        if current_spread > order.entry_spread * 1.5:
            return True, "spread_widened"
        
        # TASK 3b: Cancel if OFI flips sign
        current_ofi = current_features.get('ofi_1', 0.0)
        if not np.isnan(current_ofi) and not np.isnan(order.entry_ofi):
            if np.sign(current_ofi) != np.sign(order.entry_ofi) and order.entry_ofi != 0:
                return True, "ofi_flip"
        
        # TASK 3c: Cancel if queue imbalance deteriorates
        current_qi = current_features.get('queue_imbalance_1', 0.0)
        if not np.isnan(current_qi) and not np.isnan(order.entry_queue_imb):
            if order.side == 'buy' and current_qi < order.entry_queue_imb - 0.2:
                return True, "queue_deteriorated"
            elif order.side == 'sell' and current_qi > order.entry_queue_imb + 0.2:
                return True, "queue_deteriorated"
        
        return False, ""

    
    def _manage_pending_orders(self, event: pd.Series, current_event: pd.Series,
                               current_features: pd.Series, event_idx: int):
        """Check pending orders for fills or cancellations."""
        filled_orders = []
        
        for order in self.pending_orders:
            order.events_alive += 1
            
            # Check for cancellation (TASK 3)
            should_cancel, reason = self._should_cancel_order(
                order, current_event, current_features
            )
            
            if should_cancel:
                if reason == "ofi_flip":
                    self.diagnostics.limit_orders_cancelled_ofi += 1
                elif reason == "queue_deteriorated":
                    self.diagnostics.limit_orders_cancelled_queue += 1
                elif reason == "spread_widened":
                    self.diagnostics.limit_orders_cancelled_spread += 1
                elif reason == "timeout":
                    self.diagnostics.limit_orders_cancelled_timeout += 1
                
                self.attribution.cancelled_orders.append(order)
                filled_orders.append(order)
                continue
            
            # Check for fill
            filled = False
            current_spread = event['ask_price_1'] - event['bid_price_1']
            
            if order.side == 'buy':
                if event['ask_price_1'] <= order.price:
                    self._execute_fill(order, event['timestamp'], current_spread)
                    filled = True
            elif order.side == 'sell':
                if event['bid_price_1'] >= order.price:
                    self._execute_fill(order, event['timestamp'], current_spread)
                    filled = True
            
            if filled:
                filled_orders.append(order)
        
        # Remove filled/cancelled orders
        for order in filled_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)

    
    def _process_signal(self, event: pd.Series, signal: pd.Series,
                       feature: pd.Series, event_idx: int):
        """Process trading signal with EV and spread filtering."""
        # Compute confidence
        prob_up = signal.get('prob_up', signal[2] if len(signal) > 2 else 0)
        prob_down = signal.get('prob_down', signal[0] if len(signal) > 0 else 0)
        confidence = prob_up - prob_down
        
        self.diagnostics.confidence_scores.append(confidence)
        
        # Get market state
        bid_price = event['bid_price_1']
        ask_price = event['ask_price_1']
        spread = ask_price - bid_price
        
        self.diagnostics.spread_at_signal.append(spread)
        
        # TASK 1 & 2: EV and spread filtering
        should_trade, reason = self._should_enter_trade(
            confidence, spread, []
        )
        
        if not should_trade:
            if reason == "negative_ev":
                self.attribution.missed_opportunities += 1
            return
        
        # Get microstructure features for order management
        ofi = feature.get('ofi_1', 0.0)
        queue_imb = feature.get('queue_imbalance_1', 0.0)
        
        # Entry logic
        if self.position.size == 0:
            if confidence > 0:
                size = self._compute_order_size(confidence)
                self._post_limit_buy(event['timestamp'], bid_price, size,
                                    event_idx, confidence, spread, ofi, queue_imb)
            elif confidence < 0:
                size = self._compute_order_size(abs(confidence))
                self._post_limit_sell(event['timestamp'], ask_price, size,
                                     event_idx, confidence, spread, ofi, queue_imb)
        
        # Exit logic (TASK 2: exit if spread widens)
        elif self.position.size > 0:
            if confidence < 0 or spread > self.position.entry_spread * 1.5:
                self._post_limit_sell(event['timestamp'], ask_price,
                                     self.position.size, event_idx, confidence,
                                     spread, ofi, queue_imb)
        
        elif self.position.size < 0:
            if confidence > 0 or spread > self.position.entry_spread * 1.5:
                self._post_limit_buy(event['timestamp'], bid_price,
                                    -self.position.size, event_idx, confidence,
                                    spread, ofi, queue_imb)

    
    def _compute_order_size(self, confidence: float) -> int:
        """Compute order size based on confidence."""
        size = int(self.BASE_SIZE * (1 + abs(confidence)))
        size = min(size, self.MAX_POSITION - abs(self.position.size))
        return max(size, 10)
    
    def _post_limit_buy(self, timestamp: int, price: float, size: int,
                       event_idx: int, confidence: float, spread: float,
                       ofi: float, queue_imb: float):
        """Post passive buy limit order."""
        if size <= 0:
            return
        
        order = LimitOrder(
            timestamp=timestamp,
            side='buy',
            price=price,
            size=size,
            event_idx=event_idx,
            entry_spread=spread,
            entry_ofi=ofi,
            entry_queue_imb=queue_imb,
            confidence=confidence
        )
        self.pending_orders.append(order)
        self.diagnostics.limit_orders_posted += 1
    
    def _post_limit_sell(self, timestamp: int, price: float, size: int,
                        event_idx: int, confidence: float, spread: float,
                        ofi: float, queue_imb: float):
        """Post passive sell limit order."""
        if size <= 0:
            return
        
        order = LimitOrder(
            timestamp=timestamp,
            side='sell',
            price=price,
            size=size,
            event_idx=event_idx,
            entry_spread=spread,
            entry_ofi=ofi,
            entry_queue_imb=queue_imb,
            confidence=confidence
        )
        self.pending_orders.append(order)
        self.diagnostics.limit_orders_posted += 1

    
    def _execute_fill(self, order: LimitOrder, timestamp: int, fill_spread: float):
        """
        Execute limit order fill with TASK 4: maker rebate.
        """
        self.diagnostics.limit_orders_filled += 1
        self.diagnostics.signals_executed += 1
        
        # TASK 4: Apply fixed maker rebate
        rebate = order.size * self.MAKER_REBATE_TICKS * self.TICK_SIZE
        
        # Minimal transaction cost (exchange fees)
        cost = order.size * order.price * 0.00001  # 0.1 bps
        
        # Net cost after rebate
        net_cost = cost - rebate
        
        # Track holding time
        holding_time = 0  # Will be computed on exit
        
        if order.side == 'buy':
            # Update position
            if self.position.size < 0:
                # Closing short - compute PnL
                pnl = -self.position.size * (order.price - self.position.avg_price)
                holding_time = order.event_idx - self.position.entry_event
                
                # TASK 5: PnL attribution
                self._attribute_pnl(pnl, net_cost, rebate, order, fill_spread)
                
                self.position.realized_pnl += pnl - net_cost
                self.position.size += order.size
                
                if self.position.size > 0:
                    self.position.avg_price = order.price
                    self.position.entry_spread = order.entry_spread
                    self.position.entry_event = order.event_idx
            else:
                # Opening/adding long
                total_cost = self.position.size * self.position.avg_price + order.size * order.price
                self.position.size += order.size
                self.position.avg_price = total_cost / self.position.size
                self.position.entry_spread = order.entry_spread
                self.position.entry_event = order.event_idx
                self.position.realized_pnl -= net_cost
        
        elif order.side == 'sell':
            # Update position
            if self.position.size > 0:
                # Closing long - compute PnL
                pnl = order.size * (order.price - self.position.avg_price)
                holding_time = order.event_idx - self.position.entry_event
                
                # TASK 5: PnL attribution
                self._attribute_pnl(pnl, net_cost, rebate, order, fill_spread)
                
                self.position.realized_pnl += pnl - net_cost
                self.position.size -= order.size
                
                if self.position.size < 0:
                    self.position.avg_price = order.price
                    self.position.entry_spread = order.entry_spread
                    self.position.entry_event = order.event_idx
            else:
                # Opening/adding short
                total_cost = -self.position.size * self.position.avg_price + order.size * order.price
                self.position.size -= order.size
                self.position.avg_price = total_cost / (-self.position.size) if self.position.size < 0 else 0
                self.position.entry_spread = order.entry_spread
                self.position.entry_event = order.event_idx
                self.position.realized_pnl -= net_cost
        
        # Record trade
        trade = Trade(
            timestamp=timestamp,
            side=order.side,
            price=order.price,
            size=order.size,
            confidence=order.confidence,
            spread_at_entry=order.entry_spread,
            spread_at_fill=fill_spread,
            cost=cost,
            rebate=rebate,
            holding_time=holding_time
        )
        self.trades.append(trade)
        self.attribution.filled_trades.append(trade)
        
        if holding_time > 0:
            self.diagnostics.holding_times.append(holding_time)

    
    def _attribute_pnl(self, pnl: float, cost: float, rebate: float,
                      order: LimitOrder, fill_spread: float):
        """
        TASK 5: Detailed PnL attribution.
        """
        # Directional PnL (before costs)
        self.attribution.directional_pnl += pnl
        
        # Transaction costs
        self.attribution.transaction_costs += cost
        
        # Maker rebates
        self.attribution.maker_rebates += rebate
        
        # Spread capture (if we captured spread improvement)
        spread_improvement = order.entry_spread - fill_spread
        if spread_improvement > 0:
            self.attribution.spread_capture += spread_improvement * order.size
        
        # Adverse selection (if spread widened)
        if spread_improvement < 0:
            self.attribution.adverse_selection += abs(spread_improvement) * order.size
        
        # Track by confidence bucket
        self.attribution.pnl_by_confidence.append((order.confidence, pnl - cost + rebate))
        
        # Track by spread regime
        self.attribution.pnl_by_spread.append((order.entry_spread, pnl - cost + rebate))
    
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
        """Print comprehensive execution diagnostics."""
        print("\n" + "="*70)
        print("EV-BASED EXECUTION DIAGNOSTICS")
        print("="*70)
        
        print(f"\nSignal Filtering:")
        print(f"  Total signals generated:        {self.diagnostics.signals_generated}")
        print(f"  Filtered by EV:                 {self.diagnostics.signals_filtered_ev}")
        print(f"  Filtered by spread:             {self.diagnostics.signals_filtered_spread}")
        print(f"  Signals executed:               {self.diagnostics.signals_executed}")
        
        total_filtered = (self.diagnostics.signals_filtered_ev + 
                         self.diagnostics.signals_filtered_spread)
        filter_rate = total_filtered / max(self.diagnostics.signals_generated, 1)
        print(f"  Filter rate:                    {filter_rate*100:.1f}%")
        
        print(f"\nLimit Order Management:")
        print(f"  Orders posted:                  {self.diagnostics.limit_orders_posted}")
        print(f"  Orders filled:                  {self.diagnostics.limit_orders_filled}")
        print(f"  Cancelled (OFI flip):           {self.diagnostics.limit_orders_cancelled_ofi}")
        print(f"  Cancelled (queue):              {self.diagnostics.limit_orders_cancelled_queue}")
        print(f"  Cancelled (spread):             {self.diagnostics.limit_orders_cancelled_spread}")
        print(f"  Cancelled (timeout):            {self.diagnostics.limit_orders_cancelled_timeout}")
        
        fill_rate = self.diagnostics.limit_orders_filled / max(self.diagnostics.limit_orders_posted, 1)
        print(f"  Fill rate:                      {fill_rate*100:.1f}%")
        
        print(f"\nEV Distribution:")
        if self.diagnostics.ev_scores:
            ev_array = np.array(self.diagnostics.ev_scores)
            print(f"  Mean EV:                        ${ev_array.mean():.6f}")
            print(f"  Positive EV signals:            {(ev_array > 0).sum()} ({(ev_array > 0).mean()*100:.1f}%)")
            print(f"  Negative EV signals:            {(ev_array <= 0).sum()} ({(ev_array <= 0).mean()*100:.1f}%)")
        
        print(f"\nHolding Times:")
        if self.diagnostics.holding_times:
            ht_array = np.array(self.diagnostics.holding_times)
            print(f"  Mean:                           {ht_array.mean():.1f} events")
            print(f"  Median:                         {np.median(ht_array):.1f} events")
            print(f"  Min/Max:                        {ht_array.min()}/{ht_array.max()} events")
        
        print(f"\nFinal Results:")
        print(f"  Total trades:                   {len(self.trades)}")
        print(f"  Realized PnL:                   ${self.position.realized_pnl:.2f}")
        print(f"  Missed opportunities:           {self.attribution.missed_opportunities}")
        print("="*70)

    
    def _print_attribution(self):
        """TASK 5: Print detailed PnL attribution."""
        print("\n" + "="*70)
        print("PNL ATTRIBUTION ANALYSIS")
        print("="*70)
        
        print(f"\nPnL Components:")
        print(f"  Directional PnL:                ${self.attribution.directional_pnl:.2f}")
        print(f"  Spread capture:                 ${self.attribution.spread_capture:.2f}")
        print(f"  Transaction costs:              -${self.attribution.transaction_costs:.2f}")
        print(f"  Maker rebates:                  +${self.attribution.maker_rebates:.2f}")
        print(f"  Adverse selection:              -${self.attribution.adverse_selection:.2f}")
        
        net_pnl = (self.attribution.directional_pnl + 
                  self.attribution.spread_capture -
                  self.attribution.transaction_costs +
                  self.attribution.maker_rebates -
                  self.attribution.adverse_selection)
        print(f"  Net PnL:                        ${net_pnl:.2f}")
        
        print(f"\nTrade Outcomes:")
        print(f"  Filled trades:                  {len(self.attribution.filled_trades)}")
        print(f"  Cancelled orders:               {len(self.attribution.cancelled_orders)}")
        print(f"  Missed opportunities:           {self.attribution.missed_opportunities}")
        
        if self.attribution.filled_trades:
            pnls = [pnl for _, pnl in self.attribution.pnl_by_confidence]
            if len(pnls) > 0:
                win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                print(f"  Win rate:                       {win_rate*100:.1f}%")
                print(f"  Avg PnL per trade:              ${np.mean(pnls):.2f}")
        
        print("="*70)
    
    def get_diagnostics(self) -> Dict:
        """Return diagnostics as dictionary."""
        return {
            'signals_generated': self.diagnostics.signals_generated,
            'signals_filtered_ev': self.diagnostics.signals_filtered_ev,
            'signals_filtered_spread': self.diagnostics.signals_filtered_spread,
            'signals_executed': self.diagnostics.signals_executed,
            'limit_orders_posted': self.diagnostics.limit_orders_posted,
            'limit_orders_filled': self.diagnostics.limit_orders_filled,
            'limit_orders_cancelled_ofi': self.diagnostics.limit_orders_cancelled_ofi,
            'limit_orders_cancelled_queue': self.diagnostics.limit_orders_cancelled_queue,
            'limit_orders_cancelled_spread': self.diagnostics.limit_orders_cancelled_spread,
            'limit_orders_cancelled_timeout': self.diagnostics.limit_orders_cancelled_timeout,
            'fill_rate': self.diagnostics.limit_orders_filled / max(self.diagnostics.limit_orders_posted, 1),
            'filter_rate': (self.diagnostics.signals_filtered_ev + self.diagnostics.signals_filtered_spread) / max(self.diagnostics.signals_generated, 1),
            'mean_ev': np.mean(self.diagnostics.ev_scores) if self.diagnostics.ev_scores else 0,
            'positive_ev_rate': (np.array(self.diagnostics.ev_scores) > 0).mean() if self.diagnostics.ev_scores else 0,
        }
    
    def get_attribution(self) -> Dict:
        """Return PnL attribution as dictionary."""
        return {
            'directional_pnl': self.attribution.directional_pnl,
            'spread_capture': self.attribution.spread_capture,
            'transaction_costs': self.attribution.transaction_costs,
            'maker_rebates': self.attribution.maker_rebates,
            'adverse_selection': self.attribution.adverse_selection,
            'filled_trades': len(self.attribution.filled_trades),
            'cancelled_orders': len(self.attribution.cancelled_orders),
            'missed_opportunities': self.attribution.missed_opportunities,
            'pnl_by_confidence': self.attribution.pnl_by_confidence,
            'pnl_by_spread': self.attribution.pnl_by_spread,
        }
