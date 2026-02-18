"""
Backtesting engine for market-making strategies.
"""
import numpy as np
from lob.limit_order_book import LimitOrderBook
from lob.event_processor import EventProcessor
from agents.market_maker import MarketMaker
from backtest.metrics import generate_performance_report


class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        tick_size = config['lob']['tick_size']
        initial_mid = config['lob']['initial_mid_price']
        
        self.lob = LimitOrderBook(tick_size=tick_size)
        self.event_processor = EventProcessor(
            lob=self.lob,
            tick_size=tick_size,
            initial_mid_price=initial_mid
        )
        
        self.agent = MarketMaker(
            lob=self.lob,
            tick_size=tick_size,
            config=config['agent']
        )
        
        # Quote update frequency
        self.quote_update_interval = 0.1  # Update quotes every 0.1 time units
        self.last_quote_update = 0.0
    
    def run(self, events):
        """
        Run backtest on event stream.
        
        Parameters
        ----------
        events : list of tuples
            List of (time, event_type) from Hawkes simulation
        
        Returns
        -------
        results : dict
            Backtest results and metrics
        """
        # Initialize book
        self.event_processor.initialize_book(
            num_levels=self.config['lob']['num_price_levels'],
            volume_per_level=self.config['lob']['initial_depth']
        )
        
        # Initial quotes
        self.agent.update_quotes(0.0)
        
        # Process events
        for timestamp, event_type in events:
            # Process market event
            trades = self.event_processor.process_event(timestamp, event_type)
            
            # Check if agent was filled
            for trade in trades:
                # Check if trade involved agent's orders
                if self.agent.active_bid_id in self.lob.orders or self.agent.active_ask_id in self.lob.orders:
                    # Simplified: assume agent might be filled
                    pass
            
            # Update quotes periodically
            if timestamp - self.last_quote_update >= self.quote_update_interval:
                self.agent.update_quotes(timestamp)
                self.last_quote_update = timestamp
            
            # Mark to market
            if len(events) > 0 and (timestamp - events[0][0]) % 10 < 0.1:
                self.agent.mark_to_market(timestamp)
        
        # Final mark to market
        if len(events) > 0:
            self.agent.mark_to_market(events[-1][0])
        
        # Generate report
        total_time = events[-1][0] - events[0][0] if len(events) > 0 else 0.0
        report = generate_performance_report(self.agent, total_time)
        
        results = {
            'config': self.config,
            'num_events': len(events),
            'total_time': total_time,
            'performance': report,
            'agent': self.agent
        }
        
        return results
    
    def print_summary(self, results):
        """Print backtest summary."""
        print("\n" + "="*60)
        print("BACKTEST SUMMARY")
        print("="*60)
        
        print(f"\nSimulation:")
        print(f"  Total Events: {results['num_events']}")
        print(f"  Total Time: {results['total_time']:.2f}")
        
        perf = results['performance']
        
        if 'pnl' in perf:
            print(f"\nPnL:")
            print(f"  Final: ${perf['pnl']['final']:.2f}")
            print(f"  Max: ${perf['pnl']['max']:.2f}")
            print(f"  Min: ${perf['pnl']['min']:.2f}")
        
        if 'risk' in perf:
            print(f"\nRisk Metrics:")
            print(f"  Sharpe Ratio: {perf['risk']['sharpe_ratio']:.3f}")
            print(f"  Max Drawdown: ${perf['risk']['max_drawdown']:.2f} ({perf['risk']['max_drawdown_pct']:.2f}%)")
        
        if 'trading' in perf:
            print(f"\nTrading:")
            print(f"  Total Trades: {perf['trading']['num_trades']}")
            print(f"  Spread Capture: {perf['trading']['spread_capture_ticks']:.2f} ticks")
            print(f"  Trade Rate: {perf['trading']['trades_per_second']:.3f} trades/sec")
        
        if 'inventory' in perf:
            print(f"\nInventory:")
            print(f"  Final: {perf['inventory']['final']}")
            print(f"  Max: {perf['inventory']['max']}")
            print(f"  Min: {perf['inventory']['min']}")
            print(f"  Mean: {perf['inventory']['mean']:.2f}")
            print(f"  Turnover: {perf['inventory']['turnover']:.2f}")
        
        print("\n" + "="*60)
