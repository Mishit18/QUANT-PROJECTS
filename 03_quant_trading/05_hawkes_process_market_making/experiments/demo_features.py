"""
Demonstration of core components.
This script showcases individual components before running the full simulation.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from hawkes.kernels import MultiDimensionalKernel, ExponentialKernel
from hawkes.simulation import HawkesSimulator
from hawkes.estimation import HawkesMLEEstimator
from hawkes.diagnostics import HawkesDiagnostics
from lob.limit_order_book import LimitOrderBook
from lob.order import Side, OrderType
from lob.event_processor import EventProcessor
from agents.market_maker import MarketMaker
from agents.inventory_control import InventoryController


def demo_exponential_kernel():
    """Demonstrate exponential kernel."""
    print("\n" + "="*60)
    print("DEMO 1: Exponential Kernel")
    print("="*60)
    
    kernel = ExponentialKernel(alpha=0.5, beta=2.0)
    
    t = np.linspace(0, 5, 100)
    values = kernel(t)
    
    print(f"Kernel parameters: alpha={kernel.alpha}, beta={kernel.beta}")
    print(f"Kernel(0.5) = {kernel(0.5):.4f}")
    print(f"Integral(0 to 1) = {kernel.integral(1.0):.4f}")
    
    plt.figure(figsize=(8, 4))
    plt.plot(t, values, linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('φ(t)')
    plt.title('Exponential Kernel: φ(t) = α β exp(-β t)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('demo_kernel.png', dpi=150)
    print("Saved: demo_kernel.png")


def demo_hawkes_simulation():
    """Demonstrate Hawkes simulation."""
    print("\n" + "="*60)
    print("DEMO 2: Hawkes Process Simulation")
    print("="*60)
    
    # 2-type Hawkes process
    baseline = np.array([0.5, 0.5])
    alpha = np.array([[0.3, 0.1], [0.1, 0.3]])
    beta = np.array([[2.0, 2.0], [2.0, 2.0]])
    
    kernel = MultiDimensionalKernel(alpha, beta)
    is_stationary, spec_rad = kernel.check_stationarity()
    
    print(f"Baseline intensities: {baseline}")
    print(f"Excitation matrix:\n{alpha}")
    print(f"Spectral radius: {spec_rad:.4f}")
    print(f"Stationary: {is_stationary}")
    
    simulator = HawkesSimulator(baseline, kernel, random_state=42)
    events = simulator.simulate(T=100, max_events=500)
    
    print(f"\nSimulated {len(events)} events")
    
    # Separate by type
    times_0 = [t for t, typ in events if typ == 0]
    times_1 = [t for t, typ in events if typ == 1]
    
    print(f"Type 0: {len(times_0)} events")
    print(f"Type 1: {len(times_1)} events")
    
    # Plot event times
    plt.figure(figsize=(10, 4))
    plt.scatter(times_0, [0]*len(times_0), alpha=0.6, s=20, label='Type 0')
    plt.scatter(times_1, [1]*len(times_1), alpha=0.6, s=20, label='Type 1')
    plt.xlabel('Time')
    plt.ylabel('Event Type')
    plt.title('Hawkes Process Event Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('demo_hawkes_events.png', dpi=150)
    print("Saved: demo_hawkes_events.png")
    
    return events, kernel, baseline


def demo_mle_estimation(events, true_baseline, true_kernel):
    """Demonstrate MLE estimation."""
    print("\n" + "="*60)
    print("DEMO 3: MLE Parameter Estimation")
    print("="*60)
    
    T = events[-1][0]
    estimator = HawkesMLEEstimator(num_types=2)
    
    print("Fitting parameters via MLE...")
    result = estimator.fit(events, T, max_iter=30)
    
    print(f"\nEstimation result:")
    print(f"  Success: {result['success']}")
    print(f"  Log-likelihood: {result['log_likelihood']:.2f}")
    
    print(f"\nTrue vs Estimated Baseline:")
    for i in range(2):
        print(f"  Type {i}: True={true_baseline[i]:.3f}, Est={result['baseline'][i]:.3f}")
    
    print(f"\nTrue vs Estimated Excitation:")
    print("True alpha:")
    print(true_kernel.alpha)
    print("Estimated alpha:")
    print(result['alpha'])


def demo_diagnostics(events, kernel, baseline):
    """Demonstrate residual diagnostics."""
    print("\n" + "="*60)
    print("DEMO 4: Residual Diagnostics")
    print("="*60)
    
    diagnostics = HawkesDiagnostics(baseline, kernel)
    
    # Compute residuals
    residuals = diagnostics.compute_residuals(events)
    
    print("Residual statistics:")
    for i in range(2):
        if len(residuals[i]) > 0:
            print(f"  Type {i}: n={len(residuals[i])}, mean={np.mean(residuals[i]):.3f}, std={np.std(residuals[i]):.3f}")
    
    # KS tests
    ks_results = diagnostics.ks_test(events)
    print("\nKolmogorov-Smirnov Tests:")
    for i, result in ks_results.items():
        if result:
            print(f"  Type {i}: KS={result['statistic']:.4f}, p-value={result['p_value']:.4f}")


def demo_lob():
    """Demonstrate limit order book."""
    print("\n" + "="*60)
    print("DEMO 5: Limit Order Book")
    print("="*60)
    
    lob = LimitOrderBook(tick_size=0.01)
    
    # Add orders
    print("Adding limit orders...")
    lob.add_limit_order(Side.BUY, 99.95, 100, 0.0)
    lob.add_limit_order(Side.BUY, 99.96, 150, 0.0)
    lob.add_limit_order(Side.BUY, 99.97, 200, 0.0)
    
    lob.add_limit_order(Side.SELL, 100.03, 200, 0.0)
    lob.add_limit_order(Side.SELL, 100.04, 150, 0.0)
    lob.add_limit_order(Side.SELL, 100.05, 100, 0.0)
    
    print(f"\nBook state:")
    print(f"  Best Bid: {lob.best_bid():.2f}")
    print(f"  Best Ask: {lob.best_ask():.2f}")
    print(f"  Mid Price: {lob.mid_price():.2f}")
    print(f"  Spread: {lob.spread():.4f}")
    print(f"  Imbalance: {lob.imbalance():.3f}")
    
    # Show depth
    bids, asks = lob.get_depth(num_levels=3)
    print(f"\nDepth (top 3 levels):")
    print("  Bids:")
    for price, vol in bids:
        print(f"    {price:.2f}: {vol}")
    print("  Asks:")
    for price, vol in asks:
        print(f"    {price:.2f}: {vol}")
    
    # Execute market order
    print("\nExecuting market buy order (size=250)...")
    trades = lob.add_market_order(Side.BUY, 250, 1.0)
    
    print(f"  Executed {len(trades)} trades:")
    for trade in trades:
        print(f"    {trade}")
    
    print(f"\nNew book state:")
    print(f"  Best Bid: {lob.best_bid():.2f}")
    print(f"  Best Ask: {lob.best_ask():.2f}")
    print(f"  Mid Price: {lob.mid_price():.2f}")


def demo_market_maker():
    """Demonstrate market-making agent."""
    print("\n" + "="*60)
    print("DEMO 6: Market-Making Agent")
    print("="*60)
    
    # Setup
    lob = LimitOrderBook(tick_size=0.01)
    
    # Initialize book
    for i in range(1, 11):
        lob.add_limit_order(Side.BUY, 100.0 - i*0.01, 100, 0.0)
        lob.add_limit_order(Side.SELL, 100.0 + i*0.01, 100, 0.0)
    
    # Create agent
    config = {
        'inventory_limit': 50,
        'risk_aversion': 0.01,
        'target_spread_ticks': 2,
        'min_spread_ticks': 1,
        'max_spread_ticks': 5,
        'quote_size': 10,
        'latency_ms': 1.0,
        'adverse_selection_threshold': 0.7
    }
    
    agent = MarketMaker(lob, tick_size=0.01, config=config)
    
    print("Agent configuration:")
    print(f"  Inventory limit: ±{agent.inventory_limit}")
    print(f"  Risk aversion: {agent.risk_aversion}")
    print(f"  Target spread: {agent.target_spread_ticks} ticks")
    print(f"  Quote size: {agent.quote_size}")
    
    # Update quotes
    print("\nUpdating quotes...")
    agent.update_quotes(timestamp=0.0)
    
    print(f"  Active bid ID: {agent.active_bid_id}")
    print(f"  Active ask ID: {agent.active_ask_id}")
    
    if agent.active_bid_id and agent.active_bid_id in lob.orders:
        bid_order = lob.orders[agent.active_bid_id]
        print(f"  Bid: {bid_order.price:.2f} x {bid_order.size}")
    
    if agent.active_ask_id and agent.active_ask_id in lob.orders:
        ask_order = lob.orders[agent.active_ask_id]
        print(f"  Ask: {ask_order.price:.2f} x {ask_order.size}")
    
    # Simulate some fills
    print("\nSimulating fills...")
    agent.process_fill(agent.active_bid_id, 99.99, 10, 1.0)
    print(f"  Inventory after buy: {agent.inventory_controller.inventory}")
    
    agent.update_quotes(timestamp=2.0)
    agent.process_fill(agent.active_ask_id, 100.01, 10, 3.0)
    print(f"  Inventory after sell: {agent.inventory_controller.inventory}")
    
    # Mark to market
    agent.mark_to_market(4.0)
    
    summary = agent.get_performance_summary()
    print(f"\nPerformance summary:")
    print(f"  Final PnL: ${summary.get('final_pnl', 0):.2f}")
    print(f"  Num trades: {summary.get('num_trades', 0)}")


def main():
    """Run component demonstrations."""
    print("\n" + "="*60)
    print("COMPONENT DEMONSTRATIONS")
    print("="*60)
    
    # Demo 1: Kernel
    demo_exponential_kernel()
    
    # Demo 2: Hawkes simulation
    events, kernel, baseline = demo_hawkes_simulation()
    
    # Demo 3: MLE estimation
    demo_mle_estimation(events, baseline, kernel)
    
    # Demo 4: Diagnostics
    demo_diagnostics(events, kernel, baseline)
    
    # Demo 5: LOB
    demo_lob()
    
    # Demo 6: Market maker
    demo_market_maker()
    
    print("\n" + "="*60)
    print("DEMONSTRATIONS COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - demo_kernel.png")
    print("  - demo_hawkes_events.png")
    print("\nNext: Run 'python experiments/run_simulation.py' for complete simulation")
    print()


if __name__ == '__main__':
    main()
