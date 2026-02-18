"""
Hawkes-driven LOB simulation with market-making agent.
"""
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '.')

from hawkes.kernels import MultiDimensionalKernel
from hawkes.simulation import HawkesSimulator
from hawkes.estimation import HawkesMLEEstimator
from hawkes.diagnostics import HawkesDiagnostics
from backtest.engine import BacktestEngine


def load_config(config_path='configs/default.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_hawkes_simulation(config):
    """Simulate Hawkes process."""
    print("\n" + "="*60)
    print("STEP 1: HAWKES PROCESS SIMULATION")
    print("="*60)
    
    hawkes_config = config['hawkes']
    sim_config = config['simulation']
    
    # Create kernel
    kernel = MultiDimensionalKernel(
        alpha_matrix=np.array(hawkes_config['excitation']),
        beta_matrix=np.array(hawkes_config['decay'])
    )
    
    # Check stationarity
    is_stationary, spec_rad = kernel.check_stationarity()
    print(f"\nStationarity Check:")
    print(f"  Spectral Radius: {spec_rad:.4f}")
    print(f"  Is Stationary: {is_stationary}")
    
    if not is_stationary:
        print("  WARNING: Process is not stationary!")
    
    # Simulate
    baseline = np.array(hawkes_config['baseline'])
    simulator = HawkesSimulator(
        baseline=baseline,
        kernel=kernel,
        random_state=sim_config['random_seed']
    )
    
    print(f"\nSimulating Hawkes process...")
    print(f"  Duration: {sim_config['duration']}")
    print(f"  Target Events: {sim_config['num_events_target']}")
    
    events = simulator.simulate(
        T=sim_config['duration'],
        max_events=sim_config['num_events_target']
    )
    
    print(f"\nSimulation Complete:")
    print(f"  Total Events: {len(events)}")
    
    # Event type distribution
    event_types = [e[1] for e in events]
    for i in range(hawkes_config['num_types']):
        count = sum(1 for t in event_types if t == i)
        print(f"  Type {i}: {count} events ({100*count/len(events):.1f}%)")
    
    return events, kernel, baseline


def run_diagnostics(events, kernel, baseline):
    """Run Hawkes diagnostics."""
    print("\n" + "="*60)
    print("STEP 2: HAWKES DIAGNOSTICS")
    print("="*60)
    
    diagnostics = HawkesDiagnostics(baseline, kernel)
    
    # KS tests
    print("\nKolmogorov-Smirnov Tests:")
    ks_results = diagnostics.ks_test(events)
    
    for i, result in ks_results.items():
        if result is not None:
            print(f"  Type {i}: KS={result['statistic']:.4f}, p-value={result['p_value']:.4f}, n={result['n_samples']}")
        else:
            print(f"  Type {i}: No data")
    
    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    try:
        diagnostics.plot_all_diagnostics(events, save_path='diagnostics.png')
        print("  Saved: diagnostics.png")
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")
    
    return diagnostics


def run_mle_estimation(events, config):
    """Run MLE parameter estimation."""
    print("\n" + "="*60)
    print("STEP 3: MLE PARAMETER ESTIMATION")
    print("="*60)
    
    num_types = config['hawkes']['num_types']
    T = events[-1][0] if events else config['simulation']['duration']
    
    estimator = HawkesMLEEstimator(num_types)
    
    print("\nFitting Hawkes parameters via MLE...")
    print("  (This may take a minute...)")
    
    result = estimator.fit(events, T, max_iter=50)
    
    print(f"\nEstimation Complete:")
    print(f"  Success: {result['success']}")
    print(f"  Log-Likelihood: {result['log_likelihood']:.2f}")
    
    print(f"\nEstimated Baseline Intensities:")
    for i, mu in enumerate(result['baseline']):
        print(f"  Type {i}: {mu:.4f}")
    
    print(f"\nEstimated Excitation Matrix (alpha):")
    print(result['alpha'])
    
    return estimator


def run_backtest(events, config):
    """Run market-making backtest."""
    print("\n" + "="*60)
    print("STEP 4: MARKET-MAKING BACKTEST")
    print("="*60)
    
    engine = BacktestEngine(config)
    
    print("\nRunning backtest...")
    results = engine.run(events)
    
    engine.print_summary(results)
    
    return results


def plot_results(results):
    """Plot backtest results."""
    print("\nGenerating result plots...")
    
    agent = results['agent']
    
    if len(agent.pnl_history) == 0:
        print("  No PnL data to plot")
        return
    
    times, pnls, inventories = zip(*agent.pnl_history)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # PnL
    axes[0].plot(times, pnls, linewidth=1.5)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('PnL ($)')
    axes[0].set_title('Market Maker PnL')
    axes[0].grid(True, alpha=0.3)
    
    # Inventory
    axes[1].plot(times, inventories, linewidth=1.5, color='orange')
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Inventory')
    axes[1].set_title('Inventory Trajectory')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print("  Saved: backtest_results.png")
    
    plt.show()


def main():
    """Execute simulation pipeline."""
    print("\n" + "="*60)
    print("HAWKES PROCESS LOB SIMULATION")
    print("="*60)
    
    # Load config
    config = load_config()
    
    # Step 1: Simulate Hawkes process
    events, kernel, baseline = run_hawkes_simulation(config)
    
    # Step 2: Run diagnostics
    diagnostics = run_diagnostics(events, kernel, baseline)
    
    # Step 3: MLE estimation (optional, for demonstration)
    estimator = run_mle_estimation(events, config)
    
    # Step 4: Run backtest
    results = run_backtest(events, config)
    
    # Step 5: Plot results
    plot_results(results)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - diagnostics.png (Hawkes residual diagnostics)")
    print("  - backtest_results.png (PnL and inventory)")
    print("\n")


if __name__ == '__main__':
    main()
