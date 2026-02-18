"""
Baseline Experiment: Single-Agent Market Making with Monte Carlo

Demonstrates core Avellaneda-Stoikov model with statistical rigor:
- 500 Monte Carlo paths
- Mean ± confidence bands
- Distribution of PnL, Sharpe, inventory variance
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm

from src.models.avellaneda_stoikov import AvellanedaStoikov
from src.models.intensity_models import ExponentialIntensity
from src.market.market_environment import MarketEnvironment
from src.agents.market_maker import MarketMaker
from src.simulation.single_agent_simulator import SingleAgentSimulator


def run_baseline():
    """Run baseline single-agent experiment with Monte Carlo."""
    
    # Load config
    with open('config/parameters.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    n_mc = config['simulation']['n_monte_carlo']
    
    print("Running baseline Avellaneda-Stoikov with Monte Carlo...")
    print(f"  Monte Carlo paths: {n_mc}")
    print(f"  Risk aversion (γ): {config['market_maker']['risk_aversion']}")
    print(f"  Volatility (σ): {config['market']['volatility']}")
    print(f"  Arrival rate (A): {config['order_flow']['arrival_rate']}")
    
    # Run Monte Carlo
    mc_results = run_monte_carlo(config, n_mc)
    
    # Print statistics
    print(f"\nMonte Carlo Results (n={n_mc}):")
    print(f"  PnL:         ${mc_results['pnl_mean']:.2f} ± ${mc_results['pnl_std']:.2f}")
    print(f"  Sharpe:      {mc_results['sharpe_mean']:.3f} ± {mc_results['sharpe_std']:.3f}")
    print(f"  Inv Std:     {mc_results['inv_std_mean']:.2f} ± {mc_results['inv_std_std']:.2f}")
    print(f"  Total Fills: {mc_results['fills_mean']:.1f} ± {mc_results['fills_std']:.1f}")
    
    # Generate plots
    fig_dir = Path(config['output']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: PnL distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(mc_results['all_pnls'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(mc_results['pnl_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('Final PnL ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'PnL Distribution (μ={mc_results["pnl_mean"]:.2f}, σ={mc_results["pnl_std"]:.2f})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(mc_results['all_sharpes'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[0, 1].axvline(mc_results['sharpe_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('Sharpe Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Sharpe Distribution (μ={mc_results["sharpe_mean"]:.3f}, σ={mc_results["sharpe_std"]:.3f})')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(mc_results['all_inv_stds'], bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].axvline(mc_results['inv_std_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].set_xlabel('Inventory Std Dev')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Inventory Risk Distribution (μ={mc_results["inv_std_mean"]:.2f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(mc_results['all_fills'], bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[1, 1].axvline(mc_results['fills_mean'], color='red', linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].set_xlabel('Total Fills')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Fill Count Distribution (μ={mc_results["fills_mean"]:.1f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Baseline Market Making: Monte Carlo Results (n={n_mc})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'baseline_monte_carlo.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Mean trajectory with confidence bands
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    time_grid = mc_results['time_grid']
    pnl_mean = mc_results['pnl_trajectory_mean']
    pnl_std = mc_results['pnl_trajectory_std']
    inv_mean = mc_results['inv_trajectory_mean']
    inv_std = mc_results['inv_trajectory_std']
    
    ax1.plot(time_grid, pnl_mean, linewidth=2, color='darkblue', label='Mean')
    ax1.fill_between(time_grid, pnl_mean - pnl_std, pnl_mean + pnl_std, 
                      alpha=0.3, color='blue', label='±1 Std')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_ylabel('PnL ($)')
    ax1.set_title('PnL Evolution (Mean ± Std)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_grid, inv_mean, linewidth=2, color='darkgreen', label='Mean')
    ax2.fill_between(time_grid, inv_mean - inv_std, inv_mean + inv_std,
                      alpha=0.3, color='green', label='±1 Std')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Inventory')
    ax2.set_title('Inventory Evolution (Mean ± Std)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'baseline_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nFigures saved to {fig_dir}/")
    
    # Save table
    table_dir = Path(config['output']['tables_dir'])
    table_dir.mkdir(parents=True, exist_ok=True)
    
    with open(table_dir / 'baseline_statistics.txt', 'w') as f:
        f.write("BASELINE MARKET MAKING: MONTE CARLO STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Number of Monte Carlo paths: {n_mc}\n\n")
        f.write(f"{'Metric':<25} {'Mean':>15} {'Std Dev':>15} {'Min':>12} {'Max':>12}\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Final PnL ($)':<25} {mc_results['pnl_mean']:>15.2f} {mc_results['pnl_std']:>15.2f} {np.min(mc_results['all_pnls']):>12.2f} {np.max(mc_results['all_pnls']):>12.2f}\n")
        f.write(f"{'Sharpe Ratio':<25} {mc_results['sharpe_mean']:>15.3f} {mc_results['sharpe_std']:>15.3f} {np.min(mc_results['all_sharpes']):>12.3f} {np.max(mc_results['all_sharpes']):>12.3f}\n")
        f.write(f"{'Inventory Std':<25} {mc_results['inv_std_mean']:>15.2f} {mc_results['inv_std_std']:>15.2f} {np.min(mc_results['all_inv_stds']):>12.2f} {np.max(mc_results['all_inv_stds']):>12.2f}\n")
        f.write(f"{'Total Fills':<25} {mc_results['fills_mean']:>15.1f} {mc_results['fills_std']:>15.1f} {np.min(mc_results['all_fills']):>12.0f} {np.max(mc_results['all_fills']):>12.0f}\n")
    
    print(f"Table saved to {table_dir}/baseline_statistics.txt")
    
    return mc_results


def run_monte_carlo(config, n_paths):
    """Run Monte Carlo simulations."""
    
    all_pnls = []
    all_sharpes = []
    all_inv_stds = []
    all_fills = []
    all_pnl_trajectories = []
    all_inv_trajectories = []
    
    for i in tqdm(range(n_paths), desc="Monte Carlo"):
        # Different seed for each path
        result = run_single_simulation(config, random_seed=config['simulation']['random_seed'] + i)
        
        all_pnls.append(result['metrics']['final_pnl'])
        all_sharpes.append(result['metrics']['sharpe_ratio'])
        all_inv_stds.append(result['metrics']['inventory_std'])
        all_fills.append(result['metrics']['total_fills'])
        all_pnl_trajectories.append(result['history']['pnl'])
        all_inv_trajectories.append(result['history']['inventory'])
    
    # Convert to arrays
    pnl_trajectories = np.array(all_pnl_trajectories)
    inv_trajectories = np.array(all_inv_trajectories)
    
    return {
        'pnl_mean': np.mean(all_pnls),
        'pnl_std': np.std(all_pnls),
        'sharpe_mean': np.mean(all_sharpes),
        'sharpe_std': np.std(all_sharpes),
        'inv_std_mean': np.mean(all_inv_stds),
        'inv_std_std': np.std(all_inv_stds),
        'fills_mean': np.mean(all_fills),
        'fills_std': np.std(all_fills),
        'all_pnls': all_pnls,
        'all_sharpes': all_sharpes,
        'all_inv_stds': all_inv_stds,
        'all_fills': all_fills,
        'time_grid': np.linspace(0, config['simulation']['T'], len(pnl_trajectories[0])),
        'pnl_trajectory_mean': np.mean(pnl_trajectories, axis=0),
        'pnl_trajectory_std': np.std(pnl_trajectories, axis=0),
        'inv_trajectory_mean': np.mean(inv_trajectories, axis=0),
        'inv_trajectory_std': np.std(inv_trajectories, axis=0)
    }


def run_single_simulation(config, random_seed=42):
    """Run single simulation."""
    
    intensity_model = ExponentialIntensity(
        arrival_rate=config['order_flow']['arrival_rate'],
        decay_rate=config['order_flow']['intensity_decay']
    )
    
    as_model = AvellanedaStoikov(
        risk_aversion=config['market_maker']['risk_aversion'],
        volatility=config['market']['volatility'],
        terminal_time=config['simulation']['T'],
        intensity_decay=config['order_flow']['intensity_decay']
    )
    
    market_env = MarketEnvironment(
        initial_price=config['market']['initial_price'],
        volatility=config['market']['volatility'],
        dt=config['simulation']['dt'],
        intensity_model=intensity_model,
        adverse_selection_coef=config['adverse_selection']['impact_coefficient'],
        random_seed=random_seed
    )
    
    agent = MarketMaker(
        model=as_model,
        initial_inventory=config['market_maker']['initial_inventory'],
        initial_cash=config['market_maker']['initial_cash'],
        name="AS_MM"
    )
    
    simulator = SingleAgentSimulator(
        market_env=market_env,
        agent=agent,
        terminal_time=config['simulation']['T']
    )
    
    return simulator.run(verbose=False)


if __name__ == '__main__':
    run_baseline()
