"""Visualization tools for execution analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def plot_cost_distributions(results: Dict[str, List[float]], save_path: str = None):
    """Plot cost distributions for different strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Histogram
    for strategy_name, costs in results.items():
        axes[0].hist(costs, alpha=0.6, bins=30, label=strategy_name, density=True)
    
    axes[0].set_xlabel('Execution Cost')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Cost Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data = [costs for costs in results.values()]
    labels = list(results.keys())
    
    axes[1].boxplot(data, labels=labels)
    axes[1].set_ylabel('Execution Cost')
    axes[1].set_title('Cost Distribution (Box Plot)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_execution_trajectory(inventory_trajectory: np.ndarray,
                              price_trajectory: np.ndarray,
                              strategy_name: str = "Strategy",
                              save_path: str = None):
    """Plot inventory and price evolution during execution."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    time_steps = np.arange(len(inventory_trajectory))
    
    # Inventory trajectory
    axes[0].plot(time_steps, inventory_trajectory, marker='o', linewidth=2)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Remaining Inventory')
    axes[0].set_title(f'{strategy_name}: Inventory Trajectory')
    axes[0].grid(True, alpha=0.3)
    
    # Price trajectory
    axes[1].plot(time_steps, price_trajectory, marker='o', linewidth=2, color='red')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Mid-Price')
    axes[1].set_title(f'{strategy_name}: Price Evolution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_strategy_comparison(results: Dict[str, Dict[str, float]], save_path: str = None):
    """Bar chart comparing strategy performance."""
    strategies = list(results.keys())
    mean_costs = [results[s]['mean_cost'] for s in strategies]
    std_costs = [results[s]['std_cost'] for s in strategies]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Mean cost
    axes[0].bar(strategies, mean_costs, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Mean Execution Cost')
    axes[0].set_title('Mean Cost Comparison')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Risk (std)
    axes[1].bar(strategies, std_costs, color='coral', alpha=0.7)
    axes[1].set_ylabel('Cost Standard Deviation')
    axes[1].set_title('Risk Comparison')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_stress_test_results(stress_results: Dict[str, Dict[str, Dict]], save_path: str = None):
    """Visualize stress test results across scenarios."""
    scenarios = list(stress_results.keys())
    strategies = list(stress_results[scenarios[0]].keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, scenario in enumerate(scenarios):
        if idx >= 4:
            break
        
        strategy_names = []
        mean_costs = []
        tail_risks = []
        
        for strategy in strategies:
            if strategy in stress_results[scenario]:
                metrics = stress_results[scenario][strategy]
                strategy_names.append(strategy)
                mean_costs.append(metrics['mean_cost'])
                tail_risks.append(metrics['tail_risk'])
        
        x = np.arange(len(strategy_names))
        width = 0.35
        
        ax = axes[idx]
        ax.bar(x - width/2, mean_costs, width, label='Mean Cost', alpha=0.7)
        ax.bar(x + width/2, tail_risks, width, label='Tail Risk', alpha=0.7)
        
        ax.set_ylabel('Cost')
        ax.set_title(scenario.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_ac_trajectories(risk_aversions: List[float], 
                        initial_inventory: float,
                        num_steps: int,
                        save_path: str = None):
    """Plot AC optimal trajectories for different risk aversions."""
    from models.almgren_chriss import AlmgrenChrissLinear
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lam in risk_aversions:
        ac = AlmgrenChrissLinear(
            initial_inventory=initial_inventory,
            num_steps=num_steps,
            volatility=0.02,
            eta=0.01,
            gamma=0.001,
            risk_aversion=lam
        )
        
        inventory = ac.get_inventory_trajectory()
        time_steps = np.arange(len(inventory))
        
        label = f'λ = {lam:.2f}' if lam > 0 else 'Risk-Neutral (TWAP)'
        ax.plot(time_steps, inventory, marker='o', label=label, linewidth=2)
    
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Remaining Inventory')
    ax.set_title('Almgren-Chriss Optimal Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def generate_all_plots():
    """Generate all analysis plots."""
    print("Generating analysis plots...")
    
    os.makedirs('results/figures', exist_ok=True)
    
    # Plot AC trajectories
    print("Plotting AC trajectories...")
    plot_ac_trajectories(
        risk_aversions=[0.0, 0.1, 0.5, 1.0, 2.0],
        initial_inventory=1000.0,
        num_steps=20,
        save_path='results/figures/ac_trajectories.png'
    )
    
    # Load and plot benchmark results if available
    if os.path.exists('results/benchmark_results.npy'):
        print("Plotting benchmark results...")
        results = np.load('results/benchmark_results.npy', allow_pickle=True).item()
        
        plot_strategy_comparison(
            results,
            save_path='results/figures/strategy_comparison.png'
        )
    
    # Load and plot stress test results if available
    if os.path.exists('results/stress_test_results.npy'):
        print("Plotting stress test results...")
        stress_results = np.load('results/stress_test_results.npy', allow_pickle=True).item()
        
        plot_stress_test_results(
            stress_results,
            save_path='results/figures/stress_tests.png'
        )
    
    print("All plots generated in results/figures/")


if __name__ == '__main__':
    generate_all_plots()
