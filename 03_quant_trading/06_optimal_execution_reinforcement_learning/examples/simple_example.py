"""
Execution strategy comparison example.

Demonstrates:
1. Environment setup
2. Strategy execution (TWAP, VWAP, AC)
3. Results visualization

Run from repository root: python examples/simple_example.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt

from env.execution_env import ExecutionEnv
from env.liquidity_models import MeanRevertingLiquidity
from env.impact_models import LinearImpact
from models.twap import TWAP
from models.vwap import VWAP
from models.almgren_chriss import AlmgrenChrissLinear


class ACPolicy:
    """Wrapper for AC policy."""
    def __init__(self, ac_model, initial_inventory):
        self.ac_model = ac_model
        self.initial_inventory = initial_inventory
        self.trades = ac_model.get_trades()
        self.step = 0
    
    def reset(self):
        self.step = 0
    
    def get_action(self, state):
        remaining_inventory = state[0] * self.initial_inventory
        if remaining_inventory < 1e-6 or self.step >= len(self.trades):
            return 0.0
        trade_size = self.trades[self.step]
        trade_fraction = trade_size / remaining_inventory
        self.step += 1
        return min(trade_fraction, 1.0)


def run_strategy(env, strategy, name="Strategy"):
    """Run a single strategy and collect results."""
    print(f"\nRunning {name}...")
    
    state, _ = env.reset()
    strategy.reset()
    done = False
    
    inventories = [state[0] * 1000.0]
    prices = [state[2] * 100.0]
    costs = []
    
    while not done:
        action = np.array([strategy.get_action(state)])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        inventories.append(info['remaining_inventory'])
        prices.append(info['price'])
        costs.append(info['total_cost'])
        
        state = next_state
    
    summary = env.get_execution_summary()
    
    return {
        'inventories': inventories,
        'prices': prices,
        'costs': costs,
        'summary': summary
    }


def main():
    print("="*60)
    print("EXECUTION STRATEGY COMPARISON")
    print("="*60)
    
    # Parameters
    initial_inventory = 1000.0
    num_steps = 20
    volatility = 0.02
    eta = 0.01
    gamma = 0.001
    risk_aversion = 0.5
    seed = 42
    
    np.random.seed(seed)
    
    # Create environment
    print("\nCreating execution environment...")
    liquidity_process = MeanRevertingLiquidity(mean=1.0, speed=0.5, volatility=0.2)
    impact_model = LinearImpact(eta=eta, gamma=gamma)
    
    env = ExecutionEnv(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=volatility,
        risk_aversion=risk_aversion,
        liquidity_process=liquidity_process,
        impact_model=impact_model,
        seed=seed
    )
    
    # Create strategies
    print("Creating strategies...")
    
    twap = TWAP(initial_inventory, num_steps)
    vwap = VWAP(initial_inventory, num_steps)
    
    ac_neutral = AlmgrenChrissLinear(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=volatility,
        eta=eta,
        gamma=gamma,
        risk_aversion=0.0
    )
    ac_neutral_policy = ACPolicy(ac_neutral, initial_inventory)
    
    ac_averse = AlmgrenChrissLinear(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=volatility,
        eta=eta,
        gamma=gamma,
        risk_aversion=risk_aversion
    )
    ac_averse_policy = ACPolicy(ac_averse, initial_inventory)
    
    # Run strategies
    strategies = {
        'TWAP': twap,
        'VWAP': vwap,
        'AC-Neutral': ac_neutral_policy,
        'AC-Averse': ac_averse_policy
    }
    
    results = {}
    for name, strategy in strategies.items():
        results[name] = run_strategy(env, strategy, name)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Strategy':<15} {'Total Cost':<12} {'Avg Price':<12} {'Completion':<12}")
    print("-"*60)
    
    for name, result in results.items():
        summary = result['summary']
        print(f"{name:<15} "
              f"{summary['total_cost']:<12.2f} "
              f"{summary['average_price']:<12.2f} "
              f"{summary['completion_rate']:<12.2%}")
    
    # Plot results
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Inventory trajectories
    ax = axes[0, 0]
    for name, result in results.items():
        ax.plot(result['inventories'], marker='o', label=name, linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Remaining Inventory')
    ax.set_title('Inventory Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Price evolution
    ax = axes[0, 1]
    for name, result in results.items():
        ax.plot(result['prices'], marker='o', label=name, linewidth=2, alpha=0.7)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mid-Price')
    ax.set_title('Price Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative cost (negative PnL)
    ax = axes[1, 0]
    for name, result in results.items():
        if result['costs']:
            # Plot as negative (cost = negative PnL)
            pnl = [-c for c in result['costs']]
            ax.plot(pnl, marker='o', label=name, linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cumulative PnL (Negative Cost)')
    ax.set_title('Execution PnL Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final cost comparison
    ax = axes[1, 1]
    names = list(results.keys())
    costs = [results[name]['summary']['total_cost'] for name in names]
    colors = ['steelblue', 'coral', 'lightgreen', 'gold']
    ax.bar(names, costs, color=colors, alpha=0.7)
    ax.set_ylabel('Total Execution Cost')
    ax.set_title('Final Cost Comparison')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('execution_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'execution_comparison.png'")
    
    print("\n" + "="*60)
    print("ALMGREN-CHRISS ANALYSIS")
    print("="*60)
    
    print(f"\nRisk-Neutral (lambda=0):")
    print(f"  Expected cost: {ac_neutral.expected_cost():.2f}")
    print(f"  Cost variance: {ac_neutral.cost_variance():.2f}")
    print(f"  Strategy: Uniform liquidation")
    
    print(f"\nRisk-Averse (lambda={risk_aversion}):")
    print(f"  Expected cost: {ac_averse.expected_cost():.2f}")
    print(f"  Cost variance: {ac_averse.cost_variance():.2f}")
    print(f"  Strategy: Front-loaded execution")
    
    kappa = np.sqrt(risk_aversion * volatility**2 / eta)
    print(f"\nUrgency parameter kappa = {kappa:.4f}")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print("\nObservations:")
    print("- AC-Averse reduces variance through early execution")
    print("- TWAP minimizes impact through uniform trading")
    print("- VWAP follows volume patterns")
    print("\nNext steps:")
    print("  python experiments/train_rl.py")
    print("  python experiments/run_benchmarks.py")
    print("  python experiments/stress_tests.py")


if __name__ == '__main__':
    main()
