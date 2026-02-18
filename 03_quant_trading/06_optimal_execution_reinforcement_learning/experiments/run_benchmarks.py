"""
Run and compare all execution strategies.

Benchmarks:
- TWAP
- VWAP
- Almgren-Chriss (risk-neutral and risk-averse)
- BCQ (if trained)
- TD3+BC (if trained)

Run from repository root: python experiments/run_benchmarks.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import os

from env.execution_env import ExecutionEnv
from env.liquidity_models import MeanRevertingLiquidity
from env.impact_models import LinearImpact
from models.twap import TWAP
from models.vwap import VWAP
from models.almgren_chriss import AlmgrenChrissLinear
from models.bcq import BCQ
from models.td3_bc import TD3_BC


class ACPolicy:
    """Wrapper for Almgren-Chriss policy."""
    
    def __init__(self, ac_model: AlmgrenChrissLinear, initial_inventory: float):
        self.ac_model = ac_model
        self.initial_inventory = initial_inventory
        self.trades = ac_model.get_trades()
        self.step = 0
    
    def reset(self):
        self.step = 0
    
    def get_action(self, state: np.ndarray) -> float:
        remaining_inventory = state[0] * self.initial_inventory
        
        if remaining_inventory < 1e-6 or self.step >= len(self.trades):
            return 0.0
        
        trade_size = self.trades[self.step]
        trade_fraction = trade_size / remaining_inventory
        trade_fraction = min(trade_fraction, 1.0)
        
        self.step += 1
        return trade_fraction
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        return np.array([self.get_action(state)])


def evaluate_strategy(env, strategy, num_episodes: int = 100, name: str = "Strategy") -> dict:
    """Evaluate execution strategy."""
    print(f"\nEvaluating {name}...")
    
    costs = []
    completion_rates = []
    price_impacts = []
    
    for _ in tqdm(range(num_episodes)):
        state, _ = env.reset()
        strategy.reset()
        done = False
        
        while not done:
            if hasattr(strategy, 'select_action'):
                action = strategy.select_action(state)
            else:
                action = np.array([strategy.get_action(state)])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
        
        summary = env.get_execution_summary()
        costs.append(summary['total_cost'])
        completion_rates.append(summary['completion_rate'])
        price_impacts.append(summary['price_impact'])
    
    results = {
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'mean_completion': np.mean(completion_rates),
        'mean_impact': np.mean(price_impacts),
        'cost_5th': np.percentile(costs, 5),
        'cost_95th': np.percentile(costs, 95),
        'sharpe': -np.mean(costs) / (np.std(costs) + 1e-6)
    }
    
    return results


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Environment parameters
    initial_inventory = 1000.0
    num_steps = 20
    volatility = 0.02
    eta = 0.01
    gamma = 0.001
    risk_aversion = 0.5
    
    # Create environment
    liquidity_process = MeanRevertingLiquidity(mean=1.0, speed=0.5, volatility=0.2)
    
    env = ExecutionEnv(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=volatility,
        risk_aversion=risk_aversion,
        liquidity_process=liquidity_process,
        impact_model=LinearImpact(eta=eta, gamma=gamma),
        seed=seed
    )
    
    results = {}
    
    # TWAP
    twap = TWAP(initial_inventory, num_steps)
    results['TWAP'] = evaluate_strategy(env, twap, num_episodes=100, name='TWAP')
    
    # VWAP
    vwap = VWAP(initial_inventory, num_steps)
    results['VWAP'] = evaluate_strategy(env, vwap, num_episodes=100, name='VWAP')
    
    # Almgren-Chriss (risk-neutral)
    ac_neutral = AlmgrenChrissLinear(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=volatility,
        eta=eta,
        gamma=gamma,
        risk_aversion=0.0
    )
    ac_neutral_policy = ACPolicy(ac_neutral, initial_inventory)
    results['AC-Neutral'] = evaluate_strategy(env, ac_neutral_policy, num_episodes=100, name='AC Risk-Neutral')
    
    # Almgren-Chriss (risk-averse)
    ac_averse = AlmgrenChrissLinear(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=volatility,
        eta=eta,
        gamma=gamma,
        risk_aversion=risk_aversion
    )
    ac_averse_policy = ACPolicy(ac_averse, initial_inventory)
    results['AC-Averse'] = evaluate_strategy(env, ac_averse_policy, num_episodes=100, name='AC Risk-Averse')
    
    # BCQ (if available)
    if os.path.exists('models/checkpoints/bcq.pt'):
        print("\nLoading BCQ agent...")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        bcq = BCQ(state_dim, action_dim, max_action)
        bcq.load('models/checkpoints/bcq.pt')
        results['BCQ'] = evaluate_strategy(env, bcq, num_episodes=100, name='BCQ')
    
    # TD3+BC (if available)
    if os.path.exists('models/checkpoints/td3_bc.pt'):
        print("\nLoading TD3+BC agent...")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        
        td3_bc = TD3_BC(state_dim, action_dim, max_action)
        td3_bc.load('models/checkpoints/td3_bc.pt')
        results['TD3+BC'] = evaluate_strategy(env, td3_bc, num_episodes=100, name='TD3+BC')
    
    # Print comparison table
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Strategy':<15} {'Mean Cost':<12} {'Std Cost':<12} {'Completion':<12} {'Sharpe':<10}")
    print("-"*80)
    
    for strategy_name, strategy_results in results.items():
        print(f"{strategy_name:<15} "
              f"{strategy_results['mean_cost']:<12.2f} "
              f"{strategy_results['std_cost']:<12.2f} "
              f"{strategy_results['mean_completion']:<12.4f} "
              f"{strategy_results['sharpe']:<10.4f}")
    
    print("="*80)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    np.save('results/benchmark_results.npy', results)
    print("\nResults saved to results/benchmark_results.npy")


if __name__ == '__main__':
    main()
