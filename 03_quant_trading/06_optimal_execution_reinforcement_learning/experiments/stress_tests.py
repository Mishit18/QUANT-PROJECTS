"""
Stress testing under adverse market conditions.

Scenarios:
1. Liquidity collapse
2. Volatility spike
3. Impact regime shift
4. Combined stress

Run from repository root: python experiments/stress_tests.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import os

from env.execution_env import ExecutionEnv
from env.liquidity_models import MeanRevertingLiquidity, RegimeSwitchingLiquidity, LiquidityWithShocks
from env.impact_models import LinearImpact
from models.twap import TWAP
from models.almgren_chriss import AlmgrenChrissLinear
from models.bcq import BCQ
from models.td3_bc import TD3_BC


class ACPolicy:
    """Wrapper for AC policy."""
    
    def __init__(self, ac_model, initial_inventory: float):
        self.ac_model = ac_model
        self.initial_inventory = initial_inventory
        self.trades = ac_model.get_trades()
        self.step = 0
    
    def reset(self):
        self.step = 0
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        remaining_inventory = state[0] * self.initial_inventory
        
        if remaining_inventory < 1e-6 or self.step >= len(self.trades):
            return np.array([0.0])
        
        trade_size = self.trades[self.step]
        trade_fraction = trade_size / remaining_inventory
        trade_fraction = min(trade_fraction, 1.0)
        
        self.step += 1
        return np.array([trade_fraction])


def stress_test(env, strategies: dict, num_episodes: int = 100, scenario_name: str = "Stress") -> dict:
    """Run stress test on all strategies."""
    print(f"\n{'='*60}")
    print(f"Stress Test: {scenario_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for strategy_name, strategy in strategies.items():
        print(f"\nTesting {strategy_name}...")
        
        costs = []
        completion_rates = []
        failures = 0
        
        for _ in tqdm(range(num_episodes)):
            state, _ = env.reset()
            strategy.reset()
            done = False
            
            try:
                while not done:
                    action = strategy.select_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    state = next_state
                
                summary = env.get_execution_summary()
                costs.append(summary['total_cost'])
                completion_rates.append(summary['completion_rate'])
                
            except Exception as e:
                failures += 1
                costs.append(np.inf)
                completion_rates.append(0.0)
        
        # Filter out failures
        valid_costs = [c for c in costs if c != np.inf]
        
        if len(valid_costs) > 0:
            results[strategy_name] = {
                'mean_cost': np.mean(valid_costs),
                'std_cost': np.std(valid_costs),
                'max_cost': np.max(valid_costs),
                'cost_95th': np.percentile(valid_costs, 95),
                'mean_completion': np.mean(completion_rates),
                'failure_rate': failures / num_episodes,
                'tail_risk': np.percentile(valid_costs, 95) - np.mean(valid_costs)
            }
        else:
            results[strategy_name] = {
                'mean_cost': np.inf,
                'std_cost': np.inf,
                'max_cost': np.inf,
                'cost_95th': np.inf,
                'mean_completion': 0.0,
                'failure_rate': 1.0,
                'tail_risk': np.inf
            }
    
    return results


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    initial_inventory = 1000.0
    num_steps = 20
    
    # Load strategies
    strategies = {}
    
    # TWAP
    strategies['TWAP'] = TWAP(initial_inventory, num_steps)
    
    # AC
    ac_model = AlmgrenChrissLinear(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=0.02,
        eta=0.01,
        gamma=0.001,
        risk_aversion=0.5
    )
    strategies['AC'] = ACPolicy(ac_model, initial_inventory)
    
    # BCQ
    if os.path.exists('models/checkpoints/bcq.pt'):
        bcq = BCQ(state_dim=6, action_dim=1, max_action=0.3)
        bcq.load('models/checkpoints/bcq.pt')
        strategies['BCQ'] = bcq
    
    # TD3+BC
    if os.path.exists('models/checkpoints/td3_bc.pt'):
        td3_bc = TD3_BC(state_dim=6, action_dim=1, max_action=0.3)
        td3_bc.load('models/checkpoints/td3_bc.pt')
        strategies['TD3+BC'] = td3_bc
    
    all_results = {}
    
    # Scenario 1: Liquidity Collapse
    print("\n" + "="*60)
    print("SCENARIO 1: LIQUIDITY COLLAPSE")
    print("="*60)
    
    liquidity_collapse = MeanRevertingLiquidity(mean=0.3, speed=0.5, volatility=0.1)
    env_collapse = ExecutionEnv(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=0.02,
        liquidity_process=liquidity_collapse,
        impact_model=LinearImpact(eta=0.01, gamma=0.001),
        seed=seed
    )
    
    all_results['liquidity_collapse'] = stress_test(env_collapse, strategies, num_episodes=100, 
                                                     scenario_name="Liquidity Collapse")
    
    # Scenario 2: Volatility Spike
    print("\n" + "="*60)
    print("SCENARIO 2: VOLATILITY SPIKE")
    print("="*60)
    
    env_vol_spike = ExecutionEnv(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=0.10,  # 5x normal volatility
        liquidity_process=MeanRevertingLiquidity(mean=1.0, speed=0.5, volatility=0.2),
        impact_model=LinearImpact(eta=0.01, gamma=0.001),
        seed=seed
    )
    
    all_results['volatility_spike'] = stress_test(env_vol_spike, strategies, num_episodes=100,
                                                   scenario_name="Volatility Spike")
    
    # Scenario 3: Impact Regime Shift
    print("\n" + "="*60)
    print("SCENARIO 3: IMPACT REGIME SHIFT")
    print("="*60)
    
    env_impact_shift = ExecutionEnv(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=0.02,
        liquidity_process=MeanRevertingLiquidity(mean=1.0, speed=0.5, volatility=0.2),
        impact_model=LinearImpact(eta=0.05, gamma=0.005),  # 5x impact
        seed=seed
    )
    
    all_results['impact_shift'] = stress_test(env_impact_shift, strategies, num_episodes=100,
                                              scenario_name="Impact Regime Shift")
    
    # Scenario 4: Liquidity Shocks
    print("\n" + "="*60)
    print("SCENARIO 4: LIQUIDITY SHOCKS")
    print("="*60)
    
    liquidity_shocks = LiquidityWithShocks(
        base_process=MeanRevertingLiquidity(mean=1.0, speed=0.5, volatility=0.2),
        shock_prob=0.05,  # 5% chance per step
        shock_magnitude=-0.7  # 70% drop
    )
    
    env_shocks = ExecutionEnv(
        initial_inventory=initial_inventory,
        num_steps=num_steps,
        volatility=0.02,
        liquidity_process=liquidity_shocks,
        impact_model=LinearImpact(eta=0.01, gamma=0.001),
        seed=seed
    )
    
    all_results['liquidity_shocks'] = stress_test(env_shocks, strategies, num_episodes=100,
                                                   scenario_name="Liquidity Shocks")
    
    # Print summary
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    
    for scenario_name, scenario_results in all_results.items():
        print(f"\n{scenario_name.upper().replace('_', ' ')}:")
        print(f"{'Strategy':<15} {'Mean Cost':<12} {'Std Cost':<12} {'Tail Risk':<12} {'Failures':<10}")
        print("-"*80)
        
        for strategy_name, metrics in scenario_results.items():
            print(f"{strategy_name:<15} "
                  f"{metrics['mean_cost']:<12.2f} "
                  f"{metrics['std_cost']:<12.2f} "
                  f"{metrics['tail_risk']:<12.2f} "
                  f"{metrics['failure_rate']:<10.2%}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    np.save('results/stress_test_results.npy', all_results)
    print("\n\nResults saved to results/stress_test_results.npy")


if __name__ == '__main__':
    main()
