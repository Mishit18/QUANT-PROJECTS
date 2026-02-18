"""
Train RL agents for optimal execution.

Workflow:
1. Collect offline data using TWAP/VWAP/AC
2. Train BCQ and TD3+BC on offline data
3. Evaluate trained agents

Run from repository root: python experiments/train_rl.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
import argparse
import os

from env.execution_env import ExecutionEnv
from env.liquidity_models import MeanRevertingLiquidity, LiquidityWithShocks
from env.impact_models import LinearImpact
from models.bcq import BCQ
from models.td3_bc import TD3_BC
from models.replay_buffer import ReplayBuffer
from models.twap import TWAP
from models.almgren_chriss import AlmgrenChrissLinear


def collect_offline_data(env, policy, num_episodes: int, replay_buffer: ReplayBuffer):
    """Collect offline dataset using behavioral policy."""
    print(f"Collecting {num_episodes} episodes of offline data...")
    
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        policy.reset()
        done = False
        
        while not done:
            action = policy.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
            
            replay_buffer.add(state, np.array([action]), next_state, reward, float(done))
            state = next_state
    
    print(f"Collected {replay_buffer.size} transitions")


def evaluate_policy(env, agent, num_episodes: int = 100) -> dict:
    """Evaluate agent performance."""
    costs = []
    completion_rates = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_cost = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(np.array([action]))
            done = terminated or truncated
            episode_cost += -reward
            state = next_state
        
        summary = env.get_execution_summary()
        costs.append(summary['total_cost'])
        completion_rates.append(summary['completion_rate'])
    
    return {
        'mean_cost': np.mean(costs),
        'std_cost': np.std(costs),
        'mean_completion': np.mean(completion_rates),
        'cost_5th_percentile': np.percentile(costs, 5),
        'cost_95th_percentile': np.percentile(costs, 95)
    }


def train_bcq(env, replay_buffer, args):
    """Train BCQ agent."""
    print("\nTraining BCQ...")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = BCQ(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=args.device
    )
    
    for iteration in tqdm(range(args.training_iterations)):
        metrics = agent.train(replay_buffer, batch_size=args.batch_size)
        
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: {metrics}")
    
    # Save model
    os.makedirs('models/checkpoints', exist_ok=True)
    agent.save('models/checkpoints/bcq.pt')
    
    # Evaluate
    results = evaluate_policy(env, agent, num_episodes=100)
    print(f"\nBCQ Results: {results}")
    
    return agent, results


def train_td3_bc(env, replay_buffer, args):
    """Train TD3+BC agent."""
    print("\nTraining TD3+BC...")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3_BC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=args.device
    )
    
    for iteration in tqdm(range(args.training_iterations)):
        metrics = agent.train(replay_buffer, batch_size=args.batch_size)
        
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}: {metrics}")
    
    # Save model
    os.makedirs('models/checkpoints', exist_ok=True)
    agent.save('models/checkpoints/td3_bc.pt')
    
    # Evaluate
    results = evaluate_policy(env, agent, num_episodes=100)
    print(f"\nTD3+BC Results: {results}")
    
    return agent, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='bcq', choices=['bcq', 'td3_bc', 'both'])
    parser.add_argument('--offline_episodes', type=int, default=1000)
    parser.add_argument('--training_iterations', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment with stochastic liquidity
    liquidity_process = LiquidityWithShocks(
        base_process=MeanRevertingLiquidity(mean=1.0, speed=0.5, volatility=0.2),
        shock_prob=0.01,
        shock_magnitude=-0.5
    )
    
    env = ExecutionEnv(
        initial_inventory=1000.0,
        num_steps=20,
        volatility=0.02,
        risk_aversion=0.5,
        liquidity_process=liquidity_process,
        impact_model=LinearImpact(eta=0.01, gamma=0.001),
        seed=args.seed
    )
    
    # Create replay buffer
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    replay_buffer = ReplayBuffer(state_dim, action_dim, device=args.device)
    
    # Collect offline data using TWAP
    twap_policy = TWAP(initial_inventory=1000.0, num_steps=20)
    collect_offline_data(env, twap_policy, args.offline_episodes, replay_buffer)
    
    # Train agents
    results = {}
    
    if args.agent in ['bcq', 'both']:
        bcq_agent, bcq_results = train_bcq(env, replay_buffer, args)
        results['bcq'] = bcq_results
    
    if args.agent in ['td3_bc', 'both']:
        td3_bc_agent, td3_bc_results = train_td3_bc(env, replay_buffer, args)
        results['td3_bc'] = td3_bc_results
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    for agent_name, agent_results in results.items():
        print(f"\n{agent_name.upper()}:")
        for metric, value in agent_results.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == '__main__':
    main()
