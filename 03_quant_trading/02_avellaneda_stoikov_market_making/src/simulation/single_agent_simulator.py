"""
Single-Agent Market Making Simulator

Simulates optimal market making with one agent.
"""

import numpy as np
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..market.market_environment import MarketEnvironment
from ..agents.market_maker import MarketMaker


class SingleAgentSimulator:
    """
    Simulator for single-agent market making.
    
    Runs complete simulation of:
    1. Price evolution
    2. Optimal quote generation
    3. Order arrivals and fills
    4. PnL tracking
    """
    
    def __init__(
        self,
        market_env: MarketEnvironment,
        agent: MarketMaker,
        terminal_time: float
    ):
        """
        Initialize simulator.
        
        Args:
            market_env: Market environment
            agent: Market maker agent
            terminal_time: Simulation end time
        """
        self.market_env = market_env
        self.agent = agent
        self.T = terminal_time
        self.dt = market_env.dt
        self.n_steps = int(self.T / self.dt)
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run single simulation.
        
        Args:
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with simulation results
        """
        # Reset environment and agent
        self.market_env.reset()
        self.agent.reset()
        
        # Simulation loop
        iterator = tqdm(range(self.n_steps), desc="Simulating") if verbose else range(self.n_steps)
        
        for step in iterator:
            current_time = step * self.dt
            mid_price = self.market_env.mid_price
            
            # Agent generates quotes
            bid, ask = self.agent.quote(mid_price, current_time)
            
            # Market processes quotes
            result = self.market_env.step(bid, ask)
            
            # Process fills
            if result['bid_filled']:
                self.agent.process_fill(
                    side='buy',
                    price=result['bid_fill_price'],
                    size=1,
                    time=result['time'],
                    mid_price=mid_price
                )
            
            if result['ask_filled']:
                self.agent.process_fill(
                    side='sell',
                    price=result['ask_fill_price'],
                    size=1,
                    time=result['time'],
                    mid_price=mid_price
                )
            
            # Update agent state
            self.agent.update_state(result['time'], result['mid_price'])
        
        # Get final results
        final_mid = self.market_env.mid_price
        metrics = self.agent.get_metrics(final_mid)
        history = self.agent.get_history()
        
        results = {
            'metrics': metrics,
            'history': history,
            'final_mid_price': final_mid,
            'n_steps': self.n_steps
        }
        
        return results
    
    def run_monte_carlo(
        self,
        n_simulations: int,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations.
        
        Args:
            n_simulations: Number of simulation runs
            verbose: Whether to show progress
        
        Returns:
            Dictionary with aggregated results
        """
        all_pnls = []
        all_inventories = []
        all_sharpes = []
        all_fills = []
        
        iterator = tqdm(range(n_simulations), desc="Monte Carlo") if verbose else range(n_simulations)
        
        for sim in iterator:
            result = self.run(verbose=False)
            metrics = result['metrics']
            
            all_pnls.append(metrics['final_pnl'])
            all_inventories.append(metrics['final_inventory'])
            all_sharpes.append(metrics['sharpe_ratio'])
            all_fills.append(metrics['total_fills'])
        
        # Aggregate statistics
        mc_results = {
            'n_simulations': n_simulations,
            'pnl': {
                'mean': np.mean(all_pnls),
                'std': np.std(all_pnls),
                'median': np.median(all_pnls),
                'min': np.min(all_pnls),
                'max': np.max(all_pnls),
                'q25': np.percentile(all_pnls, 25),
                'q75': np.percentile(all_pnls, 75)
            },
            'inventory': {
                'mean': np.mean(all_inventories),
                'std': np.std(all_inventories),
                'median': np.median(all_inventories)
            },
            'sharpe': {
                'mean': np.mean(all_sharpes),
                'std': np.std(all_sharpes),
                'median': np.median(all_sharpes)
            },
            'fills': {
                'mean': np.mean(all_fills),
                'std': np.std(all_fills)
            },
            'all_pnls': all_pnls,
            'all_inventories': all_inventories,
            'all_sharpes': all_sharpes
        }
        
        return mc_results
