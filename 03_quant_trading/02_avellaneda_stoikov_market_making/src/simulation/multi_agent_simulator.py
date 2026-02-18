"""
Multi-Agent Market Making Simulator

Simulates competitive market making with multiple agents.
"""

import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm

from ..market.price_process import ArithmeticBrownianMotion
from ..market.order_flow import MultiAgentOrderFlow
from ..agents.competitive_mm import CompetitiveMarketMaker


class MultiAgentSimulator:
    """
    Simulator for multi-agent competitive market making.
    
    Features:
    - Multiple agents with heterogeneous preferences
    - Order allocation to best quotes
    - Competition dynamics
    - Equilibrium analysis
    """
    
    def __init__(
        self,
        agents: List[CompetitiveMarketMaker],
        initial_price: float,
        volatility: float,
        dt: float,
        terminal_time: float,
        order_flow: MultiAgentOrderFlow,
        random_seed: int = None
    ):
        """
        Initialize multi-agent simulator.
        
        Args:
            agents: List of competitive market maker agents
            initial_price: Starting mid-price
            volatility: Price volatility
            dt: Time step
            terminal_time: Simulation end time
            order_flow: Multi-agent order flow model
            random_seed: Random seed
        """
        self.agents = agents
        self.n_agents = len(agents)
        self.order_flow = order_flow
        self.T = terminal_time
        self.dt = dt
        self.n_steps = int(self.T / self.dt)
        
        self.rng = np.random.default_rng(random_seed)
        
        # Price process
        self.price_process = ArithmeticBrownianMotion(
            initial_price=initial_price,
            volatility=volatility,
            dt=dt
        )
        
        self.current_time = 0.0
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run multi-agent simulation.
        
        Args:
            verbose: Whether to show progress bar
        
        Returns:
            Dictionary with simulation results for all agents
        """
        # Reset all agents and price process
        self.price_process.reset()
        for agent in self.agents:
            agent.reset()
        
        self.current_time = 0.0
        
        # Track market-wide statistics
        bid_ask_spreads = []
        best_bid_history = []
        best_ask_history = []
        
        # Simulation loop
        iterator = tqdm(range(self.n_steps), desc="Multi-Agent Sim") if verbose else range(self.n_steps)
        
        for step in iterator:
            current_time = step * self.dt
            mid_price = self.price_process.current_price
            
            # All agents generate quotes SIMULTANEOUSLY (no sequential advantage)
            # Each agent uses their own AS optimal quotes without seeing others
            all_bids = []
            all_asks = []
            agent_quotes = {}
            
            for i, agent in enumerate(self.agents):
                # Each agent quotes independently based on AS model
                # No competitor information (simultaneous quoting)
                bid, ask = agent.competitive_quote(
                    mid_price, current_time, 
                    competitor_bids=None,  # Simultaneous quoting
                    competitor_asks=None
                )
                
                all_bids.append((i, bid))
                all_asks.append((i, ask))
                agent_quotes[i] = (bid, ask)
            
            # Update competition statistics
            for i, agent in enumerate(self.agents):
                bid, ask = agent_quotes[i]
                agent.update_competition_stats(
                    bid, ask,
                    [b for _, b in all_bids],
                    [a for _, a in all_asks]
                )
            
            # Record market statistics
            best_bid = max(b for _, b in all_bids)
            best_ask = min(a for _, a in all_asks)
            bid_ask_spread = best_ask - best_bid
            
            best_bid_history.append(best_bid)
            best_ask_history.append(best_ask)
            bid_ask_spreads.append(bid_ask_spread)
            
            # Allocate orders to best quotes
            bid_winner, ask_winner = self.order_flow.allocate_orders(
                all_bids, all_asks, self.dt, self.rng
            )
            
            # Process fills
            if bid_winner is not None:
                winner_agent = self.agents[bid_winner]
                fill_price = agent_quotes[bid_winner][0]
                winner_agent.process_fill('buy', fill_price, 1, current_time, mid_price)
                
                # Adverse selection
                impact = self.order_flow.adverse_selection_impact('buy', mid_price, self.rng)
                self.price_process.current_price += impact
            
            if ask_winner is not None:
                winner_agent = self.agents[ask_winner]
                fill_price = agent_quotes[ask_winner][1]
                winner_agent.process_fill('sell', fill_price, 1, current_time, mid_price)
                
                # Adverse selection
                impact = self.order_flow.adverse_selection_impact('sell', mid_price, self.rng)
                self.price_process.current_price += impact
            
            # Update price (Brownian motion)
            self.price_process.step(self.rng)
            new_mid = self.price_process.current_price
            
            # Update all agents
            for agent in self.agents:
                agent.update_state(current_time + self.dt, new_mid)
        
        # Collect results
        final_mid = self.price_process.current_price
        
        agent_results = []
        for agent in self.agents:
            metrics = agent.get_metrics(final_mid)
            history = agent.get_history()
            agent_results.append({
                'name': agent.name,
                'metrics': metrics,
                'history': history
            })
        
        market_stats = {
            'bid_ask_spreads': np.array(bid_ask_spreads),
            'best_bids': np.array(best_bid_history),
            'best_asks': np.array(best_ask_history),
            'avg_spread': np.mean(bid_ask_spreads),
            'min_spread': np.min(bid_ask_spreads),
            'max_spread': np.max(bid_ask_spreads)
        }
        
        results = {
            'agents': agent_results,
            'market_stats': market_stats,
            'final_mid_price': final_mid,
            'n_steps': self.n_steps,
            'n_agents': self.n_agents
        }
        
        return results
    
    def run_monte_carlo(
        self,
        n_simulations: int,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for multi-agent scenario.
        
        Args:
            n_simulations: Number of simulation runs
            verbose: Whether to show progress
        
        Returns:
            Aggregated results across simulations
        """
        all_agent_pnls = {agent.name: [] for agent in self.agents}
        all_spreads = []
        
        iterator = tqdm(range(n_simulations), desc="MC Multi-Agent") if verbose else range(n_simulations)
        
        for sim in iterator:
            result = self.run(verbose=False)
            
            # Collect agent PnLs
            for agent_result in result['agents']:
                name = agent_result['name']
                pnl = agent_result['metrics']['final_pnl']
                all_agent_pnls[name].append(pnl)
            
            # Collect spread statistics
            all_spreads.append(result['market_stats']['avg_spread'])
        
        # Aggregate
        mc_results = {
            'n_simulations': n_simulations,
            'agent_pnls': {
                name: {
                    'mean': np.mean(pnls),
                    'std': np.std(pnls),
                    'median': np.median(pnls)
                }
                for name, pnls in all_agent_pnls.items()
            },
            'spreads': {
                'mean': np.mean(all_spreads),
                'std': np.std(all_spreads),
                'median': np.median(all_spreads)
            },
            'all_agent_pnls': all_agent_pnls,
            'all_spreads': all_spreads
        }
        
        return mc_results
