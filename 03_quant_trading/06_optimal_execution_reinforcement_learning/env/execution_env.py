"""
Gym-style execution environment with stochastic microstructure.

State: [inventory, time_remaining, price, volatility, liquidity, recent_impact]
Action: trade_size (fraction of remaining inventory)
Reward: -execution_cost - risk_penalty - constraint_violations
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional
from env.liquidity_models import LiquidityProcess, MeanRevertingLiquidity
from env.impact_models import ImpactModel, LinearImpact, StochasticImpactDecay


class ExecutionEnv(gym.Env):
    """Optimal execution environment with stochastic microstructure."""
    
    def __init__(self,
                 initial_inventory: float = 1000.0,
                 num_steps: int = 20,
                 initial_price: float = 100.0,
                 volatility: float = 0.02,
                 risk_aversion: float = 0.5,
                 liquidity_process: Optional[LiquidityProcess] = None,
                 impact_model: Optional[ImpactModel] = None,
                 max_trade_fraction: float = 0.3,
                 slippage_std: float = 0.001,
                 seed: Optional[int] = None):
        """
        Args:
            initial_inventory: Starting inventory to liquidate
            num_steps: Number of execution periods
            initial_price: Starting mid-price
            volatility: Price volatility (σ)
            risk_aversion: Risk aversion parameter (λ)
            liquidity_process: Stochastic liquidity model
            impact_model: Market impact model
            max_trade_fraction: Maximum fraction of inventory per trade
            slippage_std: Standard deviation of random slippage
            seed: Random seed
        """
        super().__init__()
        
        self.initial_inventory = initial_inventory
        self.num_steps = num_steps
        self.initial_price = initial_price
        self.volatility = volatility
        self.risk_aversion = risk_aversion
        self.max_trade_fraction = max_trade_fraction
        self.slippage_std = slippage_std
        self.dt = 1.0
        
        # Default models
        self.liquidity_process = liquidity_process or MeanRevertingLiquidity()
        self.impact_model = impact_model or LinearImpact()
        self.impact_decay = StochasticImpactDecay()
        
        # State: [inventory, time_remaining, price, volatility, liquidity, recent_impact]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, 0, -np.inf], dtype=np.float32),
            high=np.array([np.inf, num_steps, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action: trade size as fraction of remaining inventory [0, max_trade_fraction]
        self.action_space = gym.spaces.Box(
            low=0.0,
            high=max_trade_fraction,
            shape=(1,),
            dtype=np.float32
        )
        
        if seed is not None:
            self.seed(seed)
        
        self.reset()
    
    def seed(self, seed: int):
        """Set random seed."""
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed)
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        self.inventory = self.initial_inventory
        self.time_step = 0
        self.price = self.initial_price
        self.realized_volatility = self.volatility
        
        self.liquidity_process.reset()
        self.liquidity = self.liquidity_process.step(self.dt)
        
        self.impact_decay.reset()
        self.recent_impact = 0.0
        
        self.total_cost = 0.0
        self.execution_prices = []
        self.trades = []
        
        state = self._get_state()
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one trading step.
        
        Returns:
            state, reward, terminated, truncated, info
        """
        trade_fraction = np.clip(action[0], 0, self.max_trade_fraction)
        trade_size = trade_fraction * self.inventory
        
        # Ensure we don't overtrade
        trade_size = min(trade_size, self.inventory)
        
        # Compute market impact
        temp_impact = self.impact_model.temporary_impact(trade_size, self.liquidity)
        perm_impact = self.impact_model.permanent_impact(trade_size, self.liquidity)
        
        # Random slippage
        slippage = self.slippage_std * np.random.randn() * self.price
        
        # Execution price (worse than mid due to impact and slippage)
        execution_price = self.price - temp_impact - slippage
        
        # Execution cost for this trade
        trade_cost = trade_size * execution_price
        self.total_cost += trade_cost
        
        # Update inventory
        self.inventory -= trade_size
        
        # Update price with permanent impact and diffusion
        price_diffusion = self.volatility * np.sqrt(self.dt) * np.random.randn()
        self.price = self.price + self.price * price_diffusion - perm_impact
        self.price = max(self.price, 0.01)  # Prevent negative prices
        
        # Update liquidity
        self.liquidity = self.liquidity_process.step(self.dt)
        
        # Update impact decay
        self.recent_impact = self.impact_decay.update(temp_impact, self.dt)
        
        # Update realized volatility (simple EWMA)
        self.realized_volatility = 0.9 * self.realized_volatility + 0.1 * abs(price_diffusion)
        
        # Store trade info
        self.execution_prices.append(execution_price)
        self.trades.append(trade_size)
        
        # Advance time
        self.time_step += 1
        
        # Compute reward
        reward = self._compute_reward(trade_size, temp_impact)
        
        # Check termination
        terminated = (self.time_step >= self.num_steps) or (self.inventory < 1e-6)
        
        # Terminal penalty for remaining inventory
        if terminated and self.inventory > 1e-6:
            # Liquidate remaining at unfavorable price
            terminal_penalty = self.inventory * self.price * 0.5  # 50% haircut
            reward -= terminal_penalty
        
        state = self._get_state()
        
        info = {
            'total_cost': self.total_cost,
            'remaining_inventory': self.inventory,
            'liquidity': self.liquidity,
            'price': self.price,
            'trade_size': trade_size
        }
        
        return state, reward, terminated, False, info
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector."""
        time_remaining = self.num_steps - self.time_step
        
        state = np.array([
            self.inventory / self.initial_inventory,  # Normalized inventory
            time_remaining / self.num_steps,  # Normalized time
            self.price / self.initial_price,  # Normalized price
            self.realized_volatility,
            self.liquidity,
            self.recent_impact
        ], dtype=np.float32)
        
        return state
    
    def _compute_reward(self, trade_size: float, impact: float) -> float:
        """
        Compute step reward.
        
        Reward = -execution_cost - risk_penalty - constraint_penalty
        """
        # Execution cost (negative of revenue)
        execution_cost = trade_size * self.price
        
        # Risk penalty (variance of cost)
        risk_penalty = self.risk_aversion * (trade_size ** 2) * (self.volatility ** 2)
        
        # Constraint penalty
        constraint_penalty = 0.0
        
        # Penalty for trading too aggressively
        if trade_size > self.max_trade_fraction * self.inventory:
            constraint_penalty += 100.0
        
        # Penalty for not finishing on time
        if self.time_step == self.num_steps - 1 and self.inventory > 0.1 * self.initial_inventory:
            constraint_penalty += 1000.0 * (self.inventory / self.initial_inventory)
        
        reward = -execution_cost - risk_penalty - constraint_penalty
        
        return reward
    
    def get_execution_summary(self) -> Dict:
        """Return summary statistics of execution."""
        avg_price = np.mean(self.execution_prices) if self.execution_prices else 0
        
        return {
            'total_cost': self.total_cost,
            'average_price': avg_price,
            'price_impact': (self.initial_price - avg_price) / self.initial_price,
            'num_trades': len(self.trades),
            'remaining_inventory': self.inventory,
            'completion_rate': 1 - (self.inventory / self.initial_inventory)
        }
