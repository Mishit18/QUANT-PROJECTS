"""
Stochastic liquidity models for execution simulation.

Implements:
- Mean-reverting liquidity (Ornstein-Uhlenbeck)
- Regime-switching liquidity
- Liquidity shocks
"""

import numpy as np


class LiquidityProcess:
    """Base class for liquidity dynamics."""
    
    def step(self, dt: float = 1.0) -> float:
        """Generate next liquidity level."""
        raise NotImplementedError
    
    def reset(self):
        """Reset process to initial state."""
        raise NotImplementedError


class MeanRevertingLiquidity(LiquidityProcess):
    """Ornstein-Uhlenbeck process for liquidity."""
    
    def __init__(self, 
                 mean: float = 1.0,
                 speed: float = 0.5,
                 volatility: float = 0.2,
                 initial: float = None):
        """
        dL_t = κ(μ - L_t)dt + σ dW_t
        
        Args:
            mean: Long-run mean liquidity level
            speed: Mean reversion speed (κ)
            volatility: Liquidity volatility (σ)
            initial: Initial liquidity level
        """
        self.mean = mean
        self.speed = speed
        self.volatility = volatility
        self.initial = initial if initial is not None else mean
        self.current = self.initial
    
    def step(self, dt: float = 1.0) -> float:
        """Simulate one step of OU process."""
        drift = self.speed * (self.mean - self.current) * dt
        diffusion = self.volatility * np.sqrt(dt) * np.random.randn()
        
        self.current = np.maximum(0.1, self.current + drift + diffusion)
        return self.current
    
    def reset(self):
        """Reset to initial liquidity."""
        self.current = self.initial


class RegimeSwitchingLiquidity(LiquidityProcess):
    """Two-state Markov regime switching liquidity."""
    
    def __init__(self,
                 high_liquidity: float = 1.5,
                 low_liquidity: float = 0.5,
                 transition_prob: float = 0.05):
        """
        Args:
            high_liquidity: Liquidity in high regime
            low_liquidity: Liquidity in low regime
            transition_prob: Probability of regime switch per step
        """
        self.high_liquidity = high_liquidity
        self.low_liquidity = low_liquidity
        self.transition_prob = transition_prob
        self.high_regime = True
        self.current = high_liquidity
    
    def step(self, dt: float = 1.0) -> float:
        """Simulate regime switch."""
        if np.random.rand() < self.transition_prob * dt:
            self.high_regime = not self.high_regime
        
        self.current = self.high_liquidity if self.high_regime else self.low_liquidity
        return self.current
    
    def reset(self):
        """Reset to high regime."""
        self.high_regime = True
        self.current = self.high_liquidity


class LiquidityWithShocks(LiquidityProcess):
    """Mean-reverting liquidity with occasional shocks."""
    
    def __init__(self,
                 base_process: LiquidityProcess,
                 shock_prob: float = 0.01,
                 shock_magnitude: float = -0.5):
        """
        Args:
            base_process: Underlying liquidity process
            shock_prob: Probability of shock per step
            shock_magnitude: Multiplicative shock size
        """
        self.base_process = base_process
        self.shock_prob = shock_prob
        self.shock_magnitude = shock_magnitude
    
    def step(self, dt: float = 1.0) -> float:
        """Step with potential shock."""
        liquidity = self.base_process.step(dt)
        
        if np.random.rand() < self.shock_prob * dt:
            liquidity *= (1 + self.shock_magnitude)
            liquidity = np.maximum(0.1, liquidity)
        
        return liquidity
    
    def reset(self):
        """Reset base process."""
        self.base_process.reset()
