"""
Price Process Models

Stochastic models for mid-price evolution.
"""

import numpy as np
from typing import Optional


class ArithmeticBrownianMotion:
    """
    Arithmetic Brownian Motion for mid-price.
    
    The mid-price follows:
        dS_t = σ dW_t
    
    where W_t is a standard Brownian motion.
    
    This is the simplest model and is analytically tractable.
    In practice, more sophisticated models (OU, jump-diffusion) may be used.
    """
    
    def __init__(self, initial_price: float, volatility: float, dt: float):
        """
        Initialize price process.
        
        Args:
            initial_price: S_0 - starting mid-price
            volatility: σ - volatility parameter
            dt: Time step for discretization
        """
        self.S0 = initial_price
        self.sigma = volatility
        self.dt = dt
        self.current_price = initial_price
        self.time = 0.0
        self.price_history = [initial_price]
        self.time_history = [0.0]
    
    def step(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Simulate one time step of price evolution.
        
        Discretization:
            S_{t+dt} = S_t + σ √dt · Z
        
        where Z ~ N(0,1).
        
        Args:
            rng: Random number generator (optional)
        
        Returns:
            New mid-price
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Brownian increment
        dW = rng.normal(0, np.sqrt(self.dt))
        dS = self.sigma * dW
        
        self.current_price += dS
        self.time += self.dt
        
        self.price_history.append(self.current_price)
        self.time_history.append(self.time)
        
        return self.current_price
    
    def simulate_path(
        self,
        n_steps: int,
        rng: Optional[np.random.Generator] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate entire price path.
        
        Args:
            n_steps: Number of time steps
            rng: Random number generator (optional)
        
        Returns:
            (times, prices) - arrays of time points and prices
        """
        if rng is None:
            rng = np.random.default_rng()
        
        times = np.arange(n_steps + 1) * self.dt
        prices = np.zeros(n_steps + 1)
        prices[0] = self.S0
        
        # Generate all Brownian increments at once (vectorized)
        dW = rng.normal(0, np.sqrt(self.dt), n_steps)
        dS = self.sigma * dW
        
        # Cumulative sum to get price path
        prices[1:] = self.S0 + np.cumsum(dS)
        
        return times, prices
    
    def reset(self):
        """Reset price process to initial state."""
        self.current_price = self.S0
        self.time = 0.0
        self.price_history = [self.S0]
        self.time_history = [0.0]
    
    def get_history(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get price history.
        
        Returns:
            (times, prices) - historical time points and prices
        """
        return np.array(self.time_history), np.array(self.price_history)


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process for mean-reverting mid-price.
    
    The mid-price follows:
        dS_t = θ(μ - S_t)dt + σ dW_t
    
    where:
        θ: mean reversion speed
        μ: long-term mean
        σ: volatility
    
    This model is more realistic for certain assets but less tractable analytically.
    """
    
    def __init__(
        self,
        initial_price: float,
        mean_price: float,
        reversion_speed: float,
        volatility: float,
        dt: float
    ):
        """
        Initialize OU process.
        
        Args:
            initial_price: S_0 - starting price
            mean_price: μ - long-term mean
            reversion_speed: θ - mean reversion speed
            volatility: σ - volatility
            dt: Time step
        """
        self.S0 = initial_price
        self.mu = mean_price
        self.theta = reversion_speed
        self.sigma = volatility
        self.dt = dt
        self.current_price = initial_price
        self.time = 0.0
    
    def step(self, rng: Optional[np.random.Generator] = None) -> float:
        """
        Simulate one time step.
        
        Euler-Maruyama discretization:
            S_{t+dt} = S_t + θ(μ - S_t)dt + σ√dt · Z
        
        Args:
            rng: Random number generator (optional)
        
        Returns:
            New price
        """
        if rng is None:
            rng = np.random.default_rng()
        
        drift = self.theta * (self.mu - self.current_price) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * rng.normal()
        
        self.current_price += drift + diffusion
        self.time += self.dt
        
        return self.current_price
    
    def simulate_path(
        self,
        n_steps: int,
        rng: Optional[np.random.Generator] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate entire price path.
        
        Args:
            n_steps: Number of time steps
            rng: Random number generator (optional)
        
        Returns:
            (times, prices)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        times = np.arange(n_steps + 1) * self.dt
        prices = np.zeros(n_steps + 1)
        prices[0] = self.S0
        
        for i in range(n_steps):
            drift = self.theta * (self.mu - prices[i]) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * rng.normal()
            prices[i + 1] = prices[i] + drift + diffusion
        
        return times, prices
