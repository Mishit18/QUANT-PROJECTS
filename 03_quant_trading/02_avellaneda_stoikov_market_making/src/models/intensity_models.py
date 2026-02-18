"""
Order Arrival Intensity Models

Models for quote-dependent order arrival rates in market making.
"""

import numpy as np
from typing import Optional


class ExponentialIntensity:
    """
    Exponential intensity model for order arrivals.
    
    The arrival rate of orders decreases exponentially with spread:
        λ(δ) = A exp(-κ δ)
    
    where:
        A: base arrival rate (orders per unit time at zero spread)
        κ: decay rate (sensitivity to spread)
        δ: spread from mid-price
    
    Interpretation:
        - Higher spread → lower fill probability
        - κ measures market elasticity
        - A measures market activity level
    """
    
    def __init__(self, arrival_rate: float, decay_rate: float):
        """
        Initialize exponential intensity model.
        
        Args:
            arrival_rate: A - base arrival rate
            decay_rate: κ - exponential decay rate
        """
        self.A = arrival_rate
        self.kappa = decay_rate
    
    def intensity(self, spread: float) -> float:
        """
        Calculate order arrival intensity for given spread.
        
        Formula:
            λ(δ) = A exp(-κ δ)
        
        Args:
            spread: δ - spread from mid-price (must be non-negative)
        
        Returns:
            Arrival intensity λ(δ)
        """
        if spread < 0:
            raise ValueError(f"Spread must be non-negative, got {spread}")
        
        return self.A * np.exp(-self.kappa * spread)
    
    def intensity_derivative(self, spread: float) -> float:
        """
        Calculate derivative of intensity with respect to spread.
        
        Formula:
            dλ/dδ = -κ A exp(-κ δ) = -κ λ(δ)
        
        Args:
            spread: δ - spread from mid-price
        
        Returns:
            Derivative dλ/dδ
        """
        return -self.kappa * self.intensity(spread)
    
    def expected_fill_time(self, spread: float) -> float:
        """
        Calculate expected time until next fill.
        
        For Poisson process with intensity λ, expected inter-arrival time is 1/λ.
        
        Args:
            spread: δ - spread from mid-price
        
        Returns:
            Expected time until fill
        """
        lambda_val = self.intensity(spread)
        if lambda_val <= 0:
            return np.inf
        return 1.0 / lambda_val
    
    def fill_probability(self, spread: float, time_horizon: float) -> float:
        """
        Calculate probability of at least one fill within time horizon.
        
        For Poisson process: P(N(t) ≥ 1) = 1 - exp(-λt)
        
        Args:
            spread: δ - spread from mid-price
            time_horizon: Time window
        
        Returns:
            Probability of fill
        """
        lambda_val = self.intensity(spread)
        return 1.0 - np.exp(-lambda_val * time_horizon)
    
    def sample_arrival_time(self, spread: float, rng: Optional[np.random.Generator] = None) -> float:
        """
        Sample next arrival time from exponential distribution.
        
        Args:
            spread: δ - spread from mid-price
            rng: Random number generator (optional)
        
        Returns:
            Time until next arrival
        """
        if rng is None:
            rng = np.random.default_rng()
        
        lambda_val = self.intensity(spread)
        if lambda_val <= 0:
            return np.inf
        
        return rng.exponential(1.0 / lambda_val)
    
    def optimal_spread_no_inventory(self, risk_aversion: float) -> float:
        """
        Calculate optimal spread with zero inventory (symmetric case).
        
        From HJB first-order condition:
            δ* = (1/γ) log(1 + γ/κ)
        
        Args:
            risk_aversion: γ - risk aversion parameter
        
        Returns:
            Optimal spread δ*
        """
        return (1.0 / risk_aversion) * np.log(1.0 + risk_aversion / self.kappa)
    
    def elasticity(self, spread: float) -> float:
        """
        Calculate elasticity of intensity with respect to spread.
        
        Elasticity: ε = (dλ/dδ) · (δ/λ) = -κδ
        
        Args:
            spread: δ - spread from mid-price
        
        Returns:
            Elasticity ε
        """
        return -self.kappa * spread


class PowerLawIntensity:
    """
    Power-law intensity model (alternative to exponential).
    
    The arrival rate follows a power law:
        λ(δ) = A / (1 + κδ)^α
    
    where α controls the decay rate.
    
    Note: This is less tractable analytically but may better fit empirical data.
    """
    
    def __init__(self, arrival_rate: float, decay_rate: float, exponent: float = 2.0):
        """
        Initialize power-law intensity model.
        
        Args:
            arrival_rate: A - base arrival rate
            decay_rate: κ - decay rate
            exponent: α - power law exponent
        """
        self.A = arrival_rate
        self.kappa = decay_rate
        self.alpha = exponent
    
    def intensity(self, spread: float) -> float:
        """
        Calculate order arrival intensity.
        
        Formula:
            λ(δ) = A / (1 + κδ)^α
        
        Args:
            spread: δ - spread from mid-price
        
        Returns:
            Arrival intensity λ(δ)
        """
        if spread < 0:
            raise ValueError(f"Spread must be non-negative, got {spread}")
        
        return self.A / ((1.0 + self.kappa * spread) ** self.alpha)
    
    def intensity_derivative(self, spread: float) -> float:
        """
        Calculate derivative of intensity.
        
        Formula:
            dλ/dδ = -α κ A / (1 + κδ)^(α+1)
        
        Args:
            spread: δ - spread from mid-price
        
        Returns:
            Derivative dλ/dδ
        """
        return -self.alpha * self.kappa * self.A / ((1.0 + self.kappa * spread) ** (self.alpha + 1))
