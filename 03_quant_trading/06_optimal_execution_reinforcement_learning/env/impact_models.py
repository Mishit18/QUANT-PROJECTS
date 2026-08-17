"""
Market impact models for execution simulation.

Implements:
- Linear temporary impact
- Quadratic temporary impact
- Permanent impact with decay
"""

import numpy as np


class ImpactModel:
    """Base class for market impact models."""
    
    def temporary_impact(self, trade_size: float, liquidity: float) -> float:
        """Compute temporary price impact from trade."""
        raise NotImplementedError
    
    def permanent_impact(self, trade_size: float, liquidity: float) -> float:
        """Compute permanent price impact from trade."""
        raise NotImplementedError


class LinearImpact(ImpactModel):
    """Linear temporary impact model (Almgren-Chriss)."""
    
    def __init__(self, eta: float = 0.01, gamma: float = 0.0):
        """
        Args:
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
        """
        self.eta = eta
        self.gamma = gamma
    
    def temporary_impact(self, trade_size: float, liquidity: float) -> float:
        """Linear temporary impact: η * v^2 / L"""
        return self.eta * (trade_size ** 2) / liquidity
    
    def permanent_impact(self, trade_size: float, liquidity: float) -> float:
        """Linear permanent impact: γ * v"""
        return self.gamma * trade_size


class QuadraticImpact(ImpactModel):
    """Quadratic temporary impact model."""
    
    def __init__(self, eta: float = 0.01, phi: float = 0.5, gamma: float = 0.0):
        """
        Args:
            eta: Impact coefficient
            phi: Exponent (1 + phi), typically 0.5 for 3/2 power law
            gamma: Permanent impact coefficient
        """
        self.eta = eta
        self.phi = phi
        self.gamma = gamma
    
    def temporary_impact(self, trade_size: float, liquidity: float) -> float:
        """Quadratic impact: η * |v|^(1+φ) / L"""
        return self.eta * (np.abs(trade_size) ** (1 + self.phi)) / liquidity
    
    def permanent_impact(self, trade_size: float, liquidity: float) -> float:
        """Linear permanent impact: γ * v"""
        return self.gamma * trade_size


class StochasticImpactDecay:
    """Models stochastic decay of market impact over time."""
    
    def __init__(self, decay_rate: float = 0.1, decay_noise: float = 0.02):
        """
        Args:
            decay_rate: Mean exponential decay rate
            decay_noise: Volatility of decay process
        """
        self.decay_rate = decay_rate
        self.decay_noise = decay_noise
        self.accumulated_impact = 0.0
    
    def update(self, new_impact: float, dt: float = 1.0) -> float:
        """
        Update accumulated impact with decay and new impact.
        
        Returns:
            Current impact level after decay
        """
        # Exponential decay with noise
        decay_factor = np.exp(-self.decay_rate * dt + 
                             self.decay_noise * np.sqrt(dt) * np.random.randn())
        
        self.accumulated_impact = self.accumulated_impact * decay_factor + new_impact
        return self.accumulated_impact
    
    def reset(self):
        """Reset accumulated impact."""
        self.accumulated_impact = 0.0
