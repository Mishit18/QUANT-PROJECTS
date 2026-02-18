"""
Almgren-Chriss optimal execution model.

Derives and implements closed-form solutions for:
- Risk-neutral case
- Risk-averse case
- Linear temporary impact
- Quadratic temporary impact (approximation)

References:
    Almgren & Chriss (2000): Optimal execution of portfolio transactions
"""

import numpy as np
from typing import List, Tuple


class AlmgrenChrissLinear:
    """
    Almgren-Chriss model with linear temporary impact.
    
    Dynamics:
        x_{t+1} = x_t - v_t
        S_{t+1} = S_t + σ√Δt ε_t - γv_t
    
    Cost:
        E[C] = Σ v_t S_t + η Σ v_t^2
        Var[C] = σ^2 Σ (Σ_{s≥t} v_s)^2
    
    Objective:
        min E[C] + λ Var[C]
    """
    
    def __init__(self,
                 initial_inventory: float,
                 num_steps: int,
                 volatility: float,
                 eta: float,
                 gamma: float = 0.0,
                 risk_aversion: float = 0.0,
                 dt: float = 1.0):
        """
        Args:
            initial_inventory: Initial position X_0
            num_steps: Number of trading periods N
            volatility: Price volatility σ
            eta: Temporary impact coefficient
            gamma: Permanent impact coefficient
            risk_aversion: Risk aversion parameter λ
            dt: Time step size
        """
        self.X0 = initial_inventory
        self.N = num_steps
        self.sigma = volatility
        self.eta = eta
        self.gamma = gamma
        self.lam = risk_aversion
        self.dt = dt
        
        # Derived parameters
        self._compute_optimal_trajectory()
    
    def _compute_optimal_trajectory(self):
        """
        Derive closed-form optimal execution trajectory.
        
        Risk-neutral (λ=0):
            Uniform liquidation: v_t = X_0 / N
        
        Risk-averse (λ>0):
            Exponential trajectory with parameter κ
        """
        if self.lam == 0:
            # Risk-neutral: TWAP
            self.trajectory = np.ones(self.N) * (self.X0 / self.N)
        else:
            # Risk-averse: exponential decay
            # κ = sqrt(λ σ^2 / η)
            kappa = np.sqrt(self.lam * (self.sigma ** 2) / self.eta)
            
            # Trajectory: x_t = X_0 * sinh(κ(T-t)) / sinh(κT)
            T = self.N * self.dt
            times = np.arange(self.N + 1) * self.dt
            
            if kappa * T < 1e-6:
                # Small κT: linear approximation
                inventory = self.X0 * (1 - times / T)
            else:
                inventory = self.X0 * np.sinh(kappa * (T - times)) / np.sinh(kappa * T)
            
            # Ensure terminal inventory is zero
            inventory[-1] = 0.0
            
            # Trade sizes: v_t = x_t - x_{t+1}
            self.trajectory = np.diff(inventory)
            self.trajectory = np.abs(self.trajectory)  # Ensure positive
            self.inventory_trajectory = inventory[:-1]
    
    def get_trades(self) -> np.ndarray:
        """Return optimal trade schedule."""
        return self.trajectory
    
    def get_inventory_trajectory(self) -> np.ndarray:
        """Return inventory over time."""
        if hasattr(self, 'inventory_trajectory'):
            return self.inventory_trajectory
        else:
            # Reconstruct from trades
            inventory = np.zeros(self.N + 1)
            inventory[0] = self.X0
            for t in range(self.N):
                inventory[t + 1] = inventory[t] - self.trajectory[t]
            return inventory[:-1]
    
    def expected_cost(self) -> float:
        """
        Compute expected execution cost.
        
        E[C] = Σ v_t S_t + η Σ v_t^2
        
        Assuming S_t ≈ S_0 (no drift), this simplifies to:
        E[C] ≈ S_0 X_0 + η Σ v_t^2
        """
        impact_cost = self.eta * np.sum(self.trajectory ** 2)
        return impact_cost
    
    def cost_variance(self) -> float:
        """
        Compute variance of execution cost.
        
        Var[C] = σ^2 Σ_t (Σ_{s≥t} v_s)^2 Δt
        """
        cumulative_trades = np.cumsum(self.trajectory[::-1])[::-1]
        variance = (self.sigma ** 2) * np.sum(cumulative_trades ** 2) * self.dt
        return variance
    
    def objective_value(self) -> float:
        """Compute objective: E[C] + λ Var[C]"""
        return self.expected_cost() + self.lam * self.cost_variance()


class AlmgrenChrissQuadratic:
    """
    Almgren-Chriss with quadratic (power-law) temporary impact.
    
    Impact: η |v|^(1+φ)
    
    No closed-form solution; uses numerical optimization via gradient descent.
    """
    
    def __init__(self,
                 initial_inventory: float,
                 num_steps: int,
                 volatility: float,
                 eta: float,
                 phi: float = 0.5,
                 risk_aversion: float = 0.0,
                 dt: float = 1.0):
        """
        Args:
            initial_inventory: Initial position X_0
            num_steps: Number of trading periods N
            volatility: Price volatility σ
            eta: Impact coefficient
            phi: Power law exponent (impact ~ |v|^(1+φ))
            risk_aversion: Risk aversion parameter λ
            dt: Time step size
        """
        self.X0 = initial_inventory
        self.N = num_steps
        self.sigma = volatility
        self.eta = eta
        self.phi = phi
        self.lam = risk_aversion
        self.dt = dt
        
        # Initialize with TWAP
        self.trajectory = np.ones(self.N) * (self.X0 / self.N)
        
        # Optimize
        self._optimize_trajectory()
    
    def _optimize_trajectory(self, max_iter: int = 100):
        """
        Numerical optimization via projected gradient descent.
        
        Constraints:
            v_t ≥ 0
            Σ v_t = X_0
        """
        learning_rate = 0.01
        
        for _ in range(max_iter):
            # Compute gradient
            grad = self._compute_gradient()
            
            # Gradient step
            self.trajectory -= learning_rate * grad
            
            # Project onto constraints
            self.trajectory = np.maximum(self.trajectory, 0)
            self.trajectory *= self.X0 / np.sum(self.trajectory)
    
    def _compute_gradient(self) -> np.ndarray:
        """Compute gradient of objective w.r.t. trade schedule."""
        grad = np.zeros(self.N)
        
        # Impact cost gradient
        for t in range(self.N):
            grad[t] += self.eta * (1 + self.phi) * (self.trajectory[t] ** self.phi)
        
        # Variance gradient
        if self.lam > 0:
            cumulative_trades = np.cumsum(self.trajectory[::-1])[::-1]
            for t in range(self.N):
                # Contribution to variance from all future trades
                grad[t] += 2 * self.lam * (self.sigma ** 2) * np.sum(cumulative_trades[t:]) * self.dt
        
        return grad
    
    def get_trades(self) -> np.ndarray:
        """Return optimal trade schedule."""
        return self.trajectory
    
    def expected_cost(self) -> float:
        """Expected cost with power-law impact."""
        impact_cost = self.eta * np.sum(self.trajectory ** (1 + self.phi))
        return impact_cost
    
    def cost_variance(self) -> float:
        """Variance of execution cost."""
        cumulative_trades = np.cumsum(self.trajectory[::-1])[::-1]
        variance = (self.sigma ** 2) * np.sum(cumulative_trades ** 2) * self.dt
        return variance
    
    def objective_value(self) -> float:
        """Objective: E[C] + λ Var[C]"""
        return self.expected_cost() + self.lam * self.cost_variance()


def derive_ac_solution(risk_aversion: float, 
                       volatility: float, 
                       eta: float, 
                       horizon: float) -> str:
    """
    Derive and explain the Almgren-Chriss solution.
    
    Returns:
        Mathematical derivation as string
    """
    derivation = f"""
    ALMGREN-CHRISS OPTIMAL EXECUTION DERIVATION
    ============================================
    
    SETUP:
    ------
    Inventory: x_{{t+1}} = x_t - v_t
    Price: S_{{t+1}} = S_t + σ√Δt ε_t - γv_t
    
    Cost: C = Σ v_t S_t + η Σ v_t^2
    
    Objective: min E[C] + λ Var[C]
    
    SOLUTION:
    ---------
    Risk-neutral (λ=0):
        Optimal: v_t = X_0 / N  (TWAP)
        Intuition: No risk penalty, minimize impact only
    
    Risk-averse (λ>0):
        Define: κ = √(λσ²/η)
        
        Inventory trajectory:
            x_t = X_0 sinh(κ(T-t)) / sinh(κT)
        
        Trade schedule:
            v_t = x_t - x_{{t+1}}
        
        Intuition: 
            - High κ → aggressive early trading (reduce risk)
            - Low κ → uniform trading (reduce impact)
    
    PARAMETERS (current):
    ---------------------
    λ = {risk_aversion}
    σ = {volatility}
    η = {eta}
    T = {horizon}
    κ = {np.sqrt(risk_aversion * volatility**2 / eta):.4f}
    
    TRADE-OFF:
    ----------
    - Early trading: reduces variance, increases impact
    - Late trading: reduces impact, increases variance
    - Optimal balance depends on λ
    
    LIMITATIONS:
    ------------
    1. Assumes deterministic liquidity (η constant)
    2. No impact decay
    3. Linear impact model
    4. No regime changes
    5. Continuous trading (no discrete constraints)
    
    WHY AC FAILS IN PRACTICE:
    -------------------------
    - Liquidity varies stochastically
    - Impact decays over time
    - Market regimes shift
    - Execution constraints bind
    
    → Need adaptive policies (RL)
    """
    
    return derivation
