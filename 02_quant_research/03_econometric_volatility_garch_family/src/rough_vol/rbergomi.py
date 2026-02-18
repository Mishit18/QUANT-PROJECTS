"""
Rough Bergomi (rBergomi) model.

Bayer, Friz, Gatheral (2016): "Pricing under rough volatility"

Model specification:
    dS_t / S_t = sqrt(V_t) dW_t
    V_t = xi_0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})

where:
- W^H is fractional Brownian motion with Hurst parameter H < 0.5
- xi_0 is initial variance
- eta is volatility of volatility
- H controls roughness (typically H ≈ 0.1)

Key features:
- Rough volatility paths (H < 0.5)
- Fast mean reversion
- Realistic autocorrelation structure
"""

import numpy as np
from typing import Optional, Tuple, Union
from .fractional_brownian import FractionalBrownianMotion


class RoughBergomiModel:
    """Simulate rough Bergomi volatility and returns."""
    
    def __init__(
        self,
        hurst: float = 0.1,
        eta: float = 1.9,
        xi0: float = 0.04,
        rho: float = -0.7,
        seed: Optional[int] = None
    ):
        """
        Initialize rBergomi model.
        
        Args:
            hurst: Hurst parameter (roughness), typically 0.05-0.15
            eta: Volatility of volatility
            xi0: Initial variance
            rho: Correlation between price and volatility shocks (leverage effect)
            seed: Random seed
        """
        self.hurst = hurst
        self.eta = eta
        self.xi0 = xi0
        self.rho = rho
        self.seed = seed
        
        self.fbm = FractionalBrownianMotion(hurst=hurst, seed=seed)
        self.rng = np.random.default_rng(seed)
    
    def simulate_variance(self, n_steps: int, dt: float = 1/252) -> np.ndarray:
        """
        Simulate variance process.
        
        V_t = xi_0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})
        
        Args:
            n_steps: Number of time steps
            dt: Time step size (default: 1 day in years)
        
        Returns:
            Variance path (length n_steps+1)
        """
        # Simulate fBm
        fbm_path = self.fbm.simulate(n_steps, dt)
        
        # Time grid
        times = np.arange(n_steps + 1) * dt
        
        # Variance process
        variance = self.xi0 * np.exp(
            self.eta * fbm_path - 0.5 * self.eta**2 * times**(2 * self.hurst)
        )
        
        return variance
    
    def simulate_returns(
        self,
        n_steps: int,
        dt: float = 1/252,
        return_variance: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate returns with rough volatility.
        
        dS_t / S_t = sqrt(V_t) dW_t
        
        Args:
            n_steps: Number of time steps
            dt: Time step size
            return_variance: If True, also return variance path
        
        Returns:
            Returns (and optionally variance path)
        """
        # Simulate correlated fBm for volatility and price
        fbm_vol = self.fbm.simulate(n_steps, dt)
        
        # Generate correlated Brownian motion for price
        # W_price = rho * W_vol + sqrt(1 - rho^2) * W_indep
        z_indep = self.rng.standard_normal(n_steps)
        
        # Approximate fBm increments for correlation
        fbm_increments = np.diff(fbm_vol)
        
        # Correlated increments
        w_price_increments = (
            self.rho * fbm_increments +
            np.sqrt(1 - self.rho**2) * z_indep * np.sqrt(dt)
        )
        
        # Variance process
        times = np.arange(n_steps + 1) * dt
        variance = self.xi0 * np.exp(
            self.eta * fbm_vol - 0.5 * self.eta**2 * times**(2 * self.hurst)
        )
        
        # Returns: r_t = sqrt(V_t) * dW_t
        volatility = np.sqrt(variance[:-1])
        returns = volatility * w_price_increments
        
        if return_variance:
            return returns, variance
        else:
            return returns
    
    def simulate_multiple_paths(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1/252
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple return and variance paths.
        
        Args:
            n_paths: Number of paths
            n_steps: Number of time steps per path
            dt: Time step size
        
        Returns:
            (returns_array, variance_array) of shape (n_paths, n_steps) and (n_paths, n_steps+1)
        """
        returns_array = np.zeros((n_paths, n_steps))
        variance_array = np.zeros((n_paths, n_steps + 1))
        
        for i in range(n_paths):
            returns, variance = self.simulate_returns(n_steps, dt, return_variance=True)
            returns_array[i] = returns
            variance_array[i] = variance
        
        return returns_array, variance_array
    
    def impulse_response(self, max_lag: int = 50, dt: float = 1/252) -> np.ndarray:
        """
        Compute impulse response function of volatility to shocks.
        
        For rough volatility, response decays as t^{H-1/2}.
        
        Args:
            max_lag: Maximum lag to compute
            dt: Time step size
        
        Returns:
            Impulse response at each lag
        """
        lags = np.arange(1, max_lag + 1) * dt
        
        # Rough volatility impulse response: proportional to t^{H-1/2}
        response = lags**(self.hurst - 0.5)
        
        # Normalize
        response = response / response[0]
        
        return response
    
    def theoretical_autocorrelation(self, max_lag: int = 50) -> np.ndarray:
        """
        Theoretical autocorrelation of squared returns.
        
        For rough volatility: rho(k) ~ k^{2H-1}
        """
        lags = np.arange(max_lag + 1)
        lags[0] = 1  # Avoid division by zero
        
        acf = lags**(2 * self.hurst - 1)
        acf[0] = 1.0
        
        return acf
