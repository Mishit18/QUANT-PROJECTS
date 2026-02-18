"""
Fractional Brownian Motion (fBm) simulation.

fBm is a generalization of Brownian motion with long-range dependence.
Hurst parameter H controls roughness:
- H = 0.5: Standard Brownian motion
- H < 0.5: Rough (anti-persistent)
- H > 0.5: Smooth (persistent)

Rough volatility models use H < 0.5 (typically H ≈ 0.1).
"""

import numpy as np
from scipy.linalg import cholesky
from typing import Optional


class FractionalBrownianMotion:
    """Simulate fractional Brownian motion paths."""
    
    def __init__(self, hurst: float = 0.1, seed: Optional[int] = None):
        """
        Initialize fBm simulator.
        
        Args:
            hurst: Hurst parameter (0 < H < 1)
                   H < 0.5 for rough paths
            seed: Random seed for reproducibility
        """
        if not 0 < hurst < 1:
            raise ValueError("Hurst parameter must be in (0, 1)")
        
        self.hurst = hurst
        self.rng = np.random.default_rng(seed)
    
    def _covariance_matrix(self, n: int, dt: float) -> np.ndarray:
        """
        Compute covariance matrix for fBm increments.
        
        Cov(B_H(s), B_H(t)) = 0.5 * (|s|^{2H} + |t|^{2H} - |t-s|^{2H})
        """
        H = self.hurst
        times = np.arange(1, n + 1) * dt
        
        # Covariance matrix
        cov = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                s = times[i]
                t = times[j]
                cov[i, j] = 0.5 * (s**(2*H) + t**(2*H) - np.abs(t - s)**(2*H))
        
        return cov
    
    def simulate_cholesky(self, n: int, dt: float = 1.0) -> np.ndarray:
        """
        Simulate fBm using Cholesky decomposition.
        
        Exact method but O(n^3) complexity.
        Use for small n (< 1000).
        
        Args:
            n: Number of time steps
            dt: Time step size
        
        Returns:
            fBm path (length n+1, starting at 0)
        """
        # Covariance matrix
        cov = self._covariance_matrix(n, dt)
        
        # Cholesky decomposition
        L = cholesky(cov, lower=True)
        
        # Generate standard normal increments
        z = self.rng.standard_normal(n)
        
        # fBm increments
        increments = L @ z
        
        # Cumulative sum to get path
        path = np.concatenate([[0], np.cumsum(increments)])
        
        return path
    
    def simulate_davies_harte(self, n: int, dt: float = 1.0) -> np.ndarray:
        """
        Simulate fBm using Davies-Harte method (FFT-based).
        
        Faster than Cholesky: O(n log n) complexity.
        Use for large n.
        
        Args:
            n: Number of time steps
            dt: Time step size
        
        Returns:
            fBm path (length n+1, starting at 0)
        """
        H = self.hurst
        
        # Autocovariance function
        def gamma(k):
            if k == 0:
                return 1.0
            return 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + (k-1)**(2*H))
        
        # First row of circulant matrix
        r = np.array([gamma(k) for k in range(n)])
        r = np.concatenate([r, r[-2:0:-1]])
        
        # Eigenvalues via FFT
        eigenvalues = np.fft.fft(r).real
        
        # Check for numerical issues
        if np.any(eigenvalues < -1e-10):
            # Fall back to Cholesky if Davies-Harte fails
            return self.simulate_cholesky(n, dt)
        
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Generate complex Gaussian
        z1 = self.rng.standard_normal(len(eigenvalues))
        z2 = self.rng.standard_normal(len(eigenvalues))
        z = (z1 + 1j * z2) / np.sqrt(2)
        
        # Apply FFT
        w = np.sqrt(eigenvalues) * z
        increments = np.fft.fft(w).real[:n] / np.sqrt(2 * len(eigenvalues))
        
        # Scale by dt
        increments = increments * np.sqrt(dt)
        
        # Cumulative sum to get path
        path = np.concatenate([[0], np.cumsum(increments)])
        
        return path
    
    def simulate(self, n: int, dt: float = 1.0, method: str = "auto") -> np.ndarray:
        """
        Simulate fBm path.
        
        Args:
            n: Number of time steps
            dt: Time step size
            method: "cholesky", "davies_harte", or "auto"
        
        Returns:
            fBm path
        """
        if method == "auto":
            method = "davies_harte" if n > 500 else "cholesky"
        
        if method == "cholesky":
            return self.simulate_cholesky(n, dt)
        elif method == "davies_harte":
            return self.simulate_davies_harte(n, dt)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def simulate_multiple(
        self,
        n_paths: int,
        n_steps: int,
        dt: float = 1.0,
        method: str = "auto"
    ) -> np.ndarray:
        """
        Simulate multiple fBm paths.
        
        Args:
            n_paths: Number of paths
            n_steps: Number of time steps per path
            dt: Time step size
            method: Simulation method
        
        Returns:
            Array of shape (n_paths, n_steps+1)
        """
        paths = np.zeros((n_paths, n_steps + 1))
        
        for i in range(n_paths):
            paths[i] = self.simulate(n_steps, dt, method)
        
        return paths
    
    def autocorrelation(self, lags: int = 20) -> np.ndarray:
        """
        Theoretical autocorrelation function of fBm increments.
        
        rho(k) = 0.5 * ((k+1)^{2H} - 2k^{2H} + (k-1)^{2H})
        """
        H = self.hurst
        acf = np.zeros(lags + 1)
        acf[0] = 1.0
        
        for k in range(1, lags + 1):
            acf[k] = 0.5 * ((k+1)**(2*H) - 2*k**(2*H) + (k-1)**(2*H))
        
        return acf
