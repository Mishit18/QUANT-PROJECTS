"""
Fractional Brownian motion simulation using Volterra kernel representation.
Implements hybrid scheme with FFT acceleration.
"""

import numpy as np
from scipy.linalg import cholesky
from numba import jit


class FractionalBrownianMotion:
    """
    Simulate fractional Brownian motion with Hurst parameter H < 0.5.
    Uses Volterra kernel representation: Y_t = ∫₀ᵗ K(t,s) dW_s
    """

    def __init__(self, H: float, n_steps: int, T: float):
        """
        Args:
            H: Hurst exponent (0 < H < 0.5 for rough paths)
            n_steps: Number of time steps
            T: Terminal time
        """
        if not 0 < H < 0.5:
            raise ValueError("Hurst parameter must be in (0, 0.5) for rough volatility")

        self.H = H
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps
        self.times = np.linspace(0, T, n_steps + 1)

        # Precompute Volterra kernel for efficiency
        self._kernel = self._compute_volterra_kernel()

    def _compute_volterra_kernel(self) -> np.ndarray:
        """
        Compute Volterra kernel: K(t_i, t_j) = (t_i - t_j)^(H - 1/2)
        Returns lower triangular matrix for convolution.
        """
        kernel = np.zeros((self.n_steps + 1, self.n_steps + 1))

        for i in range(1, self.n_steps + 1):
            for j in range(i):
                kernel[i, j] = (self.times[i] - self.times[j]) ** (self.H - 0.5)

        return kernel

    def simulate(self, n_paths: int, rng: np.random.Generator = None) -> np.ndarray:
        """
        Simulate fBM paths using Volterra representation.

        Args:
            n_paths: Number of paths to simulate
            rng: Random number generator

        Returns:
            Array of shape (n_paths, n_steps + 1) containing fBM paths
        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate standard Brownian increments
        dW = rng.normal(0, np.sqrt(self.dt), size=(n_paths, self.n_steps))

        # Apply Volterra kernel via convolution
        Y = np.zeros((n_paths, self.n_steps + 1))

        for i in range(1, self.n_steps + 1):
            # Convolution: Y_i = Σ_j K(t_i, t_j) * dW_j
            Y[:, i] = Y[:, i-1] + self._kernel[i, i-1] * dW[:, i-1]

        return Y

    def simulate_hybrid(self, n_paths: int, rng: np.random.Generator = None) -> np.ndarray:
        """
        Hybrid scheme using FFT for long-range correlations.
        More efficient for large n_steps.

        Args:
            n_paths: Number of paths
            rng: Random number generator

        Returns:
            Array of shape (n_paths, n_steps + 1)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate Gaussian increments
        dW = rng.normal(0, np.sqrt(self.dt), size=(n_paths, self.n_steps))

        # Use FFT-based convolution for efficiency
        Y = np.zeros((n_paths, self.n_steps + 1))
        kernel_vec = self._kernel[1:, 0]

        for path_idx in range(n_paths):
            # FFT convolution
            Y[path_idx, 1:] = np.convolve(dW[path_idx], kernel_vec, mode='full')[:self.n_steps]

        return Y


class HybridFBMGenerator:
    """
    Advanced hybrid scheme combining exact covariance and FFT.
    Optimized for rough volatility simulation.
    """

    def __init__(self, H: float, n_steps: int, T: float, cutoff: int = 50):
        """
        Args:
            H: Hurst parameter
            n_steps: Time steps
            T: Terminal time
            cutoff: Transition point between exact and FFT methods
        """
        self.H = H
        self.n_steps = n_steps
        self.T = T
        self.dt = T / n_steps
        self.cutoff = min(cutoff, n_steps)

        # Precompute covariance for short lags
        self._cov_matrix = self._compute_covariance_matrix()
        self._chol = cholesky(self._cov_matrix, lower=True)

    def _compute_covariance_matrix(self) -> np.ndarray:
        """
        Exact covariance: Cov(Y_s, Y_t) = (1/2H)[s^(2H) + t^(2H) - |t-s|^(2H)]
        """
        times = np.linspace(0, self.cutoff * self.dt, self.cutoff + 1)
        cov = np.zeros((self.cutoff + 1, self.cutoff + 1))

        for i in range(self.cutoff + 1):
            for j in range(i + 1):
                s, t = times[j], times[i]
                cov[i, j] = 0.5 * (s**(2*self.H) + t**(2*self.H) - abs(t - s)**(2*self.H))
                cov[j, i] = cov[i, j]

        # Ensure positive definiteness
        cov += 1e-10 * np.eye(self.cutoff + 1)
        return cov

    def simulate(self, n_paths: int, rng: np.random.Generator = None) -> np.ndarray:
        """
        Hybrid simulation: exact for short lags, FFT for long lags.

        Args:
            n_paths: Number of paths
            rng: Random generator

        Returns:
            fBM paths of shape (n_paths, n_steps + 1)
        """
        if rng is None:
            rng = np.random.default_rng()

        Y = np.zeros((n_paths, self.n_steps + 1))

        # Short lags: exact covariance via Cholesky
        Z_short = rng.standard_normal(size=(n_paths, self.cutoff + 1))
        Y[:, :self.cutoff + 1] = (self._chol @ Z_short.T).T

        # Long lags: FFT-based extension
        if self.n_steps > self.cutoff:
            kernel = np.array([(i * self.dt)**(self.H - 0.5) for i in range(1, self.n_steps - self.cutoff + 1)])
            dW = rng.normal(0, np.sqrt(self.dt), size=(n_paths, self.n_steps - self.cutoff))

            for path_idx in range(n_paths):
                increments = np.convolve(dW[path_idx], kernel, mode='full')[:self.n_steps - self.cutoff]
                Y[path_idx, self.cutoff + 1:] = Y[path_idx, self.cutoff] + np.cumsum(increments)

        return Y


@jit(nopython=True)
def volterra_kernel(t: float, s: float, H: float) -> float:
    """Volterra kernel K(t,s) = (t-s)^(H-1/2) for t > s."""
    if t <= s:
        return 0.0
    return (t - s) ** (H - 0.5)


@jit(nopython=True)
def fbm_covariance(s: float, t: float, H: float) -> float:
    """Exact covariance function for fBM."""
    return 0.5 * (s**(2*H) + t**(2*H) - abs(t - s)**(2*H))
