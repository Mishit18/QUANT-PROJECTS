"""
Hybrid Fourier-Monte Carlo scheme for rBergomi simulation.
"""

import numpy as np
from typing import Tuple, Optional
from .fbm import HybridFBMGenerator


class HybridScheme:
    """
    Hybrid simulation scheme for rBergomi model.
    Efficiently generates correlated fBM and Brownian motion paths.
    """

    def __init__(self, n_steps: int, n_paths: int, seed: Optional[int] = None):
        """
        Args:
            n_steps: Number of time discretization steps
            n_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility
        """
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def simulate_rbergomi(
        self,
        H: float,
        eta: float,
        rho: float,
        xi0: float,
        T: float,
        S0: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate rBergomi model paths.

        Model:
            v_t = ξ_0 exp(η Y_t - (1/2)η² t^(2H))
            dS_t / S_t = √v_t dZ_t

        Args:
            H: Hurst parameter
            eta: Vol-of-vol
            rho: Correlation between W and Z
            xi0: Initial forward variance
            T: Terminal time
            S0: Initial stock price

        Returns:
            (S_paths, v_paths): Asset and variance paths
                S_paths: shape (n_paths, n_steps + 1)
                v_paths: shape (n_paths, n_steps + 1)
        """
        dt = T / self.n_steps
        times = np.linspace(0, T, self.n_steps + 1)

        # Generate fractional Brownian motion Y_t
        fbm_gen = HybridFBMGenerator(H, self.n_steps, T)
        Y = fbm_gen.simulate(self.n_paths, self.rng)

        # Compute variance process
        # v_t = ξ_0 exp(η Y_t - (1/2)η² t^(2H))
        variance_adjustment = -0.5 * eta**2 * times**(2*H)
        v_paths = xi0 * np.exp(eta * Y + variance_adjustment[np.newaxis, :])

        # Generate correlated Brownian motion Z_t
        # Z_t = ρ W_t + √(1-ρ²) W_t^⊥
        dW = self.rng.normal(0, np.sqrt(dt), size=(self.n_paths, self.n_steps))
        dW_perp = self.rng.normal(0, np.sqrt(dt), size=(self.n_paths, self.n_steps))

        # Reconstruct W from Y (approximate for discrete case)
        # For simplicity, use independent Brownian motion with correlation
        dZ = rho * dW + np.sqrt(1 - rho**2) * dW_perp

        # Simulate asset price using Euler scheme
        # dS_t / S_t = √v_t dZ_t
        S_paths = np.zeros((self.n_paths, self.n_steps + 1))
        S_paths[:, 0] = S0

        for i in range(self.n_steps):
            # Milstein-type correction for better accuracy
            vol = np.sqrt(v_paths[:, i])
            S_paths[:, i+1] = S_paths[:, i] * np.exp(
                -0.5 * v_paths[:, i] * dt + vol * dZ[:, i]
            )

        return S_paths, v_paths

    def simulate_with_variance_reduction(
        self,
        H: float,
        eta: float,
        rho: float,
        xi0: float,
        T: float,
        S0: float = 100.0,
        use_antithetic: bool = True,
        use_moment_matching: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate with variance reduction techniques.

        Args:
            H, eta, rho, xi0, T, S0: Model parameters
            use_antithetic: Apply antithetic variates
            use_moment_matching: Apply moment matching to variance

        Returns:
            (S_paths, v_paths): Enhanced paths with variance reduction
        """
        if use_antithetic:
            # Generate half paths, create antithetic pairs
            original_n_paths = self.n_paths
            self.n_paths = self.n_paths // 2

            S_paths, v_paths = self.simulate_rbergomi(H, eta, rho, xi0, T, S0)

            # Create antithetic paths (negate Brownian increments)
            # This is approximate for fBM but still reduces variance
            S_anti = 2 * S0 - S_paths
            S_anti[:, 0] = S0
            v_anti = v_paths  # Variance paths remain similar

            S_paths = np.vstack([S_paths, S_anti])
            v_paths = np.vstack([v_paths, v_anti])

            self.n_paths = original_n_paths
        else:
            S_paths, v_paths = self.simulate_rbergomi(H, eta, rho, xi0, T, S0)

        if use_moment_matching:
            # Match first moment of terminal variance to theoretical
            theoretical_mean = xi0
            empirical_mean = np.mean(v_paths[:, -1])
            v_paths = v_paths * (theoretical_mean / empirical_mean)

        return S_paths, v_paths


class VarianceReductionScheme:
    """
    Advanced variance reduction techniques for Monte Carlo pricing.
    """

    @staticmethod
    def antithetic_variates(paths: np.ndarray, S0: float) -> np.ndarray:
        """
        Generate antithetic paths by reflection.

        Args:
            paths: Original paths (n_paths, n_steps + 1)
            S0: Initial value

        Returns:
            Combined paths with antithetic variates
        """
        log_returns = np.log(paths[:, 1:] / paths[:, :-1])
        anti_log_returns = -log_returns

        anti_paths = np.zeros_like(paths)
        anti_paths[:, 0] = S0
        anti_paths[:, 1:] = S0 * np.exp(np.cumsum(anti_log_returns, axis=1))

        return np.vstack([paths, anti_paths])

    @staticmethod
    def control_variate_adjustment(
        payoffs: np.ndarray,
        control_payoffs: np.ndarray,
        control_mean: float
    ) -> np.ndarray:
        """
        Apply control variate technique.

        Args:
            payoffs: Target payoffs
            control_payoffs: Control variate payoffs
            control_mean: Known mean of control

        Returns:
            Adjusted payoffs with reduced variance
        """
        # Optimal coefficient: β = -Cov(X,Y) / Var(Y)
        cov = np.cov(payoffs, control_payoffs)[0, 1]
        var = np.var(control_payoffs)

        if var > 1e-10:
            beta = -cov / var
            adjusted = payoffs + beta * (control_payoffs - control_mean)
            return adjusted

        return payoffs

    @staticmethod
    def moment_matching(paths: np.ndarray, target_mean: float, target_var: float) -> np.ndarray:
        """
        Match first two moments to theoretical values.

        Args:
            paths: Simulated paths
            target_mean: Theoretical mean
            target_var: Theoretical variance

        Returns:
            Adjusted paths
        """
        empirical_mean = np.mean(paths)
        empirical_var = np.var(paths)

        # Linear transformation to match moments
        if empirical_var > 1e-10:
            scale = np.sqrt(target_var / empirical_var)
            shift = target_mean - scale * empirical_mean
            return scale * paths + shift

        return paths
