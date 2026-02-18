"""
SABR (Stochastic Alpha Beta Rho) model for comparison.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.stats import norm


@dataclass
class SABRParameters:
    """Parameters for SABR model."""
    alpha: float      # Initial volatility
    beta: float       # CEV exponent (0 ≤ β ≤ 1)
    rho: float        # Correlation
    nu: float         # Vol-of-vol

    def __post_init__(self):
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0 <= self.beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.nu < 0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")


class SABRModel:
    """
    SABR stochastic volatility model.

    Dynamics:
        dF_t = σ_t F_t^β dZ_t
        dσ_t = ν σ_t dW_t

    where W and Z are correlated with correlation ρ.
    Commonly used for interest rate derivatives and FX options.
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        rho: float,
        nu: float
    ):
        """
        Args:
            alpha: Initial volatility level
            beta: CEV exponent (0=normal, 1=lognormal)
            rho: Correlation between forward and volatility
            nu: Vol-of-vol
        """
        self.params = SABRParameters(alpha, beta, rho, nu)

    @property
    def alpha(self) -> float:
        return self.params.alpha

    @property
    def beta(self) -> float:
        return self.params.beta

    @property
    def rho(self) -> float:
        return self.params.rho

    @property
    def nu(self) -> float:
        return self.params.nu

    def implied_volatility_hagan(
        self,
        F: float,
        K: float,
        T: float
    ) -> float:
        """
        Hagan's approximation for SABR implied volatility.

        Valid for small T and K near F.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity

        Returns:
            Black implied volatility
        """
        if abs(F - K) < 1e-10:
            # ATM case
            F_mid = F
            sigma_atm = self.alpha / (F_mid ** (1 - self.beta)) * \
                       (1 + ((1 - self.beta)**2 / 24 * self.alpha**2 / F_mid**(2 - 2*self.beta) +
                             0.25 * self.rho * self.beta * self.nu * self.alpha / F_mid**(1 - self.beta) +
                             (2 - 3*self.rho**2) / 24 * self.nu**2) * T)
            return sigma_atm

        # General case
        log_moneyness = np.log(F / K)
        F_mid = (F * K) ** 0.5

        # z parameter
        z = (self.nu / self.alpha) * F_mid**(1 - self.beta) * log_moneyness

        # x(z) function
        if abs(z) < 1e-7:
            x_z = 1.0
        else:
            x_z = np.log((np.sqrt(1 - 2*self.rho*z + z**2) + z - self.rho) / (1 - self.rho))
            x_z = z / x_z

        # First term
        term1 = self.alpha / (F_mid**(1 - self.beta) *
                              (1 + (1 - self.beta)**2 / 24 * log_moneyness**2 +
                               (1 - self.beta)**4 / 1920 * log_moneyness**4))

        # Second term (time-dependent correction)
        term2 = 1 + ((1 - self.beta)**2 / 24 * self.alpha**2 / F_mid**(2 - 2*self.beta) +
                     0.25 * self.rho * self.beta * self.nu * self.alpha / F_mid**(1 - self.beta) +
                     (2 - 3*self.rho**2) / 24 * self.nu**2) * T

        return term1 * x_z * term2

    def simulate(
        self,
        F0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate SABR model paths using Euler scheme.

        Args:
            F0: Initial forward price
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths
            rng: Random number generator

        Returns:
            (F_paths, sigma_paths): Forward and volatility paths
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = T / n_steps

        F = np.zeros((n_paths, n_steps + 1))
        sigma = np.zeros((n_paths, n_steps + 1))

        F[:, 0] = F0
        sigma[:, 0] = self.alpha

        for i in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            dW = np.sqrt(dt) * Z1
            dZ = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)

            # Volatility process (ensure positivity)
            sigma[:, i+1] = sigma[:, i] * np.exp(
                -0.5 * self.nu**2 * dt + self.nu * dW
            )

            # Forward process
            F_beta = np.maximum(F[:, i], 1e-10) ** self.beta
            F[:, i+1] = F[:, i] + sigma[:, i] * F_beta * dZ
            F[:, i+1] = np.maximum(F[:, i+1], 1e-10)  # Ensure positivity

        return F, sigma

    def calibrate_to_smile(
        self,
        F: float,
        strikes: np.ndarray,
        market_vols: np.ndarray,
        T: float,
        beta: float = None
    ) -> 'SABRModel':
        """
        Calibrate SABR parameters to a volatility smile.

        Args:
            F: Forward price
            strikes: Array of strike prices
            market_vols: Market implied volatilities
            T: Time to maturity
            beta: Fixed beta (if None, calibrate)

        Returns:
            Calibrated SABR model
        """
        from scipy.optimize import minimize

        if beta is None:
            beta = self.beta

        def objective(params):
            alpha, rho, nu = params
            if alpha <= 0 or nu < 0 or abs(rho) > 1:
                return 1e10

            model = SABRModel(alpha, beta, rho, nu)
            model_vols = np.array([
                model.implied_volatility_hagan(F, K, T) for K in strikes
            ])

            return np.sum((model_vols - market_vols)**2)

        # Initial guess
        x0 = [self.alpha, self.rho, self.nu]

        # Bounds
        bounds = [(1e-4, 2.0), (-0.999, 0.999), (1e-4, 2.0)]

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        if result.success:
            alpha_opt, rho_opt, nu_opt = result.x
            return SABRModel(alpha_opt, beta, rho_opt, nu_opt)

        return self

    def get_parameter_dict(self) -> dict:
        """Return parameters as dictionary."""
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'rho': self.rho,
            'nu': self.nu
        }

    def __repr__(self) -> str:
        return (
            f"SABRModel(α={self.alpha:.4f}, β={self.beta:.2f}, "
            f"ρ={self.rho:.3f}, ν={self.nu:.3f})"
        )


def sabr_normal_volatility(
    F: float,
    K: float,
    T: float,
    alpha: float,
    rho: float,
    nu: float
) -> float:
    """
    SABR normal (β=0) implied volatility approximation.

    Args:
        F: Forward price
        K: Strike price
        T: Time to maturity
        alpha: Initial volatility
        rho: Correlation
        nu: Vol-of-vol

    Returns:
        Normal implied volatility
    """
    if abs(F - K) < 1e-10:
        return alpha * (1 + (2 - 3*rho**2) / 24 * nu**2 * T)

    z = (nu / alpha) * (F - K)
    x_z = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))

    if abs(x_z) < 1e-7:
        x_z = 1.0
    else:
        x_z = z / x_z

    return alpha * x_z * (1 + (2 - 3*rho**2) / 24 * nu**2 * T)
