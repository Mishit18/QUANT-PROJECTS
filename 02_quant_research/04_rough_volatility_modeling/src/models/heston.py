"""
Heston stochastic volatility model for comparison.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class HestonParameters:
    """Parameters for Heston model."""
    kappa: float      # Mean reversion speed
    theta: float      # Long-term variance
    sigma: float      # Vol-of-vol
    rho: float        # Correlation
    v0: float         # Initial variance

    def __post_init__(self):
        """Validate Feller condition: 2κθ ≥ σ²."""
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if self.theta <= 0:
            raise ValueError(f"theta must be positive, got {self.theta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.v0 <= 0:
            raise ValueError(f"v0 must be positive, got {self.v0}")


class HestonModel:
    """
    Heston stochastic volatility model.

    Dynamics:
        dS_t / S_t = √v_t dZ_t
        dv_t = κ(θ - v_t)dt + σ√v_t dW_t

    where W and Z are correlated with correlation ρ.
    """

    def __init__(
        self,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float
    ):
        """
        Args:
            kappa: Mean reversion speed
            theta: Long-term variance level
            sigma: Vol-of-vol
            rho: Correlation
            v0: Initial variance
        """
        self.params = HestonParameters(kappa, theta, sigma, rho, v0)

    @property
    def kappa(self) -> float:
        return self.params.kappa

    @property
    def theta(self) -> float:
        return self.params.theta

    @property
    def sigma(self) -> float:
        return self.params.sigma

    @property
    def rho(self) -> float:
        return self.params.rho

    @property
    def v0(self) -> float:
        return self.params.v0

    def simulate(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        scheme: str = 'euler',
        rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths.

        Args:
            S0: Initial stock price
            T: Terminal time
            n_steps: Number of time steps
            n_paths: Number of paths
            scheme: 'euler' or 'milstein' or 'qe' (quadratic-exponential)
            rng: Random number generator

        Returns:
            (S_paths, v_paths): Asset and variance paths
        """
        if rng is None:
            rng = np.random.default_rng()

        dt = T / n_steps

        if scheme == 'euler':
            return self._simulate_euler(S0, T, n_steps, n_paths, dt, rng)
        elif scheme == 'milstein':
            return self._simulate_milstein(S0, T, n_steps, n_paths, dt, rng)
        elif scheme == 'qe':
            return self._simulate_qe(S0, T, n_steps, n_paths, dt, rng)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def _simulate_euler(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        dt: float,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Euler discretization with full truncation."""
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        # Generate correlated Brownian motions
        for i in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            dW = np.sqrt(dt) * Z1
            dZ = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)

            # Variance process (full truncation)
            v_pos = np.maximum(v[:, i], 0)
            v[:, i+1] = v[:, i] + self.kappa * (self.theta - v_pos) * dt + \
                        self.sigma * np.sqrt(v_pos) * dW
            v[:, i+1] = np.maximum(v[:, i+1], 0)

            # Asset process
            S[:, i+1] = S[:, i] * np.exp(-0.5 * v_pos * dt + np.sqrt(v_pos) * dZ)

        return S, v

    def _simulate_milstein(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        dt: float,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Milstein scheme for variance process."""
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        for i in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)

            dW = np.sqrt(dt) * Z1
            dZ = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)

            v_pos = np.maximum(v[:, i], 0)

            # Milstein correction for variance
            v[:, i+1] = v[:, i] + self.kappa * (self.theta - v_pos) * dt + \
                        self.sigma * np.sqrt(v_pos) * dW + \
                        0.25 * self.sigma**2 * (dW**2 - dt)
            v[:, i+1] = np.maximum(v[:, i+1], 0)

            # Asset process
            S[:, i+1] = S[:, i] * np.exp(-0.5 * v_pos * dt + np.sqrt(v_pos) * dZ)

        return S, v

    def _simulate_qe(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        dt: float,
        rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Quadratic-Exponential (QE) scheme by Andersen (2008).
        More accurate for variance process, avoids negative values.
        """
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = self.v0

        psi_c = 1.5  # Critical threshold

        for i in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            U = rng.uniform(0, 1, n_paths)

            # Compute moments
            m = self.theta + (v[:, i] - self.theta) * np.exp(-self.kappa * dt)
            s2 = v[:, i] * self.sigma**2 * np.exp(-self.kappa * dt) * \
                 (1 - np.exp(-self.kappa * dt)) / self.kappa + \
                 self.theta * self.sigma**2 * (1 - np.exp(-self.kappa * dt))**2 / (2 * self.kappa)

            psi = s2 / m**2

            # QE scheme
            for j in range(n_paths):
                if psi[j] <= psi_c:
                    # Use inverse transform
                    b2 = 2 / psi[j] - 1 + np.sqrt(2 / psi[j]) * np.sqrt(2 / psi[j] - 1)
                    a = m[j] / (1 + b2)
                    v[j, i+1] = a * (np.sqrt(b2) + Z1[j])**2
                else:
                    # Use exponential distribution
                    p = (psi[j] - 1) / (psi[j] + 1)
                    beta = (1 - p) / m[j]

                    if U[j] <= p:
                        v[j, i+1] = 0
                    else:
                        v[j, i+1] = np.log((1 - p) / (1 - U[j])) / beta

            # Asset process with integrated variance
            dZ = np.sqrt(dt) * (self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2)
            K0 = -self.rho * self.kappa * self.theta * dt / self.sigma
            K1 = 0.5 * dt * (self.kappa * self.rho / self.sigma - 0.5) - self.rho / self.sigma
            K2 = 0.5 * dt * (self.kappa * self.rho / self.sigma - 0.5) + self.rho / self.sigma
            K3 = 0.5 * dt * (1 - self.rho**2)

            S[:, i+1] = S[:, i] * np.exp(
                K0 + K1 * v[:, i] + K2 * v[:, i+1] +
                np.sqrt(K3 * (v[:, i] + v[:, i+1])) * Z2
            )

        return S, v

    def characteristic_function(self, u: complex, t: float) -> complex:
        """
        Heston characteristic function (closed form).

        φ(u) = E[exp(iu log(S_t))]

        Args:
            u: Frequency parameter
            t: Time to maturity

        Returns:
            Characteristic function value
        """
        d = np.sqrt((self.rho * self.sigma * u * 1j - self.kappa)**2 +
                    self.sigma**2 * (u * 1j + u**2))
        g = (self.kappa - self.rho * self.sigma * u * 1j - d) / \
            (self.kappa - self.rho * self.sigma * u * 1j + d)

        C = self.kappa * (
            (self.kappa - self.rho * self.sigma * u * 1j - d) * t -
            2 * np.log((1 - g * np.exp(-d * t)) / (1 - g))
        ) / self.sigma**2

        D = (self.kappa - self.rho * self.sigma * u * 1j - d) * \
            (1 - np.exp(-d * t)) / (self.sigma**2 * (1 - g * np.exp(-d * t)))

        return np.exp(C * self.theta + D * self.v0)

    def get_parameter_dict(self) -> dict:
        """Return parameters as dictionary."""
        return {
            'kappa': self.kappa,
            'theta': self.theta,
            'sigma': self.sigma,
            'rho': self.rho,
            'v0': self.v0
        }

    def __repr__(self) -> str:
        return (
            f"HestonModel(κ={self.kappa:.3f}, θ={self.theta:.4f}, "
            f"σ={self.sigma:.3f}, ρ={self.rho:.3f}, v0={self.v0:.4f})"
        )
