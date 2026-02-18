"""
rBergomi rough volatility model implementation.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


@dataclass
class RBergomiParameters:
    """Parameters for rBergomi model."""
    H: float          # Hurst exponent (0 < H < 0.5)
    eta: float        # Vol-of-vol
    rho: float        # Correlation
    xi0: float        # Initial forward variance

    def __post_init__(self):
        """Validate parameters."""
        if not 0 < self.H < 0.5:
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if not -1 <= self.rho <= 1:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")
        if self.xi0 <= 0:
            raise ValueError(f"xi0 must be positive, got {self.xi0}")


class RBergomiModel:
    """
    Rough Bergomi volatility model.

    Dynamics:
        v_t = ξ_0(t) exp(η Y_t - (1/2)η² t^(2H))
        dS_t / S_t = √v_t dZ_t
        Y_t = ∫₀ᵗ (t-s)^(H-1/2) dW_s

    where W and Z are correlated Brownian motions with correlation ρ.
    """

    def __init__(
        self,
        H: float,
        eta: float,
        rho: float,
        xi0: float,
        forward_variance_curve: Optional[Callable[[float], float]] = None
    ):
        """
        Args:
            H: Hurst exponent
            eta: Vol-of-vol parameter
            rho: Correlation between driving Brownians
            xi0: Initial forward variance (or spot variance if curve is None)
            forward_variance_curve: Function t -> ξ_0(t), defaults to constant
        """
        self.params = RBergomiParameters(H, eta, rho, xi0)

        if forward_variance_curve is None:
            self.forward_variance_curve = lambda t: xi0
        else:
            self.forward_variance_curve = forward_variance_curve

    @property
    def H(self) -> float:
        return self.params.H

    @property
    def eta(self) -> float:
        return self.params.eta

    @property
    def rho(self) -> float:
        return self.params.rho

    @property
    def xi0(self) -> float:
        return self.params.xi0

    def variance_process(self, Y: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Compute variance process from fBM path.

        v_t = ξ_0(t) exp(η Y_t - (1/2)η² t^(2H))

        Args:
            Y: Fractional Brownian motion path (n_paths, n_steps + 1)
            times: Time grid

        Returns:
            Variance process (n_paths, n_steps + 1)
        """
        # Forward variance curve evaluation
        xi_t = np.array([self.forward_variance_curve(t) for t in times])

        # Variance adjustment term
        adjustment = -0.5 * self.eta**2 * times**(2 * self.H)

        # Compute variance
        v = xi_t[np.newaxis, :] * np.exp(self.eta * Y + adjustment[np.newaxis, :])

        return v

    def characteristic_function(self, u: complex, t: float, v0: float) -> complex:
        """
        Characteristic function (not available in closed form for rBergomi).
        This is a placeholder for potential approximations.

        Args:
            u: Frequency parameter
            t: Time to maturity
            v0: Initial variance

        Returns:
            Approximate characteristic function value
        """
        raise NotImplementedError(
            "rBergomi does not have a closed-form characteristic function. "
            "Use Monte Carlo pricing instead."
        )

    def implied_volatility_approximation(
        self,
        K: float,
        T: float,
        S0: float = 100.0,
        order: int = 1
    ) -> float:
        """
        First-order asymptotic approximation for implied volatility.
        Valid for small T (short maturities).

        Based on Bayer-Friz-Gatheral (2016) asymptotics.

        Args:
            K: Strike price
            T: Time to maturity
            S0: Spot price
            order: Approximation order (1 or 2)

        Returns:
            Approximate implied volatility
        """
        log_moneyness = np.log(K / S0)

        # Leading order term
        sigma_0 = np.sqrt(self.xi0)

        if order == 1:
            # First-order approximation
            skew_term = self.rho * self.eta * sigma_0 * T**(self.H - 0.5)
            iv = sigma_0 + skew_term * log_moneyness / sigma_0
            return max(iv, 1e-4)

        # Higher-order terms can be added here
        return sigma_0

    def get_parameter_dict(self) -> dict:
        """Return parameters as dictionary."""
        return {
            'H': self.H,
            'eta': self.eta,
            'rho': self.rho,
            'xi0': self.xi0
        }

    def __repr__(self) -> str:
        return (
            f"RBergomiModel(H={self.H:.3f}, eta={self.eta:.3f}, "
            f"rho={self.rho:.3f}, xi0={self.xi0:.4f})"
        )


class RBergomiWithJumps(RBergomiModel):
    """
    Extended rBergomi model with jumps in the asset price.

    dS_t / S_t = √v_t dZ_t + dJ_t

    where J_t is a compound Poisson process.
    """

    def __init__(
        self,
        H: float,
        eta: float,
        rho: float,
        xi0: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
        forward_variance_curve: Optional[Callable[[float], float]] = None
    ):
        """
        Args:
            H, eta, rho, xi0: rBergomi parameters
            jump_intensity: Poisson intensity (λ)
            jump_mean: Mean jump size (μ_J)
            jump_std: Jump size volatility (σ_J)
            forward_variance_curve: Forward variance curve
        """
        super().__init__(H, eta, rho, xi0, forward_variance_curve)

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def __repr__(self) -> str:
        base = super().__repr__()
        return base[:-1] + f", λ={self.jump_intensity:.3f})"
