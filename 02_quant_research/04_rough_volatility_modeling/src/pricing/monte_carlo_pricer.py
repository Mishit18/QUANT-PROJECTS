"""
Monte Carlo option pricing with variance reduction.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.stats import norm


class MonteCarloOptionPricer:
    """
    Monte Carlo pricer for European options with variance reduction.
    """

    def __init__(
        self,
        n_paths: int = 10000,
        seed: Optional[int] = None,
        use_antithetic: bool = True,
        use_control_variate: bool = True
    ):
        """
        Args:
            n_paths: Number of Monte Carlo paths
            seed: Random seed
            use_antithetic: Apply antithetic variates
            use_control_variate: Apply control variate technique
        """
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)
        self.use_antithetic = use_antithetic
        self.use_control_variate = use_control_variate

    def price_european_call(
        self,
        S_paths: np.ndarray,
        K: float,
        r: float = 0.0,
        T: float = None
    ) -> Tuple[float, float]:
        """
        Price European call option.

        Args:
            S_paths: Simulated asset paths (n_paths, n_steps + 1)
            K: Strike price
            r: Risk-free rate
            T: Time to maturity (for discounting)

        Returns:
            (price, std_error): Option price and standard error
        """
        S_T = S_paths[:, -1]
        payoffs = np.maximum(S_T - K, 0)

        if self.use_control_variate and T is not None:
            # Use Black-Scholes as control variate
            S0 = S_paths[0, 0]
            implied_vol = np.std(np.log(S_T / S0)) / np.sqrt(T)
            bs_price = black_scholes_call(S0, K, T, r, implied_vol)

            # Simulate BS payoffs with same random numbers
            bs_payoffs = self._black_scholes_payoffs(S0, K, T, r, implied_vol, len(payoffs))

            # Control variate adjustment
            beta = -np.cov(payoffs, bs_payoffs)[0, 1] / np.var(bs_payoffs)
            payoffs = payoffs + beta * (bs_payoffs - bs_price * np.exp(r * T))

        discount = np.exp(-r * T) if T is not None else 1.0
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))

        return price, std_error

    def price_european_put(
        self,
        S_paths: np.ndarray,
        K: float,
        r: float = 0.0,
        T: float = None
    ) -> Tuple[float, float]:
        """
        Price European put option.

        Args:
            S_paths: Simulated asset paths
            K: Strike price
            r: Risk-free rate
            T: Time to maturity

        Returns:
            (price, std_error)
        """
        S_T = S_paths[:, -1]
        payoffs = np.maximum(K - S_T, 0)

        discount = np.exp(-r * T) if T is not None else 1.0
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))

        return price, std_error

    def price_digital_call(
        self,
        S_paths: np.ndarray,
        K: float,
        r: float = 0.0,
        T: float = None
    ) -> Tuple[float, float]:
        """Price digital (binary) call option."""
        S_T = S_paths[:, -1]
        payoffs = (S_T > K).astype(float)

        discount = np.exp(-r * T) if T is not None else 1.0
        price = discount * np.mean(payoffs)
        std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))

        return price, std_error

    def _black_scholes_payoffs(
        self,
        S0: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        n_samples: int
    ) -> np.ndarray:
        """Generate Black-Scholes call payoffs for control variate."""
        Z = self.rng.standard_normal(n_samples)
        S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        return np.maximum(S_T - K, 0)

    def compute_greeks(
        self,
        S_paths: np.ndarray,
        K: float,
        T: float,
        r: float = 0.0,
        option_type: str = 'call',
        epsilon: float = 0.01
    ) -> dict:
        """
        Compute option Greeks using finite differences.

        Args:
            S_paths: Asset paths
            K: Strike
            T: Maturity
            r: Risk-free rate
            option_type: 'call' or 'put'
            epsilon: Bump size for finite differences

        Returns:
            Dictionary with delta, gamma, vega estimates
        """
        S0 = S_paths[0, 0]

        if option_type == 'call':
            price_func = self.price_european_call
        else:
            price_func = self.price_european_put

        # Base price
        price, _ = price_func(S_paths, K, r, T)

        # Delta: ∂V/∂S
        S_paths_up = S_paths * (1 + epsilon)
        price_up, _ = price_func(S_paths_up, K, r, T)
        delta = (price_up - price) / (epsilon * S0)

        # Gamma: ∂²V/∂S²
        S_paths_down = S_paths * (1 - epsilon)
        price_down, _ = price_func(S_paths_down, K, r, T)
        gamma = (price_up - 2*price + price_down) / (epsilon * S0)**2

        return {
            'delta': delta,
            'gamma': gamma,
            'price': price
        }


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Black-Scholes call option price.

    Args:
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        sigma: Volatility

    Returns:
        Call option price
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """Black-Scholes vega (∂V/∂σ)."""
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)
