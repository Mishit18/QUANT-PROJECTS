"""
Implied volatility calculation using root-finding methods.
"""

import numpy as np
from scipy.optimize import brentq, newton
from typing import Optional
from .monte_carlo_pricer import black_scholes_call, black_scholes_put, black_scholes_vega


def implied_volatility(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.0,
    option_type: str = 'call',
    method: str = 'brent',
    initial_guess: float = 0.2
) -> float:
    """
    Compute implied volatility from option price.

    Args:
        price: Market option price
        S: Spot price
        K: Strike price
        T: Time to maturity
        r: Risk-free rate
        option_type: 'call' or 'put'
        method: 'brent' or 'newton'
        initial_guess: Initial volatility guess

    Returns:
        Implied volatility (annualized)
    """
    if T <= 0:
        raise ValueError("Time to maturity must be positive")

    if price <= 0:
        return np.nan

    # Intrinsic value bounds
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
        bs_func = black_scholes_call
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
        bs_func = black_scholes_put

    if price < intrinsic:
        return np.nan

    # Objective function
    def objective(sigma):
        if sigma <= 0:
            return 1e10
        return bs_func(S, K, T, r, sigma) - price

    try:
        if method == 'brent':
            # Brent's method (robust, no derivative needed)
            iv = brentq(objective, 1e-6, 5.0, xtol=1e-6, maxiter=100)
        elif method == 'newton':
            # Newton-Raphson (faster, requires vega)
            def objective_with_derivative(sigma):
                return objective(sigma), black_scholes_vega(S, K, T, r, sigma)

            iv = newton(objective, initial_guess, fprime=lambda s: black_scholes_vega(S, K, T, r, s),
                       tol=1e-6, maxiter=50)
        else:
            raise ValueError(f"Unknown method: {method}")

        return iv if iv > 0 else np.nan

    except (ValueError, RuntimeError):
        return np.nan


def implied_volatility_surface(
    prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    r: float = 0.0,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Compute implied volatility surface from option prices.

    Args:
        prices: Option prices (n_maturities, n_strikes)
        S: Spot price
        strikes: Strike prices array
        maturities: Time to maturity array
        r: Risk-free rate
        option_type: 'call' or 'put'

    Returns:
        Implied volatility surface (n_maturities, n_strikes)
    """
    n_maturities = len(maturities)
    n_strikes = len(strikes)

    iv_surface = np.zeros((n_maturities, n_strikes))

    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            iv_surface[i, j] = implied_volatility(
                prices[i, j], S, K, T, r, option_type
            )

    return iv_surface


def implied_volatility_smile(
    prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float = 0.0,
    option_type: str = 'call'
) -> np.ndarray:
    """
    Compute implied volatility smile for a single maturity.

    Args:
        prices: Option prices for different strikes
        S: Spot price
        strikes: Strike prices
        T: Time to maturity
        r: Risk-free rate
        option_type: 'call' or 'put'

    Returns:
        Implied volatilities array
    """
    return np.array([
        implied_volatility(price, S, K, T, r, option_type)
        for price, K in zip(prices, strikes)
    ])


def svi_parametrization(
    log_moneyness: np.ndarray,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float
) -> np.ndarray:
    """
    SVI (Stochastic Volatility Inspired) parametrization for implied variance.

    w(k) = a + b[ρ(k-m) + √((k-m)² + σ²)]

    where k = log(K/F) is log-moneyness.

    Args:
        log_moneyness: Log-moneyness array
        a: Vertical shift
        b: Slope
        rho: Correlation-like parameter
        m: Horizontal shift
        sigma: Smoothness parameter

    Returns:
        Total implied variance w(k)
    """
    k = log_moneyness
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def fit_svi_to_smile(
    log_moneyness: np.ndarray,
    total_variance: np.ndarray,
    initial_params: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Fit SVI parametrization to implied variance smile.

    Args:
        log_moneyness: Log-moneyness points
        total_variance: Total implied variance (σ²T)
        initial_params: Initial guess [a, b, rho, m, sigma]

    Returns:
        Optimal SVI parameters
    """
    from scipy.optimize import minimize

    if initial_params is None:
        # Reasonable initial guess
        a = np.mean(total_variance)
        b = 0.1
        rho = 0.0
        m = 0.0
        sigma = 0.1
        initial_params = np.array([a, b, rho, m, sigma])

    def objective(params):
        a, b, rho, m, sigma = params

        # Parameter constraints
        if b < 0 or sigma <= 0:
            return 1e10
        if abs(rho) >= 1:
            return 1e10

        model_variance = svi_parametrization(log_moneyness, a, b, rho, m, sigma)
        return np.sum((model_variance - total_variance)**2)

    result = minimize(objective, initial_params, method='Nelder-Mead')

    return result.x if result.success else initial_params


def total_variance_to_iv(total_variance: np.ndarray, T: float) -> np.ndarray:
    """Convert total variance to implied volatility."""
    return np.sqrt(total_variance / T)


def iv_to_total_variance(iv: np.ndarray, T: float) -> np.ndarray:
    """Convert implied volatility to total variance."""
    return iv**2 * T
