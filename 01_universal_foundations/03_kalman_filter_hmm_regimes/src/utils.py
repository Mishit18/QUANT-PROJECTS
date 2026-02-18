"""
Utility functions for quantitative analysis.

Provides common operations for data manipulation, statistical calculations,
and numerical stability enhancements.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
from scipy import stats


def ensure_numpy(data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
    """
    Convert input to numpy array with consistent shape handling.
    
    Parameters
    ----------
    data : array-like
        Input data
        
    Returns
    -------
    np.ndarray
        Numpy array representation
    """
    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.values
    return np.asarray(data)


def log_returns(prices: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Calculate log returns from price series.
    
    Parameters
    ----------
    prices : array-like
        Price series
        
    Returns
    -------
    array-like
        Log returns (length n-1)
    """
    if isinstance(prices, pd.Series):
        return np.log(prices / prices.shift(1)).dropna()
    else:
        prices = np.asarray(prices)
        return np.log(prices[1:] / prices[:-1])


def simple_returns(prices: Union[np.ndarray, pd.Series]) -> Union[np.ndarray, pd.Series]:
    """
    Calculate simple returns from price series.
    
    Parameters
    ----------
    prices : array-like
        Price series
        
    Returns
    -------
    array-like
        Simple returns (length n-1)
    """
    if isinstance(prices, pd.Series):
        return (prices / prices.shift(1) - 1).dropna()
    else:
        prices = np.asarray(prices)
        return prices[1:] / prices[:-1] - 1


def rolling_window(data: np.ndarray, window: int) -> np.ndarray:
    """
    Create rolling window view of array for vectorized operations.
    
    Parameters
    ----------
    data : np.ndarray
        Input array (1D)
    window : int
        Window size
        
    Returns
    -------
    np.ndarray
        Shape (n_windows, window)
    """
    shape = (data.shape[0] - window + 1, window)
    strides = (data.strides[0], data.strides[0])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def realized_volatility(returns: np.ndarray, window: int = 20, 
                        annualization_factor: float = 252.0) -> np.ndarray:
    """
    Calculate rolling realized volatility.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    window : int
        Rolling window size
    annualization_factor : float
        Annualization factor (252 for daily data)
        
    Returns
    -------
    np.ndarray
        Realized volatility series
    """
    returns = ensure_numpy(returns)
    if len(returns) < window:
        return np.full(len(returns), np.nan)
    
    rv = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        rv[i] = np.std(returns[i - window + 1:i + 1]) * np.sqrt(annualization_factor)
    
    return rv


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                 annualization_factor: float = 252.0) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    risk_free_rate : float
        Annualized risk-free rate
    annualization_factor : float
        Annualization factor
        
    Returns
    -------
    float
        Sharpe ratio
    """
    returns = ensure_numpy(returns)
    excess_returns = returns - risk_free_rate / annualization_factor
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annualization_factor)


def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                  annualization_factor: float = 252.0) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation).
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    risk_free_rate : float
        Annualized risk-free rate
    annualization_factor : float
        Annualization factor
        
    Returns
    -------
    float
        Sortino ratio
    """
    returns = ensure_numpy(returns)
    excess_returns = returns - risk_free_rate / annualization_factor
    
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    return np.mean(excess_returns) / downside_std * np.sqrt(annualization_factor)


def max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
        
    Returns
    -------
    max_dd : float
        Maximum drawdown (positive value)
    start_idx : int
        Start index of max drawdown
    end_idx : int
        End index of max drawdown
    """
    returns = ensure_numpy(returns)
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    
    end_idx = np.argmin(drawdown)
    start_idx = np.argmax(cumulative[:end_idx + 1])
    max_dd = -drawdown[end_idx]
    
    return max_dd, start_idx, end_idx


def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                      annualization_factor: float = 252.0) -> float:
    """
    Calculate information ratio (active return / tracking error).
    
    Parameters
    ----------
    returns : np.ndarray
        Strategy returns
    benchmark_returns : np.ndarray
        Benchmark returns
    annualization_factor : float
        Annualization factor
        
    Returns
    -------
    float
        Information ratio
    """
    active_returns = ensure_numpy(returns) - ensure_numpy(benchmark_returns)
    tracking_error = np.std(active_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(active_returns) / tracking_error * np.sqrt(annualization_factor)


def ensure_positive_definite(matrix: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Ensure matrix is positive definite with PRODUCTION-GRADE numerical stability.
    
    Uses eigenvalue decomposition and regularization to guarantee
    positive definiteness while preserving matrix structure.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (must be square)
    epsilon : float
        Minimum eigenvalue threshold
        
    Returns
    -------
    np.ndarray
        Positive definite matrix
        
    Raises
    ------
    ValueError
        If matrix is not square or contains NaN/Inf
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    
    # Handle scalar case
    if matrix.ndim == 0:
        return np.maximum(matrix, epsilon)
    
    # Handle 1D case
    if matrix.ndim == 1:
        return np.maximum(matrix, epsilon)
    
    # Validate square matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape}")
    
    # Check for NaN/Inf
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        raise ValueError("Matrix contains NaN or Inf values")
    
    # Make symmetric (average with transpose)
    matrix = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        # Fallback: add diagonal regularization
        return matrix + epsilon * np.eye(matrix.shape[0])
    
    # Clip negative eigenvalues
    eigenvalues = np.maximum(eigenvalues, epsilon)
    
    # Reconstruct matrix
    result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure symmetry
    result = (result + result.T) / 2
    
    # Final validation
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        # Emergency fallback
        return matrix + epsilon * np.eye(matrix.shape[0])
    
    return result


def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    """
    Compute matrix square root via eigendecomposition.
    
    Parameters
    ----------
    matrix : np.ndarray
        Symmetric positive definite matrix
        
    Returns
    -------
    np.ndarray
        Matrix square root
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Numerical stability
    return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T


def winsorize(data: np.ndarray, lower_percentile: float = 1.0,
              upper_percentile: float = 99.0) -> np.ndarray:
    """
    Winsorize data to handle outliers.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    lower_percentile : float
        Lower percentile threshold
    upper_percentile : float
        Upper percentile threshold
        
    Returns
    -------
    np.ndarray
        Winsorized data
    """
    data = ensure_numpy(data)
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)


def z_score(data: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """
    Calculate z-scores (standardized values).
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    window : int, optional
        Rolling window for mean/std calculation. If None, use full sample.
        
    Returns
    -------
    np.ndarray
        Z-scores
    """
    data = ensure_numpy(data)
    
    if window is None:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-10)
    else:
        z_scores = np.full(len(data), np.nan)
        for i in range(window - 1, len(data)):
            window_data = data[i - window + 1:i + 1]
            mean = np.mean(window_data)
            std = np.std(window_data)
            z_scores[i] = (data[i] - mean) / (std + 1e-10)
        return z_scores


def autocorrelation(data: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Calculate autocorrelation function.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    max_lag : int
        Maximum lag
        
    Returns
    -------
    np.ndarray
        Autocorrelation values for lags 0 to max_lag
    """
    data = ensure_numpy(data)
    data = data - np.mean(data)
    
    acf = np.zeros(max_lag + 1)
    variance = np.sum(data ** 2) / len(data)
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            acf[lag] = np.sum(data[:-lag] * data[lag:]) / (len(data) * variance)
    
    return acf


def half_life(data: np.ndarray) -> float:
    """
    Estimate mean-reversion half-life via AR(1) model.
    
    Parameters
    ----------
    data : np.ndarray
        Input data (assumed to be mean-reverting)
        
    Returns
    -------
    float
        Half-life in units of data frequency
    """
    data = ensure_numpy(data)
    lagged = data[:-1]
    current = data[1:]
    
    # AR(1): y_t = α + β * y_{t-1} + ε_t
    # Half-life = -log(2) / log(β)
    beta = np.cov(current, lagged)[0, 1] / np.var(lagged)
    
    if beta >= 1 or beta <= 0:
        return np.inf
    
    return -np.log(2) / np.log(beta)


def hurst_exponent(data: np.ndarray, max_lag: int = 20) -> float:
    """
    Estimate Hurst exponent via rescaled range analysis.
    
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    max_lag : int
        Maximum lag for R/S calculation
        
    Returns
    -------
    float
        Hurst exponent
    """
    data = ensure_numpy(data)
    lags = range(2, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        # Split data into chunks
        n_chunks = len(data) // lag
        if n_chunks == 0:
            continue
            
        rs_chunk = []
        for i in range(n_chunks):
            chunk = data[i * lag:(i + 1) * lag]
            mean = np.mean(chunk)
            deviations = chunk - mean
            cumulative_deviations = np.cumsum(deviations)
            
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(chunk)
            
            if S > 0:
                rs_chunk.append(R / S)
        
        if len(rs_chunk) > 0:
            rs_values.append(np.mean(rs_chunk))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Fit log(R/S) = log(c) + H * log(lag)
    log_lags = np.log(list(lags)[:len(rs_values)])
    log_rs = np.log(rs_values)
    
    hurst = np.polyfit(log_lags, log_rs, 1)[0]
    return hurst
