# src/drift_removal.py
"""
Drift removal and detrending methods.
Implements: HP filter, linear detrending, rolling normalization.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hp_filter(series: pd.Series, lamb: float = 1600.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Hodrick-Prescott filter decomposition.
    
    Decomposes series into:
    - trend: long-term component
    - cycle: cyclical component
    - residual: stationary residual
    
    lamb: smoothing parameter
        - 1600 for quarterly data (standard)
        - 14400 for monthly data
        - 129600 for weekly data
        - For minute data: use higher values (e.g., 100000-1000000)
    
    Returns: (trend, cycle, residual)
    """
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        
        series_clean = series.dropna()
        
        cycle, trend = hpfilter(series_clean, lamb=lamb)
        
        # Residual is the difference
        residual = series_clean - trend
        
        logger.info(f"HP Filter applied: lambda={lamb}")
        logger.info(f"  Trend std: {trend.std():.6f}")
        logger.info(f"  Cycle std: {cycle.std():.6f}")
        logger.info(f"  Residual std: {residual.std():.6f}")
        
        # Align back to original index
        trend_full = pd.Series(index=series.index, dtype=float)
        cycle_full = pd.Series(index=series.index, dtype=float)
        residual_full = pd.Series(index=series.index, dtype=float)
        
        trend_full.loc[series_clean.index] = trend
        cycle_full.loc[series_clean.index] = cycle
        residual_full.loc[series_clean.index] = residual
        
        return trend_full, cycle_full, residual_full
        
    except ImportError:
        logger.error("statsmodels not installed. Install with: pip install statsmodels")
        raise
    except Exception as e:
        logger.error(f"HP filter failed: {e}")
        raise


def linear_detrend(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Remove linear trend using OLS regression.
    
    Returns: (trend, detrended)
    """
    series_clean = series.dropna()
    
    # Create time index (0, 1, 2, ...)
    x = np.arange(len(series_clean))
    y = series_clean.values
    
    # Fit linear trend
    coeffs = np.polyfit(x, y, deg=1)
    trend_values = np.polyval(coeffs, x)
    
    detrended_values = y - trend_values
    
    # Create series
    trend = pd.Series(trend_values, index=series_clean.index)
    detrended = pd.Series(detrended_values, index=series_clean.index)
    
    # Align to original index
    trend_full = pd.Series(index=series.index, dtype=float)
    detrended_full = pd.Series(index=series.index, dtype=float)
    
    trend_full.loc[series_clean.index] = trend
    detrended_full.loc[series_clean.index] = detrended
    
    logger.info(f"Linear detrend: slope={coeffs[0]:.6f}, intercept={coeffs[1]:.6f}")
    
    return trend_full, detrended_full


def rolling_normalize(series: pd.Series, window: int = 1440) -> pd.Series:
    """
    Rolling z-score normalization.
    
    Subtracts rolling mean and divides by rolling std.
    window: rolling window size (e.g., 1440 minutes = 1 day)
    
    Returns: normalized series
    """
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    
    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)
    
    normalized = (series - rolling_mean) / rolling_std
    
    logger.info(f"Rolling normalization: window={window}")
    logger.info(f"  Normalized mean: {normalized.mean():.6f}")
    logger.info(f"  Normalized std: {normalized.std():.6f}")
    
    return normalized


def remove_drift_and_decompose(df: pd.DataFrame, 
                                price_col: str = 'close',
                                method: str = 'hp',
                                hp_lambda: float = 100000.0,
                                rolling_window: int = 1440) -> pd.DataFrame:
    """
    Apply drift removal to price series and add decomposed components.
    
    Args:
        df: DataFrame with price data
        price_col: column name for price
        method: 'hp', 'linear', or 'rolling'
        hp_lambda: lambda for HP filter
        rolling_window: window for rolling normalization
    
    Returns:
        DataFrame with added columns:
        - {price_col}_trend
        - {price_col}_cycle (HP only)
        - {price_col}_detrended
        - {price_col}_normalized
    """
    df = df.copy()
    
    if price_col not in df.columns:
        raise ValueError(f"Column {price_col} not found in DataFrame")
    
    series = df[price_col]
    
    logger.info(f"\nApplying drift removal method: {method}")
    logger.info(f"Original series - mean: {series.mean():.4f}, std: {series.std():.4f}")
    
    if method == 'hp':
        trend, cycle, residual = hp_filter(series, lamb=hp_lambda)
        df[f'{price_col}_trend'] = trend
        df[f'{price_col}_cycle'] = cycle
        df[f'{price_col}_detrended'] = residual
        
    elif method == 'linear':
        trend, detrended = linear_detrend(series)
        df[f'{price_col}_trend'] = trend
        df[f'{price_col}_detrended'] = detrended
        
    elif method == 'rolling':
        normalized = rolling_normalize(series, window=rolling_window)
        df[f'{price_col}_normalized'] = normalized
        df[f'{price_col}_detrended'] = normalized
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'hp', 'linear', or 'rolling'")
    
    # Always add rolling normalized version
    if method != 'rolling':
        df[f'{price_col}_normalized'] = rolling_normalize(series, window=rolling_window)
    
    logger.info(f"Detrended series - mean: {df[f'{price_col}_detrended'].mean():.4f}, "
               f"std: {df[f'{price_col}_detrended'].std():.4f}")
    
    return df


if __name__ == "__main__":
    from config import CLEANED_PARQUET, STATIONARY_PARQUET
    
    logger.info("Loading cleaned data...")
    df = pd.read_parquet(CLEANED_PARQUET)
    
    # Apply HP filter decomposition
    df = remove_drift_and_decompose(
        df, 
        price_col='close',
        method='hp',
        hp_lambda=100000.0
    )
    
    # Save
    df.to_parquet(STATIONARY_PARQUET)
    logger.info(f"Saved stationary data to {STATIONARY_PARQUET}")
