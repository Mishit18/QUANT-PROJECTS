"""
Utility functions for statistical arbitrage research.

This module provides common utilities used across the project including
configuration loading, logging, data validation, and helper functions.
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/strategy_config.yaml") -> Dict:
    """
    Load strategy configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(directory: Union[str, Path]) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def validate_price_data(prices: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate price data quality.
    
    Args:
        prices: DataFrame with price data
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check for missing values
    missing_pct = prices.isnull().sum() / len(prices)
    for col in missing_pct[missing_pct > 0.05].index:
        issues.append(f"{col}: {missing_pct[col]:.1%} missing data")
    
    # Check for non-positive prices
    if (prices <= 0).any().any():
        issues.append("Non-positive prices detected")
    
    # Check for extreme returns
    returns = prices.pct_change()
    extreme_returns = (returns.abs() > 0.5).sum()
    for col in extreme_returns[extreme_returns > 0].index:
        issues.append(f"{col}: {extreme_returns[col]} extreme returns (>50%)")
    
    # Check for sufficient data
    if len(prices) < 252:
        issues.append(f"Insufficient data: {len(prices)} observations (need 252+)")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def align_series(*series: pd.Series) -> List[pd.Series]:
    """
    Align multiple time series to common dates.
    
    Args:
        *series: Variable number of pandas Series
        
    Returns:
        List of aligned series
    """
    # Find common dates
    common_index = series[0].index
    for s in series[1:]:
        common_index = common_index.intersection(s.index)
    
    # Align all series
    aligned = [s.loc[common_index] for s in series]
    return aligned


def compute_returns(prices: pd.DataFrame, 
                   method: str = "log") -> pd.DataFrame:
    """
    Compute returns from price series.
    
    Args:
        prices: DataFrame with price data
        method: "simple" or "log" returns
        
    Returns:
        DataFrame with returns
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1))
    elif method == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns


def compute_rolling_stats(series: pd.Series, 
                         window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Compute rolling mean and standard deviation.
    
    Args:
        series: Input time series
        window: Rolling window size
        
    Returns:
        Tuple of (rolling_mean, rolling_std)
    """
    rolling_mean = series.rolling(window=window, min_periods=window//2).mean()
    rolling_std = series.rolling(window=window, min_periods=window//2).std()
    return rolling_mean, rolling_std


def compute_zscore(series: pd.Series, 
                   window: int = 60,
                   min_periods: Optional[int] = None) -> pd.Series:
    """
    Compute rolling z-score of a time series.
    
    Args:
        series: Input time series
        window: Rolling window for mean and std calculation
        min_periods: Minimum observations required
        
    Returns:
        Z-score series
    """
    if min_periods is None:
        min_periods = window // 2
    
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # Avoid division by zero
    zscore = (series - rolling_mean) / (rolling_std + 1e-8)
    return zscore


def annualize_return(returns: Union[float, pd.Series], 
                     periods_per_year: int = 252) -> Union[float, pd.Series]:
    """
    Annualize returns.
    
    Args:
        returns: Period returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized returns
    """
    if isinstance(returns, pd.Series):
        return (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
    else:
        return (1 + returns) ** periods_per_year - 1


def annualize_volatility(returns: pd.Series, 
                        periods_per_year: int = 252) -> float:
    """
    Annualize volatility.
    
    Args:
        returns: Period returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized volatility
    """
    return returns.std() * np.sqrt(periods_per_year)


def compute_sharpe_ratio(returns: pd.Series, 
                        risk_free_rate: float = 0.02,
                        periods_per_year: int = 252) -> float:
    """
    Compute Sharpe ratio.
    
    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def compute_sortino_ratio(returns: pd.Series,
                         risk_free_rate: float = 0.02,
                         periods_per_year: int = 252) -> float:
    """
    Compute Sortino ratio (uses downside deviation).
    
    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods in a year
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def compute_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Compute maximum drawdown and its duration.
    
    Args:
        equity_curve: Cumulative equity curve
        
    Returns:
        Tuple of (max_drawdown, peak_date, trough_date)
    """
    cummax = equity_curve.expanding().max()
    drawdown = (equity_curve - cummax) / cummax
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    
    # Find peak date (last maximum before trough)
    peak_date = equity_curve[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def compute_calmar_ratio(returns: pd.Series,
                        periods_per_year: int = 252) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Period returns
        periods_per_year: Number of periods in a year
        
    Returns:
        Calmar ratio
    """
    equity_curve = (1 + returns).cumprod()
    max_dd, _, _ = compute_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    annual_return = annualize_return(returns, periods_per_year)
    return annual_return / abs(max_dd)


def save_results(data: Union[pd.DataFrame, pd.Series, Dict],
                filename: str,
                output_dir: str = "results") -> None:
    """
    Save results to file.
    
    Args:
        data: Data to save
        filename: Output filename
        output_dir: Output directory
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if filename.endswith('.parquet'):
            data.to_parquet(filepath)
        else:
            data.to_csv(filepath)
    elif isinstance(data, dict):
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def load_results(filename: str,
                input_dir: str = "results") -> Union[pd.DataFrame, Dict]:
    """
    Load results from file.
    
    Args:
        filename: Input filename
        input_dir: Input directory
        
    Returns:
        Loaded data
    """
    filepath = os.path.join(input_dir, filename)
    
    if filename.endswith('.parquet'):
        return pd.read_parquet(filepath)
    elif filename.endswith('.csv'):
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif filename.endswith('.yaml') or filename.endswith('.yml'):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file type: {filename}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string."""
    return f"${value:,.{decimals}f}"


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
