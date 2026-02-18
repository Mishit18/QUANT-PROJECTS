"""
Utility functions for systematic factor modeling
Production-grade helper functions with proper error handling
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union, List
from scipy import stats
from pathlib import Path
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def ensure_directories(config: dict) -> None:
    """Create necessary directories if they don't exist"""
    dirs = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['results'],
        config['paths']['plots'],
        config['paths']['reports'],
        config['paths']['logs']
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directory structure validated")


def compute_returns(prices: pd.DataFrame, 
                    return_type: str = 'log') -> pd.DataFrame:
    """
    Compute returns from price data
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data with datetime index
    return_type : str
        'log' for log returns, 'simple' for arithmetic returns
    
    Returns:
    --------
    pd.DataFrame : Returns
    """
    if return_type == 'log':
        returns = np.log(prices / prices.shift(1))
    elif return_type == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"Unknown return type: {return_type}")
    
    return returns.dropna()


def compute_excess_returns(returns: pd.DataFrame, 
                          rf_rate: float = 0.02) -> pd.DataFrame:
    """
    Compute excess returns over risk-free rate
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Asset returns
    rf_rate : float
        Annual risk-free rate
    
    Returns:
    --------
    pd.DataFrame : Excess returns
    """
    # Convert annual rate to daily
    daily_rf = (1 + rf_rate) ** (1/252) - 1
    excess_returns = returns - daily_rf
    
    return excess_returns


def winsorize_returns(returns: pd.DataFrame, 
                     limits: Tuple[float, float] = (0.01, 0.01)) -> pd.DataFrame:
    """
    Winsorize returns to handle outliers
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    limits : tuple
        Lower and upper percentile limits
    
    Returns:
    --------
    pd.DataFrame : Winsorized returns
    """
    return returns.apply(lambda x: stats.mstats.winsorize(x, limits=limits))


def detect_outliers(returns: pd.DataFrame, 
                   threshold: float = 5.0) -> pd.DataFrame:
    """
    Detect outliers using z-score method
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    threshold : float
        Z-score threshold for outlier detection
    
    Returns:
    --------
    pd.DataFrame : Boolean mask of outliers
    """
    z_scores = np.abs(stats.zscore(returns, nan_policy='omit'))
    return z_scores > threshold


def handle_missing_data(data: pd.DataFrame, 
                       max_missing_pct: float = 0.05,
                       method: str = 'drop') -> pd.DataFrame:
    """
    Handle missing data in panel
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with potential missing values
    max_missing_pct : float
        Maximum allowed missing percentage
    method : str
        'drop' or 'fill'
    
    Returns:
    --------
    pd.DataFrame : Cleaned data
    """
    missing_pct = data.isnull().sum() / len(data)
    
    # Drop columns with too much missing data
    valid_cols = missing_pct[missing_pct <= max_missing_pct].index
    data_clean = data[valid_cols].copy()
    
    logger.info(f"Dropped {len(data.columns) - len(valid_cols)} columns due to missing data")
    
    if method == 'fill':
        # Forward fill then backward fill
        data_clean = data_clean.fillna(method='ffill').fillna(method='bfill')
    elif method == 'drop':
        data_clean = data_clean.dropna()
    
    return data_clean


def compute_summary_statistics(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive summary statistics
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    
    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    stats_dict = {
        'Mean': returns.mean() * 252,  # Annualized
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe': (returns.mean() / returns.std()) * np.sqrt(252),
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis(),
        'Min': returns.min(),
        'Max': returns.max(),
        'VaR_95': returns.quantile(0.05),
        'CVaR_95': returns[returns <= returns.quantile(0.05)].mean()
    }
    
    return pd.DataFrame(stats_dict).T


def compute_sharpe_ratio(returns: pd.Series, 
                        rf_rate: float = 0.0,
                        annualize: bool = True) -> float:
    """
    Compute Sharpe ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    rf_rate : float
        Risk-free rate (already in same frequency as returns)
    annualize : bool
        Whether to annualize the ratio
    
    Returns:
    --------
    float : Sharpe ratio
    """
    excess_returns = returns - rf_rate
    sharpe = excess_returns.mean() / excess_returns.std()
    
    if annualize:
        sharpe *= np.sqrt(252)
    
    return sharpe


def compute_information_ratio(returns: pd.Series, 
                             benchmark: pd.Series) -> float:
    """
    Compute information ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Strategy returns
    benchmark : pd.Series
        Benchmark returns
    
    Returns:
    --------
    float : Information ratio
    """
    active_returns = returns - benchmark
    ir = active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    return ir


def compute_maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Compute maximum drawdown and its timing
    
    Parameters:
    -----------
    returns : pd.Series
        Return series
    
    Returns:
    --------
    tuple : (max_drawdown, peak_date, trough_date)
    """
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    max_dd = drawdown.min()
    trough_date = drawdown.idxmin()
    peak_date = cum_returns[:trough_date].idxmax()
    
    return max_dd, peak_date, trough_date


def compute_turnover(weights_old: pd.Series, 
                    weights_new: pd.Series) -> float:
    """
    Compute portfolio turnover
    
    Parameters:
    -----------
    weights_old : pd.Series
        Previous period weights
    weights_new : pd.Series
        Current period weights
    
    Returns:
    --------
    float : Turnover (sum of absolute weight changes)
    """
    # Align indices
    all_assets = weights_old.index.union(weights_new.index)
    w_old = weights_old.reindex(all_assets, fill_value=0)
    w_new = weights_new.reindex(all_assets, fill_value=0)
    
    turnover = np.abs(w_new - w_old).sum()
    
    return turnover


def apply_transaction_costs(returns: pd.Series, 
                           turnover: pd.Series,
                           cost_bps: float = 10) -> pd.Series:
    """
    Apply transaction costs to returns
    
    Parameters:
    -----------
    returns : pd.Series
        Gross returns
    turnover : pd.Series
        Portfolio turnover
    cost_bps : float
        Transaction cost in basis points
    
    Returns:
    --------
    pd.Series : Net returns after costs
    """
    costs = turnover * (cost_bps / 10000)
    net_returns = returns - costs
    
    return net_returns


def standardize(data: pd.DataFrame, 
               method: str = 'zscore') -> pd.DataFrame:
    """
    Standardize data
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to standardize
    method : str
        'zscore' or 'minmax'
    
    Returns:
    --------
    pd.DataFrame : Standardized data
    """
    if method == 'zscore':
        return (data - data.mean()) / data.std()
    elif method == 'minmax':
        return (data - data.min()) / (data.max() - data.min())
    else:
        raise ValueError(f"Unknown standardization method: {method}")


def rank_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rank transformation (cross-sectional)
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to rank
    
    Returns:
    --------
    pd.DataFrame : Ranked data (0 to 1)
    """
    return data.rank(axis=1, pct=True)


def test_stationarity(series: pd.Series, 
                     test: str = 'adf') -> Tuple[float, float]:
    """
    Test for stationarity using ADF or KPSS test
    
    Parameters:
    -----------
    series : pd.Series
        Time series to test
    test : str
        'adf' for Augmented Dickey-Fuller or 'kpss' for KPSS
    
    Returns:
    --------
    tuple : (test_statistic, p_value)
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    series_clean = series.dropna()
    
    if test == 'adf':
        result = adfuller(series_clean)
        return result[0], result[1]
    elif test == 'kpss':
        result = kpss(series_clean)
        return result[0], result[1]
    else:
        raise ValueError(f"Unknown test: {test}")


def compute_correlation_matrix(returns: pd.DataFrame, 
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Compute correlation matrix with proper handling
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Return data
    method : str
        'pearson', 'spearman', or 'kendall'
    
    Returns:
    --------
    pd.DataFrame : Correlation matrix
    """
    return returns.corr(method=method)


def save_results(data: Union[pd.DataFrame, pd.Series, dict],
                filename: str,
                output_dir: str = "results") -> None:
    """
    Save results to file
    
    Parameters:
    -----------
    data : DataFrame, Series, or dict
        Data to save
    filename : str
        Output filename
    output_dir : str
        Output directory
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / filename
    
    if isinstance(data, (pd.DataFrame, pd.Series)):
        if filename.endswith('.csv'):
            data.to_csv(filepath)
        elif filename.endswith('.parquet'):
            try:
                data.to_parquet(filepath)
            except:
                # Fallback to CSV if parquet fails
                csv_path = filepath.with_suffix('.csv')
                data.to_csv(csv_path)
                logger.warning(f"Parquet save failed, saved as CSV: {csv_path}")
        else:
            data.to_pickle(filepath)
    elif isinstance(data, dict):
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Results saved to {filepath}")
