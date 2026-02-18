"""
Data preprocessing and feature engineering for financial time series.

Handles missing values, outliers, normalization, and feature construction.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from scipy import stats
from src.utils import log_returns, winsorize, z_score, realized_volatility


class DataPreprocessor:
    """
    Comprehensive preprocessing pipeline for financial data.
    """
    
    def __init__(self, 
                 handle_missing: str = 'forward_fill',
                 outlier_method: str = 'winsorize',
                 outlier_threshold: float = 3.0,
                 normalize: bool = False):
        """
        Initialize preprocessor.
        
        Parameters
        ----------
        handle_missing : str
            Method for handling missing values: 'forward_fill', 'drop', 'interpolate'
        outlier_method : str
            Method for handling outliers: 'winsorize', 'clip', 'none'
        outlier_threshold : float
            Threshold for outlier detection (z-score or percentile)
        normalize : bool
            Whether to normalize features
        """
        self.handle_missing = handle_missing
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.normalize = normalize
        
        self.mean_ = None
        self.std_ = None
    
    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit preprocessor parameters (mean, std for normalization).
        
        Parameters
        ----------
        data : pd.DataFrame
            Training data
            
        Returns
        -------
        self
        """
        if self.normalize:
            self.mean_ = data.mean()
            self.std_ = data.std()
        
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing transformations.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        data = data.copy()
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Handle outliers
        data = self._handle_outliers(data)
        
        # Normalize
        if self.normalize and self.mean_ is not None:
            data = (data - self.mean_) / (self.std_ + 1e-10)
        
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        return self.fit(data).transform(data)
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to specified method."""
        if self.handle_missing == 'forward_fill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif self.handle_missing == 'drop':
            return data.dropna()
        elif self.handle_missing == 'interpolate':
            return data.interpolate(method='linear').fillna(method='bfill')
        else:
            return data
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers according to specified method."""
        if self.outlier_method == 'none':
            return data
        
        for col in data.columns:
            if self.outlier_method == 'winsorize':
                lower_pct = self.outlier_threshold
                upper_pct = 100 - self.outlier_threshold
                data[col] = winsorize(data[col].values, lower_pct, upper_pct)
            
            elif self.outlier_method == 'clip':
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                threshold = self.outlier_threshold
                mean = data[col].mean()
                std = data[col].std()
                data[col] = np.clip(data[col], 
                                   mean - threshold * std,
                                   mean + threshold * std)
        
        return data


def preprocess_data(data: pd.DataFrame,
                   return_type: str = 'log',
                   handle_missing: str = 'forward_fill',
                   outlier_method: str = 'winsorize',
                   normalize: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Complete preprocessing pipeline for market data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw price data
    return_type : str
        Type of returns: 'log' or 'simple'
    handle_missing : str
        Missing value handling method
    outlier_method : str
        Outlier handling method
    normalize : bool
        Whether to normalize
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'prices': Cleaned prices
        - 'returns': Calculated returns
        - 'log_prices': Log prices
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        handle_missing=handle_missing,
        outlier_method=outlier_method,
        normalize=False  # Don't normalize prices
    )
    
    # Clean prices
    prices = preprocessor.fit_transform(data)
    
    # Calculate returns
    if return_type == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = (prices / prices.shift(1) - 1).dropna()
    
    # Handle outliers in returns
    returns_preprocessor = DataPreprocessor(
        handle_missing='drop',
        outlier_method=outlier_method,
        normalize=normalize
    )
    returns = returns_preprocessor.fit_transform(returns)
    
    # Log prices
    log_prices = np.log(prices)
    
    return {
        'prices': prices,
        'returns': returns,
        'log_prices': log_prices
    }


def create_features(prices: pd.DataFrame, 
                   returns: pd.DataFrame,
                   windows: list = [5, 20, 60]) -> pd.DataFrame:
    """
    Create technical features for modeling.
    
    Parameters
    ----------
    prices : pd.DataFrame
        Price data
    returns : pd.DataFrame
        Return data
    windows : list
        List of window sizes for rolling features
        
    Returns
    -------
    pd.DataFrame
        Feature matrix
    """
    features = pd.DataFrame(index=returns.index)
    
    for col in returns.columns:
        # Raw returns
        features[f'{col}_return'] = returns[col]
        
        # Rolling statistics
        for window in windows:
            # Rolling mean
            features[f'{col}_ma_{window}'] = returns[col].rolling(window).mean()
            
            # Rolling volatility
            features[f'{col}_vol_{window}'] = returns[col].rolling(window).std() * np.sqrt(252)
            
            # Rolling skewness
            features[f'{col}_skew_{window}'] = returns[col].rolling(window).skew()
            
            # Rolling kurtosis
            features[f'{col}_kurt_{window}'] = returns[col].rolling(window).kurt()
            
            # Price momentum
            features[f'{col}_mom_{window}'] = prices[col].pct_change(window)
        
        # Lagged returns
        for lag in [1, 2, 3, 5]:
            features[f'{col}_lag_{lag}'] = returns[col].shift(lag)
    
    return features.dropna()


def train_test_split(data: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     validation_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.
    
    Maintains temporal order (no shuffling).
    
    Parameters
    ----------
    data : pd.DataFrame
        Input data
    train_ratio : float
        Proportion for training
    validation_ratio : float
        Proportion for validation
        
    Returns
    -------
    train : pd.DataFrame
        Training set
    val : pd.DataFrame
        Validation set
    test : pd.DataFrame
        Test set
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + validation_ratio))
    
    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    
    return train, val, test


def create_lagged_features(data: pd.Series, 
                          n_lags: int = 5) -> pd.DataFrame:
    """
    Create lagged features for time series prediction.
    
    Parameters
    ----------
    data : pd.Series
        Input series
    n_lags : int
        Number of lags
        
    Returns
    -------
    pd.DataFrame
        Lagged feature matrix
    """
    lagged = pd.DataFrame(index=data.index)
    
    for lag in range(1, n_lags + 1):
        lagged[f'lag_{lag}'] = data.shift(lag)
    
    return lagged.dropna()


def detrend(data: pd.Series, method: str = 'linear') -> Tuple[pd.Series, np.ndarray]:
    """
    Remove trend from time series.
    
    Parameters
    ----------
    data : pd.Series
        Input series
    method : str
        Detrending method: 'linear', 'quadratic', 'mean'
        
    Returns
    -------
    detrended : pd.Series
        Detrended series
    trend : np.ndarray
        Estimated trend
    """
    x = np.arange(len(data))
    y = data.values
    
    if method == 'mean':
        trend = np.full(len(data), np.mean(y))
    elif method == 'linear':
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
    elif method == 'quadratic':
        coeffs = np.polyfit(x, y, 2)
        trend = np.polyval(coeffs, x)
    else:
        raise ValueError(f"Unknown detrending method: {method}")
    
    detrended = pd.Series(y - trend, index=data.index)
    
    return detrended, trend


def calculate_market_features(returns: pd.DataFrame, 
                             market_col: str = 'SPY') -> pd.DataFrame:
    """
    Calculate market-relative features (beta, alpha, correlation).
    
    Parameters
    ----------
    returns : pd.DataFrame
        Return data with market column
    market_col : str
        Name of market return column
        
    Returns
    -------
    pd.DataFrame
        Market features
    """
    features = pd.DataFrame(index=returns.index)
    market_returns = returns[market_col]
    
    for col in returns.columns:
        if col == market_col:
            continue
        
        asset_returns = returns[col]
        
        # Rolling beta (60-day)
        window = 60
        rolling_cov = asset_returns.rolling(window).cov(market_returns)
        rolling_var = market_returns.rolling(window).var()
        features[f'{col}_beta'] = rolling_cov / rolling_var
        
        # Rolling correlation
        features[f'{col}_corr'] = asset_returns.rolling(window).corr(market_returns)
        
        # Excess return
        features[f'{col}_excess'] = asset_returns - market_returns
    
    return features.dropna()


if __name__ == '__main__':
    # Test preprocessing
    from src.data_loader import load_sample_data
    
    print("Testing preprocessing pipeline...")
    data = load_sample_data()
    
    processed = preprocess_data(data['prices'])
    print(f"\nProcessed returns shape: {processed['returns'].shape}")
    print(f"Missing values: {processed['returns'].isna().sum().sum()}")
    
    features = create_features(processed['prices'], processed['returns'])
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()[:10]}...")
