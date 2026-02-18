"""
Market data acquisition and loading utilities.

Handles downloading financial data from various sources and provides
a unified interface for data access.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Union, Optional, Dict
from datetime import datetime, timedelta
import os


class DataLoader:
    """
    Unified interface for loading financial market data.
    """
    
    def __init__(self, data_dir: str = 'data/raw'):
        """
        Initialize data loader.
        
        Parameters
        ----------
        data_dir : str
            Directory for storing raw data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_yahoo(self, 
                      tickers: Union[str, List[str]], 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      period: str = '5y') -> pd.DataFrame:
        """
        Download data from Yahoo Finance.
        
        Parameters
        ----------
        tickers : str or list of str
            Ticker symbols
        start_date : str, optional
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date (YYYY-MM-DD)
        period : str
            Period if start_date not specified (e.g., '5y', '10y')
            
        Returns
        -------
        pd.DataFrame
            OHLCV data with MultiIndex columns (ticker, field)
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        
        print(f"Downloading data for {tickers}...")
        
        if start_date is None:
            data = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        else:
            data = yf.download(tickers, start=start_date, end=end_date, 
                             auto_adjust=True, progress=False)
        
        # Handle single ticker case
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([tickers, data.columns])
        
        # Save to disk
        filename = f"{'_'.join(tickers)}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.data_dir, filename)
        data.to_csv(filepath)
        print(f"Data saved to {filepath}")
        
        return data
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Reconstruct MultiIndex if present
        if '.' in data.columns[0]:
            data.columns = pd.MultiIndex.from_tuples(
                [tuple(col.split('.')) for col in data.columns]
            )
        
        return data
    
    def get_price_series(self, data: pd.DataFrame, ticker: str, 
                        field: str = 'Close') -> pd.Series:
        """
        Extract single price series from multi-ticker DataFrame.
        
        Parameters
        ----------
        data : pd.DataFrame
            Multi-ticker data
        ticker : str
            Ticker symbol
        field : str
            Price field (Close, Open, High, Low, Volume)
            
        Returns
        -------
        pd.Series
            Price series
        """
        if isinstance(data.columns, pd.MultiIndex):
            return data[(ticker, field)].dropna()
        else:
            return data[field].dropna()
    
    def align_series(self, *series: pd.Series) -> List[pd.Series]:
        """
        Align multiple time series to common dates.
        
        Parameters
        ----------
        *series : pd.Series
            Variable number of series to align
            
        Returns
        -------
        list of pd.Series
            Aligned series
        """
        df = pd.concat(series, axis=1)
        df = df.dropna()
        return [df.iloc[:, i] for i in range(len(series))]


def load_market_data(tickers: Union[str, List[str]], 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    period: str = '5y',
                    data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Convenience function to load market data.
    
    Parameters
    ----------
    tickers : str or list of str
        Ticker symbols
    start_date : str, optional
        Start date (YYYY-MM-DD)
    end_date : str, optional
        End date (YYYY-MM-DD)
    period : str
        Period if start_date not specified
    data_dir : str
        Directory for raw data
        
    Returns
    -------
    pd.DataFrame
        Market data
    """
    loader = DataLoader(data_dir)
    return loader.download_yahoo(tickers, start_date, end_date, period)


def load_sample_data() -> Dict[str, pd.DataFrame]:
    """
    Load REAL market data with robust error handling.
    
    PRODUCTION STANDARD: Never silently fall back to synthetic data.
    Raise explicit errors if real data cannot be obtained.
    
    Returns
    -------
    dict
        Dictionary with 'prices' and 'returns' DataFrames
        
    Raises
    ------
    RuntimeError
        If real market data cannot be loaded after all attempts
    """
    tickers = ['SPY', 'QQQ', 'TLT']
    loader = DataLoader()
    
    # Attempt 1: Recent 5 years
    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1}/3: Downloading {tickers}...")
            data = loader.download_yahoo(tickers, period='5y')
            
            # Validate data structure
            if data is None or data.empty:
                raise ValueError("Empty data returned")
            
            # Extract prices with robust handling
            prices = pd.DataFrame()
            
            # Handle yfinance's inconsistent return formats
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker format: (Ticker, Field)
                for ticker in tickers:
                    try:
                        # Try standard format
                        if (ticker, 'Close') in data.columns:
                            series = data[(ticker, 'Close')].dropna()
                        elif ('Close', ticker) in data.columns:
                            series = data[('Close', ticker)].dropna()
                        else:
                            continue
                        
                        if len(series) > 0:
                            prices[ticker] = series
                    except Exception as e:
                        print(f"  Warning: Could not extract {ticker}: {e}")
                        continue
            else:
                # Single ticker or flat format
                if 'Close' in data.columns:
                    prices[tickers[0]] = data['Close'].dropna()
            
            # Validate we got usable data
            if prices.empty:
                raise ValueError("No price data extracted")
            
            if len(prices) < 252:  # Minimum 1 year
                raise ValueError(f"Insufficient data: {len(prices)} days < 252")
            
            # Check for excessive NaNs
            nan_pct = prices.isna().sum().sum() / (prices.shape[0] * prices.shape[1])
            if nan_pct > 0.05:
                raise ValueError(f"Excessive NaNs: {nan_pct:.1%}")
            
            # Calculate returns with validation
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # Validate returns
            if returns.isna().any().any():
                raise ValueError("NaNs in returns after calculation")
            
            if (returns.abs() > 0.5).any().any():
                raise ValueError("Extreme returns detected (>50% single day)")
            
            print(f"✓ Successfully loaded {len(prices)} days of data for {list(prices.columns)}")
            
            return {
                'prices': prices,
                'returns': returns
            }
            
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                import time
                time.sleep(2)  # Wait before retry
            continue
    
    # If all attempts failed, raise explicit error
    raise RuntimeError(
        "CRITICAL: Failed to load real market data after 3 attempts. "
        "Check internet connection and yfinance availability. "
        "DO NOT proceed with synthetic data in production."
    )


def generate_synthetic_data(n_samples: int = 1260, 
                           n_assets: int = 3,
                           seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic market data with regime switches.
    
    Parameters
    ----------
    n_samples : int
        Number of time periods (default 5 years of daily data)
    n_assets : int
        Number of assets
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Dictionary with 'prices' and 'returns' DataFrames
    """
    np.random.seed(seed)
    
    # Generate regime-switching returns
    # Regime 1: Low vol, positive drift
    # Regime 2: High vol, negative drift
    # Regime 3: Medium vol, mean-reverting
    
    regimes = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.2, 0.2])
    
    # Smooth regime transitions
    for i in range(1, n_samples):
        if np.random.rand() < 0.95:  # 95% persistence
            regimes[i] = regimes[i-1]
    
    returns = np.zeros((n_samples, n_assets))
    
    for i in range(n_samples):
        if regimes[i] == 0:  # Low vol, positive drift
            returns[i] = np.random.normal(0.0005, 0.01, n_assets)
        elif regimes[i] == 1:  # High vol, negative drift
            returns[i] = np.random.normal(-0.001, 0.025, n_assets)
        else:  # Mean-reverting
            if i > 0:
                returns[i] = -0.3 * returns[i-1] + np.random.normal(0, 0.015, n_assets)
            else:
                returns[i] = np.random.normal(0, 0.015, n_assets)
    
    # Convert to prices
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    # Create DataFrames
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    asset_names = [f'ASSET_{i+1}' for i in range(n_assets)]
    
    prices_df = pd.DataFrame(prices, index=dates, columns=asset_names)
    returns_df = pd.DataFrame(returns, index=dates, columns=asset_names)
    
    return {
        'prices': prices_df,
        'returns': returns_df,
        'regimes': regimes
    }


if __name__ == '__main__':
    # Test data loading
    print("Testing data loader...")
    data = load_sample_data()
    print(f"\nPrices shape: {data['prices'].shape}")
    print(f"Returns shape: {data['returns'].shape}")
    print(f"\nFirst few rows of prices:")
    print(data['prices'].head())
