"""
Data loading and preprocessing for volatility modeling.

Handles fetching market data, cleaning, and preparing returns.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from typing import Optional, Tuple


class DataLoader:
    """Load and preprocess financial market data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_data(
        self, 
        ticker: str = "^GSPC",
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Fetch price data from Yahoo Finance.
        
        Default: S&P 500 index for liquid, representative data.
        """
        cache_file = self.raw_dir / f"{ticker.replace('^', '')}_prices.csv"
        
        if cache_file.exists() and not force_download:
            print(f"Loading cached data from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Ensure numeric columns
            for col in df.columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        
        print(f"Downloading {ticker} from {start_date} to {end_date or 'present'}")
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")
        
        df.to_csv(cache_file)
        print(f"Saved to {cache_file}")
        
        return df
    
    def compute_returns(
        self, 
        prices: pd.DataFrame,
        price_col: str = "Adj Close",
        method: str = "log"
    ) -> pd.Series:
        """
        Compute returns from price series.
        
        Log returns are standard for volatility modeling:
        - Additive across time
        - Approximately normal for small changes
        - Symmetric treatment of gains/losses
        """
        if price_col not in prices.columns:
            price_col = "Close"
        
        p = prices[price_col].dropna()
        
        if method == "log":
            returns = np.log(p / p.shift(1))
        elif method == "simple":
            returns = p.pct_change()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        returns = returns.dropna()
        
        # Remove extreme outliers (likely data errors)
        # Use 10 sigma threshold - conservative for fat-tailed returns
        threshold = 10 * returns.std()
        outliers = np.abs(returns) > threshold
        
        if outliers.any():
            print(f"Warning: Removed {outliers.sum()} extreme outliers (>{threshold:.4f})")
            returns = returns[~outliers]
        
        return returns
    
    def prepare_dataset(
        self,
        ticker: str = "^GSPC",
        start_date: str = "2010-01-01",
        end_date: Optional[str] = None,
        return_method: str = "log"
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Complete data preparation pipeline.
        
        Returns:
            returns: Log returns series
            prices: Original price dataframe
        """
        prices = self.fetch_data(ticker, start_date, end_date)
        returns = self.compute_returns(prices, method=return_method)
        
        # Save processed returns
        returns_file = self.processed_dir / f"{ticker.replace('^', '')}_returns.csv"
        returns.to_csv(returns_file, header=True)
        print(f"Saved returns to {returns_file}")
        
        return returns, prices
    
    def load_processed_returns(self, ticker: str = "^GSPC") -> pd.Series:
        """Load previously processed returns."""
        returns_file = self.processed_dir / f"{ticker.replace('^', '')}_returns.csv"
        
        if not returns_file.exists():
            raise FileNotFoundError(f"No processed returns found at {returns_file}")
        
        returns = pd.read_csv(returns_file, index_col=0, parse_dates=True, squeeze=True)
        returns.name = "returns"
        
        return returns


def get_sample_data(ticker: str = "^GSPC", start: str = "2010-01-01") -> pd.Series:
    """Convenience function for quick data loading."""
    loader = DataLoader()
    returns, _ = loader.prepare_dataset(ticker=ticker, start_date=start)
    return returns
