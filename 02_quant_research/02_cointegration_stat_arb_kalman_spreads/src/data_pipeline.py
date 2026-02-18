"""
Data pipeline for downloading, cleaning, and preprocessing price data.

This module handles all data acquisition and preprocessing steps including:
- Downloading price data from Yahoo Finance
- Handling missing data and corporate actions
- Aligning series to common dates
- Train/test splitting
- Data quality validation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging
from pathlib import Path

from utils import (
    setup_logging, 
    ensure_dir, 
    validate_price_data,
    save_results
)


logger = setup_logging()


class DataPipeline:
    """
    Data pipeline for statistical arbitrage research.
    
    Handles data download, cleaning, alignment, and preprocessing.
    """
    
    def __init__(self, 
                 raw_data_dir: str = "data/raw",
                 processed_data_dir: str = "data/processed"):
        """
        Initialize data pipeline.
        
        Args:
            raw_data_dir: Directory for raw data storage
            processed_data_dir: Directory for processed data storage
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        ensure_dir(raw_data_dir)
        ensure_dir(processed_data_dir)
        
        self.raw_prices = None
        self.processed_prices = None
        self.metadata = {}
    
    def download_data(self,
                     tickers: List[str],
                     start_date: str,
                     end_date: str,
                     force_download: bool = False) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_download: Force re-download even if cached data exists
            
        Returns:
            DataFrame with adjusted close prices
        """
        cache_file = f"{self.raw_data_dir}/prices_{start_date}_{end_date}.parquet"
        
        # Check cache
        if not force_download and Path(cache_file).exists():
            logger.info(f"Loading cached data from {cache_file}")
            self.raw_prices = pd.read_parquet(cache_file)
            return self.raw_prices
        
        logger.info(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Download data
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # Use adjusted prices
        )
        
        # Handle single vs multiple tickers
        if len(tickers) == 1:
            prices = data['Close'].to_frame()
            prices.columns = tickers
        else:
            prices = data['Close']
        
        # Store metadata
        self.metadata['download_date'] = datetime.now().isoformat()
        self.metadata['tickers'] = tickers
        self.metadata['start_date'] = start_date
        self.metadata['end_date'] = end_date
        self.metadata['raw_observations'] = len(prices)
        
        # Save raw data
        prices.to_parquet(cache_file)
        logger.info(f"Downloaded {len(prices)} observations for {len(tickers)} tickers")
        
        self.raw_prices = prices
        return prices
    
    def handle_missing_data(self,
                           prices: pd.DataFrame,
                           method: str = "forward_fill",
                           max_gap: int = 5,
                           max_missing_pct: float = 0.05) -> pd.DataFrame:
        """
        Handle missing data in price series.
        
        Args:
            prices: DataFrame with price data
            method: Method for handling missing data ("forward_fill", "drop")
            max_gap: Maximum gap size to fill (days)
            max_missing_pct: Maximum percentage of missing data allowed
            
        Returns:
            DataFrame with missing data handled
        """
        logger.info("Handling missing data")
        
        # Calculate missing data percentage
        missing_pct = prices.isnull().sum() / len(prices)
        
        # Remove tickers with too much missing data
        tickers_to_drop = missing_pct[missing_pct > max_missing_pct].index.tolist()
        if tickers_to_drop:
            logger.warning(f"Dropping tickers with >{max_missing_pct:.1%} missing data: {tickers_to_drop}")
            prices = prices.drop(columns=tickers_to_drop)
        
        if method == "forward_fill":
            # Forward fill small gaps
            prices = prices.fillna(method='ffill', limit=max_gap)
            
            # Drop remaining NaN values
            prices = prices.dropna()
            
        elif method == "drop":
            # Simply drop all rows with any NaN
            prices = prices.dropna()
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"After handling missing data: {len(prices)} observations, {len(prices.columns)} tickers")
        
        return prices
    
    def detect_outliers(self,
                       prices: pd.DataFrame,
                       threshold: float = 5.0) -> pd.DataFrame:
        """
        Detect and flag outliers in return series.
        
        Args:
            prices: DataFrame with price data
            threshold: Number of standard deviations for outlier detection
            
        Returns:
            DataFrame with outlier information
        """
        returns = prices.pct_change()
        
        outliers = pd.DataFrame(index=prices.index, columns=prices.columns, data=False)
        
        for col in returns.columns:
            mean = returns[col].mean()
            std = returns[col].std()
            
            # Flag returns beyond threshold
            outliers[col] = np.abs(returns[col] - mean) > (threshold * std)
        
        # Log outlier summary
        outlier_counts = outliers.sum()
        if outlier_counts.sum() > 0:
            logger.warning(f"Detected outliers: {outlier_counts[outlier_counts > 0].to_dict()}")
        
        return outliers
    
    def align_to_common_dates(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all series have data on common trading dates.
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            DataFrame with aligned data
        """
        # Remove rows where any ticker has NaN
        aligned = prices.dropna()
        
        logger.info(f"Aligned to {len(aligned)} common trading dates")
        
        return aligned
    
    def preprocess(self,
                  handle_missing: bool = True,
                  detect_outliers: bool = True,
                  min_observations: int = 252) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            handle_missing: Whether to handle missing data
            detect_outliers: Whether to detect outliers
            min_observations: Minimum required observations
            
        Returns:
            Preprocessed DataFrame
        """
        if self.raw_prices is None:
            raise ValueError("No raw data loaded. Call download_data() first.")
        
        logger.info("Starting preprocessing pipeline")
        
        prices = self.raw_prices.copy()
        
        # Handle missing data
        if handle_missing:
            prices = self.handle_missing_data(prices)
        
        # Align to common dates
        prices = self.align_to_common_dates(prices)
        
        # Detect outliers (for logging only, don't remove)
        if detect_outliers:
            outliers = self.detect_outliers(prices)
            self.metadata['outliers'] = outliers.sum().to_dict()
        
        # Validate minimum observations
        if len(prices) < min_observations:
            raise ValueError(
                f"Insufficient data after preprocessing: {len(prices)} < {min_observations}"
            )
        
        # Validate data quality
        is_valid, issues = validate_price_data(prices)
        if not is_valid:
            logger.warning(f"Data quality issues detected: {issues}")
        
        # Update metadata
        self.metadata['processed_observations'] = len(prices)
        self.metadata['processed_tickers'] = prices.columns.tolist()
        self.metadata['date_range'] = (
            prices.index.min().strftime('%Y-%m-%d'),
            prices.index.max().strftime('%Y-%m-%d')
        )
        
        # Save processed data
        output_file = f"{self.processed_data_dir}/processed_prices.parquet"
        prices.to_parquet(output_file)
        logger.info(f"Saved processed data to {output_file}")
        
        # Save metadata
        save_results(
            self.metadata,
            "data_metadata.yaml",
            self.processed_data_dir
        )
        
        self.processed_prices = prices
        return prices
    
    def train_test_split(self,
                        prices: Optional[pd.DataFrame] = None,
                        train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            prices: DataFrame with price data (uses processed_prices if None)
            train_ratio: Ratio of data for training
            
        Returns:
            Tuple of (train_prices, test_prices)
        """
        if prices is None:
            if self.processed_prices is None:
                raise ValueError("No processed data available")
            prices = self.processed_prices
        
        split_idx = int(len(prices) * train_ratio)
        
        train_prices = prices.iloc[:split_idx]
        test_prices = prices.iloc[split_idx:]
        
        logger.info(f"Train set: {len(train_prices)} observations ({train_prices.index[0]} to {train_prices.index[-1]})")
        logger.info(f"Test set: {len(test_prices)} observations ({test_prices.index[0]} to {test_prices.index[-1]})")
        
        # Save splits
        train_prices.to_parquet(f"{self.processed_data_dir}/train_prices.parquet")
        test_prices.to_parquet(f"{self.processed_data_dir}/test_prices.parquet")
        
        return train_prices, test_prices
    
    def get_pair_data(self,
                     ticker1: str,
                     ticker2: str,
                     prices: Optional[pd.DataFrame] = None) -> Tuple[pd.Series, pd.Series]:
        """
        Extract price series for a specific pair.
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            prices: DataFrame with price data (uses processed_prices if None)
            
        Returns:
            Tuple of (series1, series2)
        """
        if prices is None:
            if self.processed_prices is None:
                raise ValueError("No processed data available")
            prices = self.processed_prices
        
        if ticker1 not in prices.columns or ticker2 not in prices.columns:
            raise ValueError(f"Tickers not found in data: {ticker1}, {ticker2}")
        
        return prices[ticker1], prices[ticker2]
    
    def compute_log_prices(self,
                          prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute log prices.
        
        Args:
            prices: DataFrame with price data (uses processed_prices if None)
            
        Returns:
            DataFrame with log prices
        """
        if prices is None:
            if self.processed_prices is None:
                raise ValueError("No processed data available")
            prices = self.processed_prices
        
        log_prices = np.log(prices)
        return log_prices
    
    def get_summary_statistics(self,
                              prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute summary statistics for price data.
        
        Args:
            prices: DataFrame with price data (uses processed_prices if None)
            
        Returns:
            DataFrame with summary statistics
        """
        if prices is None:
            if self.processed_prices is None:
                raise ValueError("No processed data available")
            prices = self.processed_prices
        
        returns = prices.pct_change().dropna()
        
        stats = pd.DataFrame({
            'mean_price': prices.mean(),
            'std_price': prices.std(),
            'min_price': prices.min(),
            'max_price': prices.max(),
            'mean_return': returns.mean() * 252,  # Annualized
            'vol_return': returns.std() * np.sqrt(252),  # Annualized
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'observations': len(prices)
        })
        
        return stats
