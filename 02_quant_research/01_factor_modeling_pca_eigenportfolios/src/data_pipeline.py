"""
Data Pipeline for Equity Factor Modeling
Handles data acquisition, cleaning, and preprocessing
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path

from utils import (
    compute_returns, compute_excess_returns, winsorize_returns,
    detect_outliers, handle_missing_data, test_stationarity,
    save_results
)

logger = logging.getLogger(__name__)


class EquityDataPipeline:
    """
    Production-grade equity data pipeline
    Handles data acquisition, validation, and preprocessing
    """
    
    def __init__(self, config: dict):
        """
        Initialize data pipeline
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        self.prices = None
        self.returns = None
        self.excess_returns = None
        self.market_data = None
        
    def get_sp100_tickers(self) -> List[str]:
        """
        Get S&P 100 tickers (OEX components)
        For production, this would come from a data vendor
        """
        # Top 50 most liquid US equities (more reliable for download)
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'UNH', 'JNJ',
            'V', 'XOM', 'WMT', 'JPM', 'PG', 'MA', 'CVX', 'HD', 'MRK', 'ABBV',
            'KO', 'PEP', 'COST', 'AVGO', 'MCD', 'CSCO', 'TMO', 'ACN', 'ABT', 'LLY',
            'ADBE', 'NKE', 'DHR', 'VZ', 'NEE', 'TXN', 'PM', 'CMCSA', 'UPS', 'RTX',
            'ORCL', 'INTC', 'CRM', 'WFC', 'DIS', 'BMY', 'QCOM', 'HON', 'UNP', 'AMGN'
        ]
        
        logger.info(f"Using {len(tickers)} tickers from S&P 100 universe")
        return tickers
    
    def download_data(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance
        
        Parameters:
        -----------
        tickers : list
            List of ticker symbols
        
        Returns:
        --------
        pd.DataFrame : Adjusted close prices
        """
        logger.info(f"Downloading data for {len(tickers)} tickers...")
        
        start_date = self.data_config['start_date']
        end_date = self.data_config['end_date']
        
        try:
            # Download data
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Use adjusted prices
            )
            
            # Extract adjusted close
            if len(tickers) == 1:
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                prices = data['Close']
            
            logger.info(f"Downloaded {len(prices)} days of data")
            return prices
            
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            raise
    
    def download_market_data(self) -> pd.DataFrame:
        """
        Download market index data (S&P 500)
        
        Returns:
        --------
        pd.DataFrame : Market returns
        """
        logger.info("Downloading market index data...")
        
        try:
            spy = yf.download(
                'SPY',
                start=self.data_config['start_date'],
                end=self.data_config['end_date'],
                progress=False,
                auto_adjust=True
            )
            
            # Extract Close price - yfinance returns DataFrame
            if isinstance(spy, pd.DataFrame):
                if 'Close' in spy.columns:
                    market_prices = spy['Close']
                else:
                    market_prices = spy.iloc[:, 0]  # First column
            else:
                market_prices = spy
            
            # Ensure it's a Series
            if isinstance(market_prices, pd.DataFrame):
                market_prices = market_prices.squeeze()
            
            market_returns = compute_returns(
                pd.DataFrame(market_prices),
                self.data_config['return_type']
            )
            
            return market_returns.squeeze()
            
        except Exception as e:
            logger.error(f"Failed to download market data: {e}")
            raise
    
    def validate_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean price data
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Raw price data
        
        Returns:
        --------
        pd.DataFrame : Validated prices
        """
        logger.info("Validating data...")
        
        # Remove columns with insufficient history
        min_days = self.data_config['min_history_days']
        valid_count = prices.notna().sum()
        valid_tickers = valid_count[valid_count >= min_days].index
        prices_valid = prices[valid_tickers].copy()
        
        logger.info(f"Kept {len(valid_tickers)} tickers with sufficient history")
        
        # Handle missing data
        prices_clean = handle_missing_data(
            prices_valid,
            max_missing_pct=self.data_config['max_missing_pct'],
            method='drop'
        )
        
        # Check for non-positive prices
        if (prices_clean <= 0).any().any():
            logger.warning("Found non-positive prices, removing affected rows")
            prices_clean = prices_clean[(prices_clean > 0).all(axis=1)]
        
        logger.info(f"Final dataset: {len(prices_clean)} days, {len(prices_clean.columns)} assets")
        
        return prices_clean
    
    def compute_returns_pipeline(self, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute returns and excess returns
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Clean price data
        
        Returns:
        --------
        tuple : (returns, excess_returns)
        """
        logger.info("Computing returns...")
        
        # Compute returns
        returns = compute_returns(prices, self.data_config['return_type'])
        
        # Winsorize to handle outliers
        returns_clean = winsorize_returns(returns, limits=(0.01, 0.01))
        
        # Compute excess returns
        excess_returns = compute_excess_returns(
            returns_clean,
            rf_rate=self.data_config['risk_free_rate']
        )
        
        logger.info(f"Returns computed: {returns_clean.shape}")
        
        return returns_clean, excess_returns
    
    def run_diagnostics(self, returns: pd.DataFrame) -> dict:
        """
        Run data quality diagnostics
        
        Parameters:
        -----------
        returns : pd.DataFrame
            Return data
        
        Returns:
        --------
        dict : Diagnostic results
        """
        logger.info("Running data diagnostics...")
        
        diagnostics = {}
        
        # Summary statistics
        diagnostics['mean_return'] = returns.mean().mean() * 252
        diagnostics['mean_volatility'] = returns.std().mean() * np.sqrt(252)
        diagnostics['mean_sharpe'] = (returns.mean() / returns.std()).mean() * np.sqrt(252)
        
        # Cross-sectional correlation
        corr_matrix = returns.corr()
        diagnostics['mean_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Stationarity test (sample of assets)
        sample_assets = returns.columns[:5]
        stationarity_results = {}
        for asset in sample_assets:
            stat, pval = test_stationarity(returns[asset], test='adf')
            stationarity_results[asset] = {'statistic': stat, 'p_value': pval}
        diagnostics['stationarity_tests'] = stationarity_results
        
        # Outlier detection
        outliers = detect_outliers(returns, threshold=self.data_config['outlier_threshold'])
        diagnostics['outlier_pct'] = (outliers.sum().sum() / returns.size) * 100
        
        logger.info("Diagnostics complete")
        
        return diagnostics
    
    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        """
        Execute full data pipeline
        
        Returns:
        --------
        tuple : (prices, returns, excess_returns, diagnostics)
        """
        logger.info("=" * 60)
        logger.info("STARTING DATA PIPELINE")
        logger.info("=" * 60)
        
        # Get tickers
        tickers = self.get_sp100_tickers()
        
        # Download data
        try:
            prices_raw = self.download_data(tickers)
            
            # Check if download was successful
            if prices_raw.empty or len(prices_raw.columns) == 0:
                raise ValueError("No data downloaded from Yahoo Finance")
                
        except Exception as e:
            logger.warning(f"Yahoo Finance download failed: {e}")
            logger.info("Using demo data instead...")
            
            # Use demo data
            from generate_demo_data import generate_demo_data
            prices_raw, _, market_returns_demo = generate_demo_data(
                n_assets=50,
                n_days=2500,
                start_date=self.data_config['start_date']
            )
            self.market_data = market_returns_demo
        
        # Download market data if not already set
        if self.market_data is None:
            try:
                self.market_data = self.download_market_data()
            except:
                logger.warning("Market data download failed, using synthetic data")
                self.market_data = pd.Series(
                    np.random.normal(0.0005, 0.015, len(prices_raw)),
                    index=prices_raw.index
                )
        
        # Validate and clean
        self.prices = self.validate_data(prices_raw)
        
        # Compute returns
        self.returns, self.excess_returns = self.compute_returns_pipeline(self.prices)
        
        # Run diagnostics
        diagnostics = self.run_diagnostics(self.returns)
        
        # Save processed data
        save_results(self.prices, 'prices_clean.csv', self.config['paths']['data_processed'])
        save_results(self.returns, 'returns.csv', self.config['paths']['data_processed'])
        save_results(self.excess_returns, 'excess_returns.csv', self.config['paths']['data_processed'])
        save_results(self.market_data, 'market_returns.csv', self.config['paths']['data_processed'])
        save_results(diagnostics, 'data_diagnostics.json', self.config['paths']['results'])
        
        logger.info("=" * 60)
        logger.info("DATA PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return self.prices, self.returns, self.excess_returns, diagnostics


if __name__ == "__main__":
    from utils import load_config, ensure_directories
    
    # Load config
    config = load_config()
    ensure_directories(config)
    
    # Run pipeline
    pipeline = EquityDataPipeline(config)
    prices, returns, excess_returns, diagnostics = pipeline.run()
    
    print("\nData Pipeline Summary:")
    print(f"Assets: {len(returns.columns)}")
    print(f"Time period: {returns.index[0]} to {returns.index[-1]}")
    print(f"Mean annual return: {diagnostics['mean_return']:.2%}")
    print(f"Mean volatility: {diagnostics['mean_volatility']:.2%}")
    print(f"Mean Sharpe ratio: {diagnostics['mean_sharpe']:.2f}")
