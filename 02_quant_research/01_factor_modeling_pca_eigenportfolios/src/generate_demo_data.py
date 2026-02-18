"""
Generate Demo Data for Testing
Creates synthetic equity return data when Yahoo Finance is unavailable
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_demo_data(n_assets=50, n_days=2500, start_date='2015-01-01'):
    """
    Generate synthetic equity return data
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    n_days : int
        Number of trading days
    start_date : str
        Start date
    
    Returns:
    --------
    tuple : (prices, returns, market_returns)
    """
    logger.info(f"Generating demo data: {n_assets} assets, {n_days} days")
    
    # Generate dates
    start = pd.to_datetime(start_date)
    dates = pd.bdate_range(start=start, periods=n_days)
    
    # Generate asset names
    tickers = [f'STOCK{i:02d}' for i in range(n_assets)]
    
    # Generate factor structure
    n_factors = 5
    factor_returns = np.random.normal(0.0005, 0.015, (n_days, n_factors))
    
    # Generate factor loadings (betas)
    factor_loadings = np.random.normal(0, 0.5, (n_assets, n_factors))
    factor_loadings[:, 0] += 1.0  # Market factor
    
    # Generate returns from factor model
    systematic_returns = factor_returns @ factor_loadings.T
    
    # Add idiosyncratic noise
    idiosyncratic = np.random.normal(0, 0.01, (n_days, n_assets))
    
    # Total returns
    returns = systematic_returns + idiosyncratic
    
    # Convert to DataFrame
    returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
    
    # Generate prices from returns
    prices = (1 + returns_df).cumprod() * 100
    
    # Generate market returns (first factor)
    market_returns = pd.Series(factor_returns[:, 0], index=dates, name='Market')
    
    logger.info("Demo data generated successfully")
    
    return prices, returns_df, market_returns


if __name__ == "__main__":
    prices, returns, market = generate_demo_data()
    print(f"Prices shape: {prices.shape}")
    print(f"Returns shape: {returns.shape}")
    print(f"Market returns shape: {market.shape}")
