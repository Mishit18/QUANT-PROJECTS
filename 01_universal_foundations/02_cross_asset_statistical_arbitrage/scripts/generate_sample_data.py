"""
Generate synthetic OHLCV data for testing.
Replace with actual data loader in production.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_synthetic_ohlcv(ticker: str, start_date: str, end_date: str, 
                             initial_price: float = 100.0) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data."""
    dates = pd.date_range(start_date, end_date, freq='B')
    n_days = len(dates)
    
    np.random.seed(hash(ticker) % 2**32)
    
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    close = prices
    open_prices = close * (1 + np.random.normal(0, 0.005, n_days))
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    
    volume = np.random.lognormal(15, 1, n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df


def main():
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = [f'TICK{i:03d}' for i in range(100)]
    
    start_date = '2015-01-01'
    end_date = '2024-12-31'
    
    print(f"Generating synthetic data for {len(tickers)} tickers...")
    
    for ticker in tickers:
        df = generate_synthetic_ohlcv(ticker, start_date, end_date)
        output_path = output_dir / f'{ticker}.csv'
        df.to_csv(output_path, index=False)
    
    ticker_list = pd.DataFrame({'ticker': tickers})
    ticker_list.to_csv(output_dir / 'universe.csv', index=False)
    
    print(f"Generated {len(tickers)} ticker files in {output_dir}")


if __name__ == '__main__':
    main()
