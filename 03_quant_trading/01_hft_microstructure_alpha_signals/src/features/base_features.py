"""
Base microstructure features: returns, depth, volatility, intensity.
"""

import numpy as np
import pandas as pd
from typing import List


def compute_returns(df: pd.DataFrame, lags: List[int] = [1, 2, 5, 10]) -> pd.DataFrame:
    """
    Compute lagged returns of mid price.
    
    Returns at lag k use information from k events ago, avoiding lookahead.
    """
    features = pd.DataFrame(index=df.index)
    
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    
    for lag in lags:
        features[f'return_lag_{lag}'] = mid_price.pct_change(lag)
    
    return features


def compute_depth_features(df: pd.DataFrame, n_levels: int = 5) -> pd.DataFrame:
    """
    Aggregate depth across multiple levels.
    
    Total depth on each side provides liquidity measure.
    """
    features = pd.DataFrame(index=df.index)
    
    # Total depth on bid and ask side
    bid_depth = sum(df[f'bid_size_{i}'] for i in range(1, n_levels + 1))
    ask_depth = sum(df[f'ask_size_{i}'] for i in range(1, n_levels + 1))
    
    features['total_bid_depth'] = bid_depth
    features['total_ask_depth'] = ask_depth
    features['total_depth'] = bid_depth + ask_depth
    
    # Depth imbalance
    features['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-8)
    
    return features


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling volatility of mid price returns.
    
    Uses realized volatility over recent window.
    """
    features = pd.DataFrame(index=df.index)
    
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    returns = mid_price.pct_change()
    
    # Rolling standard deviation
    features['volatility'] = returns.rolling(window=window, min_periods=1).std()
    
    # Rolling range (high-low)
    features['price_range'] = (
        mid_price.rolling(window=window, min_periods=1).max() - 
        mid_price.rolling(window=window, min_periods=1).min()
    )
    
    return features


def compute_event_intensity(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """
    Event arrival rate (events per second).
    
    High intensity indicates active periods with more information flow.
    """
    features = pd.DataFrame(index=df.index)
    
    timestamps = df['timestamp'].values
    intensity = np.zeros(len(df))
    
    for i in range(window, len(df)):
        window_timestamps = timestamps[i-window:i]
        time_span = (window_timestamps[-1] - window_timestamps[0]) / 1e9
        
        if time_span > 0:
            intensity[i] = window / time_span
    
    # Forward fill initial values
    if window < len(intensity):
        intensity[:window] = intensity[window]
    
    features['event_intensity'] = intensity
    
    return features


def compute_all_base_features(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Compute all base features.
    
    Args:
        df: LOB data
        config: Feature configuration dict
        
    Returns:
        DataFrame with all base features
    """
    if config is None:
        config = {
            'return_lags': [1, 2, 5, 10],
            'volatility_window': 20,
            'depth_levels': 5
        }
    
    features_list = []
    
    # Returns
    features_list.append(compute_returns(df, lags=config['return_lags']))
    
    # Depth
    features_list.append(compute_depth_features(df, n_levels=config['depth_levels']))
    
    # Volatility
    features_list.append(compute_volatility(df, window=config['volatility_window']))
    
    # Event intensity
    features_list.append(compute_event_intensity(df))
    
    # Concatenate all features
    features = pd.concat(features_list, axis=1)
    
    return features


if __name__ == "__main__":
    # Test with sample data
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Computing base features...")
    features = compute_all_base_features(df)
    
    print(f"\nFeature shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()}")
    print("\nSample features:")
    print(features.head(10))
    print("\nFeature statistics:")
    print(features.describe())
