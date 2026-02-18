"""
Spread dynamics features.

Spread behavior reveals liquidity conditions and information events.
Widening spreads often precede volatility or informed trading.
"""

import numpy as np
import pandas as pd
from typing import List


def compute_spread_features(df: pd.DataFrame, 
                           lookback_ticks: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """
    Compute spread-related features.
    
    Args:
        df: LOB data
        lookback_ticks: Lookback windows for spread statistics
        
    Returns:
        DataFrame with spread features
    """
    features = pd.DataFrame(index=df.index)
    
    # Absolute spread
    spread = df['ask_price_1'] - df['bid_price_1']
    features['spread'] = spread
    
    # Relative spread (normalized by mid price)
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    features['relative_spread'] = spread / (mid_price + 1e-8)
    
    # Spread in ticks
    tick_size = 0.01  # assume from config
    features['spread_ticks'] = spread / tick_size
    
    # Spread changes (expansion/compression)
    features['spread_change'] = spread.diff()
    features['spread_pct_change'] = spread.pct_change()
    
    # Rolling spread statistics
    for lookback in lookback_ticks:
        # Mean spread
        features[f'spread_mean_{lookback}'] = spread.rolling(
            window=lookback, min_periods=1
        ).mean()
        
        # Spread volatility
        features[f'spread_std_{lookback}'] = spread.rolling(
            window=lookback, min_periods=1
        ).std()
        
        # Spread range
        spread_max = spread.rolling(window=lookback, min_periods=1).max()
        spread_min = spread.rolling(window=lookback, min_periods=1).min()
        features[f'spread_range_{lookback}'] = spread_max - spread_min
    
    # Spread momentum (rate of change)
    features['spread_momentum'] = spread.diff(5)
    
    # Spread relative to recent average
    spread_ma_20 = spread.rolling(window=20, min_periods=1).mean()
    features['spread_vs_ma'] = (spread - spread_ma_20) / (spread_ma_20 + 1e-8)
    
    return features


def compute_effective_spread(df: pd.DataFrame) -> pd.Series:
    """
    Effective spread based on trade prices.
    
    Effective spread = 2 * |trade_price - mid_price|
    
    Measures actual cost of trading (realized spread).
    """
    is_trade = df['event_type'] == 'trade'
    
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    trade_price = df['trade_price']
    
    effective_spread = pd.Series(np.nan, index=df.index)
    effective_spread[is_trade] = 2 * np.abs(trade_price[is_trade] - mid_price[is_trade])
    
    # Forward fill for non-trade events
    effective_spread = effective_spread.fillna(method='ffill')
    
    return effective_spread


def compute_quoted_spread_depth_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Ratio of spread to depth at best levels.
    
    High ratio indicates thin liquidity (wide spread, low depth).
    """
    spread = df['ask_price_1'] - df['bid_price_1']
    depth = df['bid_size_1'] + df['ask_size_1']
    
    ratio = spread / (depth + 1e-8)
    
    return ratio


def compute_spread_asymmetry(df: pd.DataFrame) -> pd.Series:
    """
    Asymmetry in bid vs ask side spread contribution.
    
    Measures if spread widening is driven more by bid or ask side.
    """
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    
    bid_distance = mid_price - df['bid_price_1']
    ask_distance = df['ask_price_1'] - mid_price
    
    # Asymmetry: positive if ask side wider, negative if bid side wider
    asymmetry = (ask_distance - bid_distance) / (ask_distance + bid_distance + 1e-8)
    
    return asymmetry


if __name__ == "__main__":
    # Test spread features
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Computing spread features...")
    spread_features = compute_spread_features(df, lookback_ticks=[5, 10, 20])
    
    # Add additional spread features
    spread_features['effective_spread'] = compute_effective_spread(df)
    spread_features['spread_depth_ratio'] = compute_quoted_spread_depth_ratio(df)
    spread_features['spread_asymmetry'] = compute_spread_asymmetry(df)
    
    print(f"\nSpread feature shape: {spread_features.shape}")
    print(f"Spread columns: {spread_features.columns.tolist()}")
    print("\nSpread statistics:")
    print(spread_features.describe())
    
    # Check correlation with future returns
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    future_return = mid_price.pct_change(5).shift(-5)
    
    print("\nCorrelation with 5-tick future return:")
    for col in ['spread', 'relative_spread', 'spread_change', 'spread_asymmetry']:
        if col in spread_features.columns:
            corr = spread_features[col].corr(future_return)
            print(f"{col}: {corr:.4f}")
