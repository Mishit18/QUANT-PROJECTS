"""
Microprice estimator.

Microprice is a liquidity-weighted midprice that provides a better
estimate of "fair value" when the book is imbalanced.

Reference: Stoikov (2018)
"""

import numpy as np
import pandas as pd


def compute_microprice(df: pd.DataFrame, n_levels: int = 1) -> pd.Series:
    """
    Compute microprice using best bid/ask.
    
    Microprice = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size)
    
    Intuition: Weighted average of bid and ask prices by opposite side depth.
    When bid depth >> ask depth, microprice closer to ask (expect uptick).
    """
    if n_levels == 1:
        bid_price = df['bid_price_1']
        bid_size = df['bid_size_1']
        ask_price = df['ask_price_1']
        ask_size = df['ask_size_1']
        
        microprice = (bid_size * ask_price + ask_size * bid_price) / (bid_size + ask_size + 1e-8)
    else:
        # Multi-level microprice (weighted by depth at each level)
        numerator = 0.0
        denominator = 0.0
        
        for level in range(1, n_levels + 1):
            bid_price = df[f'bid_price_{level}']
            bid_size = df[f'bid_size_{level}']
            ask_price = df[f'ask_price_{level}']
            ask_size = df[f'ask_size_{level}']
            
            numerator += bid_size * ask_price + ask_size * bid_price
            denominator += bid_size + ask_size
        
        microprice = numerator / (denominator + 1e-8)
    
    return microprice


def compute_microprice_features(df: pd.DataFrame, n_levels: int = 3) -> pd.DataFrame:
    """
    Compute microprice and derived features.
    
    Args:
        df: LOB data
        n_levels: Number of levels to use in microprice calculation
        
    Returns:
        DataFrame with microprice features
    """
    features = pd.DataFrame(index=df.index)
    
    # Microprice
    microprice = compute_microprice(df, n_levels=n_levels)
    features['microprice'] = microprice
    
    # Mid price for comparison
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    features['mid_price'] = mid_price
    
    # Microprice deviation from mid
    # Positive deviation suggests buying pressure
    features['microprice_deviation'] = (microprice - mid_price) / (mid_price + 1e-8)
    
    # Microprice returns
    features['microprice_return'] = microprice.pct_change()
    
    # Microprice momentum (change in deviation)
    features['microprice_momentum'] = features['microprice_deviation'].diff()
    
    # Distance to microprice from best bid/ask
    features['bid_to_microprice'] = (microprice - df['bid_price_1']) / (mid_price + 1e-8)
    features['ask_to_microprice'] = (df['ask_price_1'] - microprice) / (mid_price + 1e-8)
    
    return features


def compute_volume_weighted_price(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Volume-weighted average price across book levels.
    
    Alternative to microprice that weights by depth at each level.
    """
    total_volume = 0.0
    weighted_price = 0.0
    
    for level in range(1, n_levels + 1):
        bid_price = df[f'bid_price_{level}']
        bid_size = df[f'bid_size_{level}']
        ask_price = df[f'ask_price_{level}']
        ask_size = df[f'ask_size_{level}']
        
        weighted_price += bid_price * bid_size + ask_price * ask_size
        total_volume += bid_size + ask_size
    
    vwap = weighted_price / (total_volume + 1e-8)
    
    return vwap


if __name__ == "__main__":
    # Test microprice computation
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Computing microprice features...")
    microprice_features = compute_microprice_features(df, n_levels=3)
    
    print(f"\nMicroprice feature shape: {microprice_features.shape}")
    print(f"Microprice columns: {microprice_features.columns.tolist()}")
    print("\nMicroprice statistics:")
    print(microprice_features.describe())
    
    # Compare microprice vs mid price
    print("\nMicroprice vs Mid Price:")
    print(f"Mean microprice deviation: {microprice_features['microprice_deviation'].mean():.6f}")
    print(f"Std microprice deviation: {microprice_features['microprice_deviation'].std():.6f}")
    
    # Check correlation with future returns
    mid_price = microprice_features['mid_price']
    future_return = mid_price.pct_change(5).shift(-5)
    
    print("\nCorrelation with 5-tick future return:")
    for col in ['microprice_deviation', 'microprice_momentum', 'bid_to_microprice']:
        if col in microprice_features.columns:
            corr = microprice_features[col].corr(future_return)
            print(f"{col}: {corr:.4f}")
