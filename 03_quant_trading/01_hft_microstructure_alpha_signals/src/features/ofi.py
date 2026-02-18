"""
Order Flow Imbalance (OFI) features.

OFI captures net aggressive order flow by tracking changes in depth
at each price level. Positive OFI indicates buying pressure.

Reference: Cont, Kukanov, Stoikov (2014)
"""

import numpy as np
import pandas as pd
from typing import List


def compute_ofi_single_level(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Compute OFI for a single book level.
    
    OFI_t = ΔBid_size * I(ΔBid_price >= 0) - ΔBid_size * I(ΔBid_price < 0)
          - ΔAsk_size * I(ΔAsk_price <= 0) + ΔAsk_size * I(ΔAsk_price > 0)
    
    Intuition: 
    - Bid size increases at same/higher price → buying pressure
    - Ask size increases at same/lower price → selling pressure
    """
    bid_price_col = f'bid_price_{level}'
    bid_size_col = f'bid_size_{level}'
    ask_price_col = f'ask_price_{level}'
    ask_size_col = f'ask_size_{level}'
    
    # Changes in price and size
    d_bid_price = df[bid_price_col].diff()
    d_bid_size = df[bid_size_col].diff()
    d_ask_price = df[ask_price_col].diff()
    d_ask_size = df[ask_size_col].diff()
    
    # OFI components
    bid_component = np.where(d_bid_price >= 0, d_bid_size, -d_bid_size)
    ask_component = np.where(d_ask_price <= 0, -d_ask_size, d_ask_size)
    
    ofi = bid_component + ask_component
    
    return pd.Series(ofi, index=df.index)


def compute_ofi_multi_level(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Compute aggregated OFI across multiple levels.
    
    Weights levels by inverse distance from mid (closer levels matter more).
    """
    ofi_total = pd.Series(0.0, index=df.index)
    
    for level in range(1, n_levels + 1):
        weight = 1.0 / level  # closer levels have higher weight
        ofi_level = compute_ofi_single_level(df, level=level)
        ofi_total += weight * ofi_level
    
    return ofi_total


def compute_ofi_features(df: pd.DataFrame, n_levels: int = 5, 
                        lookback_ticks: List[int] = [1, 5, 10]) -> pd.DataFrame:
    """
    Compute OFI features with multiple lookback windows.
    
    Args:
        df: LOB data
        n_levels: Number of book levels to include
        lookback_ticks: List of lookback windows for cumulative OFI
        
    Returns:
        DataFrame with OFI features
    """
    features = pd.DataFrame(index=df.index)
    
    # Instantaneous OFI (single tick)
    ofi_instant = compute_ofi_multi_level(df, n_levels=n_levels)
    features['ofi_instant'] = ofi_instant
    
    # Cumulative OFI over lookback windows
    for lookback in lookback_ticks:
        features[f'ofi_cum_{lookback}'] = ofi_instant.rolling(
            window=lookback, min_periods=1
        ).sum()
    
    # OFI per level (for feature importance analysis)
    for level in range(1, min(n_levels, 3) + 1):  # top 3 levels
        features[f'ofi_level_{level}'] = compute_ofi_single_level(df, level=level)
    
    return features


def compute_ofi_imbalance_ratio(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    OFI normalized by total depth.
    
    Captures relative order flow pressure accounting for book depth.
    """
    ofi = compute_ofi_multi_level(df, n_levels=n_levels)
    
    # Total depth
    total_depth = sum(
        df[f'bid_size_{i}'] + df[f'ask_size_{i}'] 
        for i in range(1, n_levels + 1)
    )
    
    # Normalize
    ofi_ratio = ofi / (total_depth + 1e-8)
    
    return ofi_ratio


if __name__ == "__main__":
    # Test OFI computation
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Computing OFI features...")
    ofi_features = compute_ofi_features(df, n_levels=5, lookback_ticks=[1, 5, 10])
    
    print(f"\nOFI feature shape: {ofi_features.shape}")
    print(f"OFI columns: {ofi_features.columns.tolist()}")
    print("\nOFI statistics:")
    print(ofi_features.describe())
    
    # Check correlation with future returns
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    future_return = mid_price.pct_change(5).shift(-5)
    
    print("\nCorrelation with 5-tick future return:")
    for col in ofi_features.columns:
        corr = ofi_features[col].corr(future_return)
        print(f"{col}: {corr:.4f}")
