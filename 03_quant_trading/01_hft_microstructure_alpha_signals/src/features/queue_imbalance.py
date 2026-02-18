"""
Queue Imbalance (QI) features.

Queue imbalance measures relative depth on bid vs ask side.
Deep bid queue suggests upward pressure; deep ask queue suggests downward pressure.
"""

import numpy as np
import pandas as pd
from typing import List


def compute_queue_imbalance_level(df: pd.DataFrame, level: int = 1) -> pd.Series:
    """
    Compute queue imbalance at a specific level.
    
    QI = (bid_size - ask_size) / (bid_size + ask_size)
    
    Range: [-1, 1]
    - QI > 0: More depth on bid (buying pressure)
    - QI < 0: More depth on ask (selling pressure)
    """
    bid_size = df[f'bid_size_{level}']
    ask_size = df[f'ask_size_{level}']
    
    qi = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
    
    return qi


def compute_queue_imbalance_multi_level(df: pd.DataFrame, 
                                       n_levels: int = 5) -> pd.Series:
    """
    Aggregate queue imbalance across multiple levels.
    
    Uses depth-weighted average across levels.
    """
    total_bid = sum(df[f'bid_size_{i}'] for i in range(1, n_levels + 1))
    total_ask = sum(df[f'ask_size_{i}'] for i in range(1, n_levels + 1))
    
    qi = (total_bid - total_ask) / (total_bid + total_ask + 1e-8)
    
    return qi


def compute_weighted_queue_imbalance(df: pd.DataFrame, 
                                    n_levels: int = 5) -> pd.Series:
    """
    Queue imbalance with exponential distance weighting.
    
    Closer levels receive higher weight since they're more likely to trade.
    """
    qi_weighted = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    
    for level in range(1, n_levels + 1):
        weight = np.exp(-0.5 * (level - 1))  # exponential decay
        qi_level = compute_queue_imbalance_level(df, level=level)
        qi_weighted += weight * qi_level
        total_weight += weight
    
    qi_weighted /= total_weight
    
    return qi_weighted


def compute_queue_imbalance_features(df: pd.DataFrame, 
                                    n_levels_list: List[int] = [1, 3, 5]) -> pd.DataFrame:
    """
    Compute queue imbalance features at multiple aggregation levels.
    
    Args:
        df: LOB data
        n_levels_list: List of level counts to aggregate over
        
    Returns:
        DataFrame with QI features
    """
    features = pd.DataFrame(index=df.index)
    
    # QI at best level (most important)
    features['qi_level_1'] = compute_queue_imbalance_level(df, level=1)
    
    # QI aggregated over multiple levels
    for n_levels in n_levels_list:
        features[f'qi_levels_{n_levels}'] = compute_queue_imbalance_multi_level(
            df, n_levels=n_levels
        )
    
    # Weighted QI (exponential distance weighting)
    features['qi_weighted'] = compute_weighted_queue_imbalance(df, n_levels=5)
    
    # QI momentum (change in QI)
    features['qi_momentum'] = features['qi_level_1'].diff()
    
    # QI volatility (rolling std)
    features['qi_volatility'] = features['qi_level_1'].rolling(
        window=20, min_periods=1
    ).std()
    
    return features


def compute_depth_ratio(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Simple depth ratio: total_bid_depth / total_ask_depth.
    
    Alternative to QI that preserves magnitude information.
    """
    total_bid = sum(df[f'bid_size_{i}'] for i in range(1, n_levels + 1))
    total_ask = sum(df[f'ask_size_{i}'] for i in range(1, n_levels + 1))
    
    ratio = total_bid / (total_ask + 1e-8)
    
    return ratio


if __name__ == "__main__":
    # Test QI computation
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Computing Queue Imbalance features...")
    qi_features = compute_queue_imbalance_features(df, n_levels_list=[1, 3, 5])
    
    print(f"\nQI feature shape: {qi_features.shape}")
    print(f"QI columns: {qi_features.columns.tolist()}")
    print("\nQI statistics:")
    print(qi_features.describe())
    
    # Check correlation with future returns
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    future_return = mid_price.pct_change(5).shift(-5)
    
    print("\nCorrelation with 5-tick future return:")
    for col in qi_features.columns:
        corr = qi_features[col].corr(future_return)
        print(f"{col}: {corr:.4f}")
