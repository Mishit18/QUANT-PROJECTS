"""
Trade sign classification and signed volume features.

Uses Lee-Ready algorithm to classify trades as buyer or seller initiated.
Signed volume aggregation reveals informed order flow.
"""

import numpy as np
import pandas as pd
from typing import List


def classify_trade_sign_lee_ready(df: pd.DataFrame) -> pd.Series:
    """
    Classify trade direction using Lee-Ready algorithm.
    
    Rules:
    1. If trade_price > mid_price: buyer-initiated (+1)
    2. If trade_price < mid_price: seller-initiated (-1)
    3. If trade_price == mid_price: use tick rule (compare to previous trade)
    
    Returns:
        Series with values {-1, 0, +1}
    """
    trade_sign = pd.Series(0, index=df.index)
    
    # Only classify trade events
    is_trade = df['event_type'] == 'trade'
    
    if not is_trade.any():
        return trade_sign
    
    # Mid price
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    
    # Quote rule: compare trade price to mid
    trade_price = df['trade_price']
    
    trade_sign[is_trade & (trade_price > mid_price)] = 1  # buy
    trade_sign[is_trade & (trade_price < mid_price)] = -1  # sell
    
    # Tick rule for trades at mid: compare to previous trade price
    at_mid = is_trade & (trade_price == mid_price)
    if at_mid.any():
        prev_trade_price = trade_price.fillna(method='ffill')
        trade_sign[at_mid & (trade_price > prev_trade_price)] = 1
        trade_sign[at_mid & (trade_price < prev_trade_price)] = -1
    
    return trade_sign


def compute_signed_volume(df: pd.DataFrame, trade_sign: pd.Series) -> pd.Series:
    """
    Compute signed volume: trade_size * trade_sign.
    
    Positive signed volume indicates net buying; negative indicates net selling.
    """
    signed_volume = pd.Series(0.0, index=df.index)
    
    is_trade = df['event_type'] == 'trade'
    signed_volume[is_trade] = df.loc[is_trade, 'trade_size'] * trade_sign[is_trade]
    
    return signed_volume


def compute_trade_sign_features(df: pd.DataFrame, 
                               lookback_ticks: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Compute trade sign and signed volume features.
    
    Args:
        df: LOB data with trade events
        lookback_ticks: Lookback windows for cumulative signed volume
        
    Returns:
        DataFrame with trade sign features
    """
    features = pd.DataFrame(index=df.index)
    
    # Classify trade sign
    trade_sign = classify_trade_sign_lee_ready(df)
    features['trade_sign'] = trade_sign
    
    # Signed volume
    signed_volume = compute_signed_volume(df, trade_sign)
    features['signed_volume'] = signed_volume
    
    # Cumulative signed volume over lookback windows
    for lookback in lookback_ticks:
        features[f'signed_volume_cum_{lookback}'] = signed_volume.rolling(
            window=lookback, min_periods=1
        ).sum()
    
    # Trade intensity (number of trades in window)
    is_trade = (df['event_type'] == 'trade').astype(int)
    for lookback in [10, 20, 50]:
        features[f'trade_count_{lookback}'] = is_trade.rolling(
            window=lookback, min_periods=1
        ).sum()
    
    # Buy/sell imbalance
    buy_volume = signed_volume.clip(lower=0)
    sell_volume = -signed_volume.clip(upper=0)
    
    for lookback in [10, 20]:
        buy_vol_cum = buy_volume.rolling(window=lookback, min_periods=1).sum()
        sell_vol_cum = sell_volume.rolling(window=lookback, min_periods=1).sum()
        
        features[f'buy_sell_imbalance_{lookback}'] = (
            (buy_vol_cum - sell_vol_cum) / (buy_vol_cum + sell_vol_cum + 1e-8)
        )
    
    return features


def compute_vwap_trade(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Volume-weighted average trade price over recent window.
    
    Useful for comparing current price to recent trade activity.
    """
    is_trade = df['event_type'] == 'trade'
    
    trade_price = df['trade_price'].copy()
    trade_size = df['trade_size'].copy()
    
    # Fill non-trade events with 0
    trade_price[~is_trade] = 0
    trade_size[~is_trade] = 0
    
    # Rolling VWAP
    price_volume = trade_price * trade_size
    vwap = (
        price_volume.rolling(window=lookback, min_periods=1).sum() /
        (trade_size.rolling(window=lookback, min_periods=1).sum() + 1e-8)
    )
    
    return vwap


if __name__ == "__main__":
    # Test trade sign classification
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Computing trade sign features...")
    trade_features = compute_trade_sign_features(df, lookback_ticks=[1, 5, 10, 20])
    
    print(f"\nTrade feature shape: {trade_features.shape}")
    print(f"Trade columns: {trade_features.columns.tolist()}")
    print("\nTrade statistics:")
    print(trade_features.describe())
    
    # Check trade sign distribution
    print("\nTrade sign distribution:")
    print(trade_features['trade_sign'].value_counts())
    
    # Check correlation with future returns
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    future_return = mid_price.pct_change(5).shift(-5)
    
    print("\nCorrelation with 5-tick future return:")
    for col in ['signed_volume', 'signed_volume_cum_10', 'buy_sell_imbalance_10']:
        if col in trade_features.columns:
            corr = trade_features[col].corr(future_return)
            print(f"{col}: {corr:.4f}")
