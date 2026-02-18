"""
Label generation for future price movements.

Creates classification labels for midprice direction at various horizons.
Critical: Labels must not leak future information into features.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple


def compute_future_midprice_move(df: pd.DataFrame, horizon: int, 
                                 threshold: float = 0.5) -> pd.Series:
    """
    Compute future midprice movement label.
    
    Args:
        df: LOB data
        horizon: Number of ticks ahead to predict
        threshold: Movement threshold in ticks for classification
        
    Returns:
        Series with labels: 0 (down), 1 (flat), 2 (up)
    """
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    
    # Future mid price (horizon ticks ahead)
    future_mid = mid_price.shift(-horizon)
    
    # Price change in ticks (assume tick_size = 0.01)
    tick_size = 0.01
    price_change_ticks = (future_mid - mid_price) / tick_size
    
    # Classify: 0 (down), 1 (flat), 2 (up) for XGBoost compatibility
    labels = pd.Series(1, index=df.index)  # default to flat
    labels[price_change_ticks >= threshold] = 2  # up
    labels[price_change_ticks <= -threshold] = 0  # down
    
    return labels


def compute_future_return(df: pd.DataFrame, horizon: int) -> pd.Series:
    """
    Compute future return (for regression tasks).
    
    Args:
        df: LOB data
        horizon: Number of ticks ahead
        
    Returns:
        Series with future returns
    """
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    future_return = mid_price.pct_change(horizon).shift(-horizon)
    
    return future_return


def create_labels_all_horizons(df: pd.DataFrame, 
                               horizons: List[int] = [1, 5, 10, 20],
                               threshold: float = 0.5) -> pd.DataFrame:
    """
    Create labels for multiple prediction horizons.
    
    Args:
        df: LOB data
        horizons: List of tick horizons to predict
        threshold: Classification threshold in ticks
        
    Returns:
        DataFrame with label columns for each horizon
    """
    labels = pd.DataFrame(index=df.index)
    
    for horizon in horizons:
        # Classification labels
        labels[f'label_{horizon}'] = compute_future_midprice_move(
            df, horizon=horizon, threshold=threshold
        )
        
        # Regression targets (optional)
        labels[f'return_{horizon}'] = compute_future_return(df, horizon=horizon)
    
    return labels


def get_label_distribution(labels: pd.DataFrame, horizon: int) -> dict:
    """
    Get distribution of labels for a specific horizon.
    
    Useful for checking class balance.
    """
    label_col = f'label_{horizon}'
    
    if label_col not in labels.columns:
        return {}
    
    counts = labels[label_col].value_counts()
    total = len(labels[label_col].dropna())
    
    distribution = {
        'down': counts.get(0, 0),
        'flat': counts.get(1, 0),
        'up': counts.get(2, 0),
        'total': total,
        'down_pct': counts.get(0, 0) / total * 100 if total > 0 else 0,
        'flat_pct': counts.get(1, 0) / total * 100 if total > 0 else 0,
        'up_pct': counts.get(2, 0) / total * 100 if total > 0 else 0,
    }
    
    return distribution


def remove_unlabeled_rows(features: pd.DataFrame, labels: pd.DataFrame, 
                         horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows where labels are NaN (end of dataset).
    
    Args:
        features: Feature DataFrame
        labels: Label DataFrame
        horizon: Horizon to use for filtering
        
    Returns:
        Tuple of (filtered_features, filtered_labels)
    """
    label_col = f'label_{horizon}'
    
    valid_mask = labels[label_col].notna()
    
    features_clean = features[valid_mask].copy()
    labels_clean = labels[valid_mask].copy()
    
    return features_clean, labels_clean


if __name__ == "__main__":
    # Test label generation
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    print("Creating labels for multiple horizons...")
    labels = create_labels_all_horizons(df, horizons=[1, 5, 10, 20], threshold=0.5)
    
    print(f"\nLabel shape: {labels.shape}")
    print(f"Label columns: {labels.columns.tolist()}")
    
    # Check label distributions
    print("\nLabel distributions:")
    for horizon in [1, 5, 10, 20]:
        dist = get_label_distribution(labels, horizon=horizon)
        print(f"\nHorizon {horizon} ticks:")
        print(f"  Up: {dist['up']} ({dist['up_pct']:.1f}%)")
        print(f"  Flat: {dist['flat']} ({dist['flat_pct']:.1f}%)")
        print(f"  Down: {dist['down']} ({dist['down_pct']:.1f}%)")
        print(f"  Total: {dist['total']}")
