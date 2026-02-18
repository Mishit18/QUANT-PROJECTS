"""
Alpha decay analysis.

Measures how prediction accuracy degrades with forecast horizon.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from pathlib import Path


def compute_hit_rate_by_horizon(features: pd.DataFrame, labels: pd.DataFrame,
                                model: any, horizons: List[int]) -> pd.DataFrame:
    """
    Compute hit rate across multiple horizons.
    
    Args:
        features: Feature DataFrame
        labels: Label DataFrame with multiple horizons
        model: Trained model
        horizons: List of prediction horizons
        
    Returns:
        DataFrame with horizon and hit_rate columns
    """
    results = []
    
    for horizon in horizons:
        label_col = f'label_{horizon}'
        
        if label_col not in labels.columns:
            continue
        
        # Get valid samples
        valid_mask = labels[label_col].notna()
        X = features[valid_mask]
        y = labels[label_col][valid_mask]
        
        # Predict
        y_pred = model.predict(X)
        
        # Compute hit rate (directional accuracy, excluding flat)
        directional_mask = (y != 0) & (y_pred != 0)
        if directional_mask.sum() > 0:
            hit_rate = (y[directional_mask] == y_pred[directional_mask]).mean()
        else:
            hit_rate = np.nan
        
        # Overall accuracy
        accuracy = (y == y_pred).mean()
        
        results.append({
            'horizon': horizon,
            'hit_rate': hit_rate,
            'accuracy': accuracy,
            'n_samples': len(y)
        })
    
    return pd.DataFrame(results)


def compute_alpha_decay_curve(features: pd.DataFrame, labels: pd.DataFrame,
                              model: any, max_horizon: int = 50) -> pd.DataFrame:
    """
    Compute alpha decay curve for fine-grained horizons.
    
    Args:
        features: Feature DataFrame
        labels: Label DataFrame
        model: Trained model
        max_horizon: Maximum horizon to test
        
    Returns:
        DataFrame with decay curve
    """
    # Generate labels for all horizons
    df = labels.copy()
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2 if 'bid_price_1' in df.columns else None
    
    results = []
    
    for horizon in range(1, max_horizon + 1):
        # Create label for this horizon
        if mid_price is not None:
            future_mid = mid_price.shift(-horizon)
            price_change = (future_mid - mid_price) / 0.01  # in ticks
            y = pd.Series(0, index=df.index)
            y[price_change >= 0.5] = 1
            y[price_change <= -0.5] = -1
        else:
            # Use pre-computed labels if available
            label_col = f'label_{horizon}'
            if label_col not in labels.columns:
                continue
            y = labels[label_col]
        
        # Get valid samples
        valid_mask = y.notna()
        X = features[valid_mask]
        y_valid = y[valid_mask]
        
        if len(y_valid) == 0:
            continue
        
        # Predict
        y_pred = model.predict(X)
        
        # Hit rate
        directional_mask = (y_valid != 0) & (y_pred != 0)
        if directional_mask.sum() > 0:
            hit_rate = (y_valid[directional_mask] == y_pred[directional_mask]).mean()
        else:
            hit_rate = np.nan
        
        results.append({
            'horizon': horizon,
            'hit_rate': hit_rate
        })
    
    return pd.DataFrame(results)


def plot_alpha_decay(decay_df: pd.DataFrame, save_path: str = None):
    """
    Plot alpha decay curve.
    
    Args:
        decay_df: DataFrame with horizon and hit_rate columns
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(decay_df['horizon'], decay_df['hit_rate'], 
             marker='o', linewidth=2, markersize=6)
    
    # Add 50% reference line (random guess)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random (50%)')
    
    plt.xlabel('Prediction Horizon (ticks)', fontsize=12)
    plt.ylabel('Hit Rate', fontsize=12)
    plt.title('Alpha Decay: Hit Rate vs Prediction Horizon', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Alpha decay plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_hit_rate_comparison(hit_rate_df: pd.DataFrame, save_path: str = None):
    """
    Plot hit rate comparison across horizons.
    
    Args:
        hit_rate_df: DataFrame with horizon, hit_rate, and accuracy
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hit rate plot
    ax1.bar(hit_rate_df['horizon'], hit_rate_df['hit_rate'], alpha=0.7)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    ax1.set_xlabel('Horizon (ticks)')
    ax1.set_ylabel('Hit Rate')
    ax1.set_title('Directional Hit Rate by Horizon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # Accuracy plot
    ax2.bar(hit_rate_df['horizon'], hit_rate_df['accuracy'], alpha=0.7, color='orange')
    ax2.set_xlabel('Horizon (ticks)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Overall Accuracy by Horizon')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Hit rate comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test alpha decay analysis
    from src.models.tree_models import XGBoostModel
    from src.features.base_features import compute_all_base_features
    from src.labels.future_ticks import create_labels_all_horizons
    
    # Load data
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    # Features and labels
    features = compute_all_base_features(df)
    labels = create_labels_all_horizons(df, horizons=[1, 5, 10, 20])
    
    # Train model
    split_idx = int(len(df) * 0.8)
    model = XGBoostModel()
    model.fit(features.iloc[:split_idx], labels['label_5'].iloc[:split_idx])
    
    # Compute alpha decay
    hit_rate_df = compute_hit_rate_by_horizon(
        features.iloc[split_idx:], 
        labels.iloc[split_idx:], 
        model, 
        horizons=[1, 5, 10, 20]
    )
    
    print("Hit rate by horizon:")
    print(hit_rate_df)
    
    # Plot
    plot_hit_rate_comparison(hit_rate_df)
