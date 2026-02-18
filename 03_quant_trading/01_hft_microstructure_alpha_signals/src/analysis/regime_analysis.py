"""
Regime analysis: model performance across different market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path


def identify_volatility_regimes(df: pd.DataFrame, window: int = 100) -> pd.Series:
    """
    Classify periods into high/low volatility regimes.
    
    Args:
        df: LOB data
        window: Rolling window for volatility calculation
        
    Returns:
        Series with regime labels: 'high_vol' or 'low_vol'
    """
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    returns = mid_price.pct_change()
    
    # Rolling volatility
    vol = returns.rolling(window=window, min_periods=1).std()
    
    # Classify based on median
    vol_median = vol.median()
    
    regime = pd.Series('low_vol', index=df.index)
    regime[vol > vol_median] = 'high_vol'
    
    return regime


def identify_liquidity_regimes(df: pd.DataFrame, n_levels: int = 5) -> pd.Series:
    """
    Classify periods into high/low liquidity regimes.
    
    Args:
        df: LOB data
        n_levels: Number of book levels to consider
        
    Returns:
        Series with regime labels: 'high_liq' or 'low_liq'
    """
    # Total depth
    total_depth = sum(
        df[f'bid_size_{i}'] + df[f'ask_size_{i}']
        for i in range(1, n_levels + 1)
    )
    
    # Classify based on median
    depth_median = total_depth.median()
    
    regime = pd.Series('low_liq', index=df.index)
    regime[total_depth > depth_median] = 'high_liq'
    
    return regime


def identify_time_of_day_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Classify periods by time of day.
    
    Args:
        df: LOB data with timestamp column
        
    Returns:
        Series with regime labels: 'open', 'mid_day', 'close'
    """
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ns')
    hour = df['datetime'].dt.hour
    
    regime = pd.Series('mid_day', index=df.index)
    regime[hour < 11] = 'open'
    regime[hour >= 15] = 'close'
    
    return regime


def evaluate_by_regime(features: pd.DataFrame, labels: pd.Series,
                      model: any, regime: pd.Series) -> pd.DataFrame:
    """
    Evaluate model performance across different regimes.
    
    Args:
        features: Feature DataFrame
        labels: Label Series
        model: Trained model
        regime: Regime classification Series
        
    Returns:
        DataFrame with performance metrics by regime
    """
    results = []
    
    for regime_name in regime.unique():
        # Filter by regime
        regime_mask = (regime == regime_name) & labels.notna()
        
        if regime_mask.sum() == 0:
            continue
        
        X_regime = features[regime_mask]
        y_regime = labels[regime_mask]
        
        # Predict
        y_pred = model.predict(X_regime)
        
        # Metrics
        accuracy = (y_pred == y_regime.values).mean()
        
        # Hit rate (directional)
        directional_mask = (y_regime != 0) & (y_pred != 0)
        if directional_mask.sum() > 0:
            hit_rate = (y_regime[directional_mask] == y_pred[directional_mask]).mean()
        else:
            hit_rate = np.nan
        
        results.append({
            'regime': regime_name,
            'accuracy': accuracy,
            'hit_rate': hit_rate,
            'n_samples': len(y_regime)
        })
    
    return pd.DataFrame(results)


def plot_regime_performance(regime_results: Dict[str, pd.DataFrame], 
                           save_path: str = None):
    """
    Plot model performance across different regime types.
    
    Args:
        regime_results: Dict mapping regime type to results DataFrame
        save_path: Path to save figure
    """
    n_regimes = len(regime_results)
    fig, axes = plt.subplots(1, n_regimes, figsize=(6*n_regimes, 5))
    
    if n_regimes == 1:
        axes = [axes]
    
    for ax, (regime_type, results) in zip(axes, regime_results.items()):
        # Plot hit rate by regime
        x = range(len(results))
        ax.bar(x, results['hit_rate'], alpha=0.7)
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        
        ax.set_xticks(x)
        ax.set_xticklabels(results['regime'], rotation=45, ha='right')
        ax.set_ylabel('Hit Rate')
        ax.set_title(f'Performance by {regime_type.replace("_", " ").title()}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Regime performance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_all_regimes(df: pd.DataFrame, features: pd.DataFrame, 
                       labels: pd.Series, model: any) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive regime analysis.
    
    Args:
        df: LOB data
        features: Feature DataFrame
        labels: Label Series
        model: Trained model
        
    Returns:
        Dictionary mapping regime type to results DataFrame
    """
    results = {}
    
    # Volatility regimes
    vol_regime = identify_volatility_regimes(df)
    results['volatility'] = evaluate_by_regime(features, labels, model, vol_regime)
    
    # Liquidity regimes
    liq_regime = identify_liquidity_regimes(df)
    results['liquidity'] = evaluate_by_regime(features, labels, model, liq_regime)
    
    # Time of day regimes (if timestamp available)
    if 'timestamp' in df.columns:
        tod_regime = identify_time_of_day_regimes(df)
        results['time_of_day'] = evaluate_by_regime(features, labels, model, tod_regime)
    
    return results


if __name__ == "__main__":
    from src.models.tree_models import XGBoostModel
    from src.features.base_features import compute_all_base_features
    from src.labels.future_ticks import create_labels_all_horizons
    
    # Load data
    df = pd.read_parquet("data/raw/lob_data.parquet")
    
    # Features and labels
    features = compute_all_base_features(df)
    labels = create_labels_all_horizons(df, horizons=[5])
    
    # Train model
    split_idx = int(len(df) * 0.8)
    model = XGBoostModel()
    model.fit(features.iloc[:split_idx], labels['label_5'].iloc[:split_idx])
    
    # Regime analysis on test set
    test_features = features.iloc[split_idx:]
    test_labels = labels['label_5'].iloc[split_idx:]
    test_df = df.iloc[split_idx:]
    
    regime_results = analyze_all_regimes(test_df, test_features, test_labels, model)
    
    print("\nRegime Analysis Results:")
    for regime_type, results in regime_results.items():
        print(f"\n{regime_type.upper()}:")
        print(results)
    
    # Plot
    plot_regime_performance(regime_results)
