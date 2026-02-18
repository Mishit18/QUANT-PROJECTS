"""
Custom metrics for HFT alpha research.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_information_coefficient(predictions: np.ndarray, 
                                    returns: np.ndarray) -> float:
    """
    Compute Information Coefficient (IC).
    
    IC is the correlation between predictions and realized returns.
    
    Args:
        predictions: Model predictions (continuous or probabilities)
        returns: Realized returns
        
    Returns:
        IC value (correlation coefficient)
    """
    # Remove NaN
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    
    if valid_mask.sum() < 2:
        return np.nan
    
    ic = np.corrcoef(predictions[valid_mask], returns[valid_mask])[0, 1]
    
    return ic


def compute_rank_ic(predictions: np.ndarray, returns: np.ndarray) -> float:
    """
    Compute Rank Information Coefficient (Spearman correlation).
    
    More robust to outliers than Pearson IC.
    """
    from scipy.stats import spearmanr
    
    valid_mask = ~(np.isnan(predictions) | np.isnan(returns))
    
    if valid_mask.sum() < 2:
        return np.nan
    
    rank_ic, _ = spearmanr(predictions[valid_mask], returns[valid_mask])
    
    return rank_ic


def compute_hit_rate_by_confidence(y_true: np.ndarray, y_pred: np.ndarray,
                                   confidence: np.ndarray, 
                                   n_bins: int = 5) -> pd.DataFrame:
    """
    Compute hit rate stratified by prediction confidence.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence: Prediction confidence scores
        n_bins: Number of confidence bins
        
    Returns:
        DataFrame with confidence bins and corresponding hit rates
    """
    # Create confidence bins
    confidence_bins = pd.qcut(confidence, q=n_bins, labels=False, duplicates='drop')
    
    results = []
    
    for bin_idx in range(n_bins):
        bin_mask = (confidence_bins == bin_idx)
        
        if bin_mask.sum() == 0:
            continue
        
        y_true_bin = y_true[bin_mask]
        y_pred_bin = y_pred[bin_mask]
        
        # Hit rate
        hit_rate = (y_true_bin == y_pred_bin).mean()
        
        # Directional hit rate
        directional_mask = (y_true_bin != 0) & (y_pred_bin != 0)
        if directional_mask.sum() > 0:
            dir_hit_rate = (y_true_bin[directional_mask] == y_pred_bin[directional_mask]).mean()
        else:
            dir_hit_rate = np.nan
        
        results.append({
            'confidence_bin': bin_idx,
            'min_confidence': confidence[bin_mask].min(),
            'max_confidence': confidence[bin_mask].max(),
            'mean_confidence': confidence[bin_mask].mean(),
            'hit_rate': hit_rate,
            'directional_hit_rate': dir_hit_rate,
            'n_samples': bin_mask.sum()
        })
    
    return pd.DataFrame(results)


def compute_turnover(positions: pd.Series) -> float:
    """
    Compute portfolio turnover.
    
    Turnover = sum of absolute position changes / average position
    """
    position_changes = positions.diff().abs()
    avg_position = positions.abs().mean()
    
    if avg_position == 0:
        return 0.0
    
    turnover = position_changes.sum() / (avg_position * len(positions))
    
    return turnover


def compute_calmar_ratio(returns: pd.Series, max_drawdown: float) -> float:
    """
    Compute Calmar ratio (annualized return / max drawdown).
    
    Args:
        returns: Series of returns
        max_drawdown: Maximum drawdown (positive value)
        
    Returns:
        Calmar ratio
    """
    if max_drawdown == 0:
        return np.inf if returns.mean() > 0 else 0.0
    
    annualized_return = returns.mean() * 252 * 6.5 * 3600  # approximate
    calmar = annualized_return / abs(max_drawdown)
    
    return calmar


def compute_sortino_ratio(returns: pd.Series, target_return: float = 0.0,
                         periods_per_year: float = 252 * 6.5 * 3600) -> float:
    """
    Compute Sortino ratio (downside risk-adjusted return).
    
    Args:
        returns: Series of returns
        target_return: Target return threshold
        periods_per_year: Periods per year for annualization
        
    Returns:
        Sortino ratio
    """
    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if returns.mean() > target_return else 0.0
    
    downside_std = downside_returns.std()
    
    if downside_std == 0:
        return np.inf if returns.mean() > target_return else 0.0
    
    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)
    
    return sortino


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    
    # Synthetic data
    predictions = np.random.randn(1000)
    returns = predictions + np.random.randn(1000) * 0.5
    
    # IC
    ic = compute_information_coefficient(predictions, returns)
    rank_ic = compute_rank_ic(predictions, returns)
    
    print(f"Information Coefficient: {ic:.4f}")
    print(f"Rank IC: {rank_ic:.4f}")
    
    # Hit rate by confidence (0=down, 1=flat, 2=up)
    y_true = np.random.choice([0, 1, 2], size=1000)
    y_pred = y_true.copy()
    noise_idx = np.random.choice(len(y_pred), size=200, replace=False)
    y_pred[noise_idx] = np.random.choice([0, 1, 2], size=200)
    confidence = np.random.random(1000)
    
    hit_rate_df = compute_hit_rate_by_confidence(y_true, y_pred, confidence, n_bins=5)
    print("\nHit rate by confidence:")
    print(hit_rate_df)
