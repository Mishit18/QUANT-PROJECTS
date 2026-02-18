import pandas as pd
import numpy as np


def compute_turnover_series(weights: pd.DataFrame) -> pd.Series:
    """Daily portfolio turnover."""
    turnover = []
    
    for i in range(1, len(weights)):
        w_prev = weights.iloc[i-1].fillna(0)
        w_curr = weights.iloc[i].fillna(0)
        
        common = w_prev.index.intersection(w_curr.index)
        turn = (w_prev[common] - w_curr[common]).abs().sum() / 2
        
        turnover.append((weights.index[i], turn))
    
    return pd.Series(dict(turnover))


def compute_position_churn(weights: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """Fraction of positions changed above threshold."""
    churn = []
    
    for i in range(1, len(weights)):
        w_prev = weights.iloc[i-1].fillna(0)
        w_curr = weights.iloc[i].fillna(0)
        
        common = w_prev.index.intersection(w_curr.index)
        changed = (w_prev[common] - w_curr[common]).abs() > threshold
        churn_rate = changed.sum() / len(common) if len(common) > 0 else 0
        
        churn.append((weights.index[i], churn_rate))
    
    return pd.Series(dict(churn))


def compute_concentration(weights: pd.DataFrame) -> pd.Series:
    """Portfolio concentration via Herfindahl index."""
    concentration = []
    
    for date in weights.index:
        w = weights.loc[date].fillna(0).abs()
        if w.sum() > 0:
            w_norm = w / w.sum()
            hhi = (w_norm ** 2).sum()
            concentration.append((date, hhi))
    
    return pd.Series(dict(concentration))


def compute_holding_period(weights: pd.DataFrame) -> float:
    """Average holding period in days."""
    turnover = compute_turnover_series(weights)
    avg_turnover = turnover.mean()
    if avg_turnover > 0:
        return 1.0 / avg_turnover
    return np.inf
