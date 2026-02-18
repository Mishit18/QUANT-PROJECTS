import pandas as pd
import numpy as np
from typing import List


def compute_alpha_decay(predictions: pd.DataFrame, returns: pd.DataFrame, 
                       horizons: List[int] = [1, 2, 3, 5, 10, 20]) -> pd.DataFrame:
    """IC decay over multiple horizons."""
    decay_ics = {}
    
    for h in horizons:
        forward_returns = returns.shift(-h)
        
        ic_list = []
        common_dates = predictions.index.intersection(forward_returns.index)
        
        for date in common_dates:
            pred = predictions.loc[date]
            ret = forward_returns.loc[date]
            
            valid = pred.notna() & ret.notna()
            if valid.sum() > 10:
                ic = pred[valid].corr(ret[valid], method='spearman')
                ic_list.append(ic)
        
        decay_ics[f'horizon_{h}'] = np.mean(ic_list) if ic_list else np.nan
    
    return pd.Series(decay_ics)


def compute_cumulative_returns_by_horizon(predictions: pd.DataFrame, returns: pd.DataFrame,
                                         horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """Cumulative returns at different holding periods."""
    cum_returns = {}
    
    for h in horizons:
        forward_returns = returns.rolling(h).sum().shift(-h)
        
        long_short = []
        common_dates = predictions.index.intersection(forward_returns.index)
        
        for date in common_dates:
            pred = predictions.loc[date]
            ret = forward_returns.loc[date]
            
            valid = pred.notna() & ret.notna()
            if valid.sum() > 20:
                top = pred[valid].quantile(0.8)
                bottom = pred[valid].quantile(0.2)
                
                long_ret = ret[valid][pred[valid] >= top].mean()
                short_ret = ret[valid][pred[valid] <= bottom].mean()
                long_short.append(long_ret - short_ret)
        
        cum_returns[f'horizon_{h}d'] = long_short
    
    return pd.DataFrame(cum_returns)


def compute_half_life(ic_series: pd.Series, threshold: float = 0.5) -> int:
    """Estimate alpha half-life in days."""
    if ic_series.iloc[0] <= 0:
        return 0
    
    target = ic_series.iloc[0] * threshold
    
    for i, ic in enumerate(ic_series):
        if ic <= target:
            return i
    
    return len(ic_series)
