import pandas as pd
import numpy as np
from typing import Tuple


def compute_ic_series(predictions: pd.DataFrame, returns: pd.DataFrame, 
                     method: str = 'spearman') -> pd.Series:
    """Time series of information coefficients."""
    ic_list = []
    
    common_dates = predictions.index.intersection(returns.index)
    
    for date in common_dates:
        pred = predictions.loc[date]
        ret = returns.loc[date]
        
        valid = pred.notna() & ret.notna()
        if valid.sum() > 10:
            ic = pred[valid].corr(ret[valid], method=method)
            ic_list.append((date, ic))
    
    return pd.Series(dict(ic_list))


def compute_ic_statistics(ic_series: pd.Series) -> dict:
    """IC summary statistics."""
    return {
        'mean_ic': ic_series.mean(),
        'std_ic': ic_series.std(),
        'ic_ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else np.nan,
        'hit_rate': (ic_series > 0).mean(),
        'skew': ic_series.skew(),
        'kurtosis': ic_series.kurtosis()
    }


def compute_quantile_returns(predictions: pd.DataFrame, returns: pd.DataFrame, 
                            n_quantiles: int = 5) -> pd.DataFrame:
    """Average returns by prediction quantile."""
    quantile_returns = []
    
    common_dates = predictions.index.intersection(returns.index)
    
    for date in common_dates:
        pred = predictions.loc[date]
        ret = returns.loc[date]
        
        valid = pred.notna() & ret.notna()
        if valid.sum() > n_quantiles * 5:
            quantiles = pd.qcut(pred[valid], n_quantiles, labels=False, duplicates='drop')
            
            qret = {}
            for q in range(n_quantiles):
                mask = quantiles == q
                if mask.sum() > 0:
                    qret[f'Q{q+1}'] = ret[valid][mask].mean()
            
            quantile_returns.append(pd.Series(qret, name=date))
    
    return pd.DataFrame(quantile_returns)


def compute_long_short_spread(predictions: pd.DataFrame, returns: pd.DataFrame, 
                              quantile: float = 0.2) -> pd.Series:
    """Long top quantile, short bottom quantile."""
    spreads = []
    
    common_dates = predictions.index.intersection(returns.index)
    
    for date in common_dates:
        pred = predictions.loc[date]
        ret = returns.loc[date]
        
        valid = pred.notna() & ret.notna()
        if valid.sum() > 20:
            top_threshold = pred[valid].quantile(1 - quantile)
            bottom_threshold = pred[valid].quantile(quantile)
            
            long_ret = ret[valid][pred[valid] >= top_threshold].mean()
            short_ret = ret[valid][pred[valid] <= bottom_threshold].mean()
            
            spreads.append((date, long_ret - short_ret))
    
    return pd.Series(dict(spreads))
