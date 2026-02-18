import numpy as np
import pandas as pd
from typing import Optional, Union


def rank_normalize(x: pd.Series) -> pd.Series:
    """Cross-sectional rank to [-1, 1]."""
    valid = x.notna()
    if valid.sum() < 2:
        return pd.Series(np.nan, index=x.index)
    ranks = x[valid].rank(pct=True)
    normalized = 2 * ranks - 1
    result = pd.Series(np.nan, index=x.index)
    result[valid] = normalized
    return result


def zscore(x: pd.Series, clip: Optional[float] = 3.0) -> pd.Series:
    """Cross-sectional z-score with optional clipping."""
    valid = x.notna()
    if valid.sum() < 2:
        return pd.Series(np.nan, index=x.index)
    mean = x[valid].mean()
    std = x[valid].std()
    if std < 1e-8:
        return pd.Series(0.0, index=x.index)
    z = (x - mean) / std
    if clip:
        z = z.clip(-clip, clip)
    return z


def demean(x: pd.Series) -> pd.Series:
    """Cross-sectional demean."""
    valid = x.notna()
    if valid.sum() < 1:
        return x
    return x - x[valid].mean()


def winsorize(x: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Winsorize to percentile bounds."""
    valid = x.notna()
    if valid.sum() < 2:
        return x
    bounds = x[valid].quantile([lower, upper])
    return x.clip(bounds.iloc[0], bounds.iloc[1])


def expanding_rank_ic(predictions: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute expanding window rank IC."""
    dates = predictions.index.intersection(returns.index)
    ic_series = []
    for date in dates:
        pred = predictions.loc[date]
        ret = returns.loc[date]
        valid = pred.notna() & ret.notna()
        if valid.sum() > 10:
            ic = pred[valid].corr(ret[valid], method='spearman')
            ic_series.append((date, ic))
    return pd.Series(dict(ic_series))


def safe_divide(num: Union[float, np.ndarray], denom: Union[float, np.ndarray], 
                fill: float = 0.0) -> Union[float, np.ndarray]:
    """Division with zero handling."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / denom
        if isinstance(result, np.ndarray):
            result[~np.isfinite(result)] = fill
        elif not np.isfinite(result):
            result = fill
    return result
