import numpy as np
import pandas as pd
from typing import Tuple


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe."""
    if len(returns) < 2:
        return np.nan
    return returns.mean() / returns.std() * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sortino."""
    if len(returns) < 2:
        return np.nan
    downside = returns[returns < 0].std()
    if downside < 1e-8:
        return np.nan
    return returns.mean() / downside * np.sqrt(periods_per_year)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    """Maximum drawdown from peak."""
    if len(cumulative_returns) < 2:
        return np.nan
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return / max drawdown."""
    cum_ret = (1 + returns).cumprod()
    mdd = max_drawdown(cum_ret)
    if abs(mdd) < 1e-8:
        return np.nan
    ann_ret = returns.mean() * periods_per_year
    return ann_ret / abs(mdd)


def information_coefficient(predictions: pd.Series, returns: pd.Series, 
                           method: str = 'spearman') -> float:
    """IC between predictions and realized returns."""
    valid = predictions.notna() & returns.notna()
    if valid.sum() < 10:
        return np.nan
    return predictions[valid].corr(returns[valid], method=method)


def ic_ir(ic_series: pd.Series) -> float:
    """Information ratio of IC time series."""
    if len(ic_series) < 2:
        return np.nan
    return ic_series.mean() / ic_series.std()


def hit_rate(predictions: pd.Series, returns: pd.Series) -> float:
    """Fraction of correct directional predictions."""
    valid = predictions.notna() & returns.notna()
    if valid.sum() < 1:
        return np.nan
    correct = (np.sign(predictions[valid]) == np.sign(returns[valid])).sum()
    return correct / valid.sum()


def turnover(weights_t0: pd.Series, weights_t1: pd.Series) -> float:
    """Portfolio turnover between periods."""
    aligned = pd.DataFrame({'t0': weights_t0, 't1': weights_t1}).fillna(0)
    return (aligned['t0'] - aligned['t1']).abs().sum() / 2
