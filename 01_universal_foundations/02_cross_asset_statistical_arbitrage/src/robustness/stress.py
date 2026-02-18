import pandas as pd
import numpy as np
from typing import Dict, List


def regime_analysis(returns: pd.Series, market_returns: pd.Series, 
                   quantiles: List[float] = [0.2, 0.8]) -> Dict[str, float]:
    """Performance across market regimes."""
    
    market_quantiles = market_returns.quantile(quantiles)
    
    bear_mask = market_returns <= market_quantiles[0]
    bull_mask = market_returns >= market_quantiles[1]
    neutral_mask = ~bear_mask & ~bull_mask
    
    return {
        'bear_sharpe': returns[bear_mask].mean() / returns[bear_mask].std() * np.sqrt(252) if bear_mask.sum() > 1 else np.nan,
        'bull_sharpe': returns[bull_mask].mean() / returns[bull_mask].std() * np.sqrt(252) if bull_mask.sum() > 1 else np.nan,
        'neutral_sharpe': returns[neutral_mask].mean() / returns[neutral_mask].std() * np.sqrt(252) if neutral_mask.sum() > 1 else np.nan,
        'bear_return': returns[bear_mask].mean() * 252,
        'bull_return': returns[bull_mask].mean() * 252,
        'neutral_return': returns[neutral_mask].mean() * 252
    }


def volatility_regime_analysis(returns: pd.Series, vol_window: int = 60,
                               quantiles: List[float] = [0.33, 0.67]) -> Dict[str, float]:
    """Performance across volatility regimes."""
    
    rolling_vol = returns.rolling(vol_window).std()
    vol_quantiles = rolling_vol.quantile(quantiles)
    
    low_vol = rolling_vol <= vol_quantiles[0]
    high_vol = rolling_vol >= vol_quantiles[1]
    mid_vol = ~low_vol & ~high_vol
    
    return {
        'low_vol_sharpe': returns[low_vol].mean() / returns[low_vol].std() * np.sqrt(252) if low_vol.sum() > 1 else np.nan,
        'high_vol_sharpe': returns[high_vol].mean() / returns[high_vol].std() * np.sqrt(252) if high_vol.sum() > 1 else np.nan,
        'mid_vol_sharpe': returns[mid_vol].mean() / returns[mid_vol].std() * np.sqrt(252) if mid_vol.sum() > 1 else np.nan
    }


def drawdown_analysis(returns: pd.Series) -> Dict[str, float]:
    """Detailed drawdown statistics."""
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    
    in_drawdown = drawdown < 0
    drawdown_periods = (in_drawdown != in_drawdown.shift()).cumsum()[in_drawdown]
    
    if len(drawdown_periods) > 0:
        avg_drawdown_length = drawdown_periods.value_counts().mean()
        max_drawdown_length = drawdown_periods.value_counts().max()
    else:
        avg_drawdown_length = 0
        max_drawdown_length = 0
    
    return {
        'max_drawdown': drawdown.min(),
        'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
        'drawdown_frequency': in_drawdown.mean(),
        'avg_drawdown_length': avg_drawdown_length,
        'max_drawdown_length': max_drawdown_length
    }


def tail_risk_analysis(returns: pd.Series) -> Dict[str, float]:
    """Tail risk metrics."""
    return {
        'var_95': returns.quantile(0.05),
        'var_99': returns.quantile(0.01),
        'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
        'cvar_99': returns[returns <= returns.quantile(0.01)].mean(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis()
    }
