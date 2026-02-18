import pandas as pd
import numpy as np
from typing import Dict


def compute_momentum(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Standard momentum signals."""
    features = {}
    for w in windows:
        features[f'mom_{w}d'] = prices.pct_change(w, fill_method=None)
    return features


def compute_acceleration(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Second derivative of price."""
    features = {}
    for w in windows:
        ret = prices.pct_change(w, fill_method=None)
        features[f'accel_{w}d'] = ret.diff(w)
    return features


def compute_reversal(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Mean reversion signals."""
    features = {}
    for w in windows:
        ret = prices.pct_change(w, fill_method=None)
        features[f'reversal_{w}d'] = -ret
    return features


def compute_trend_strength(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Linear regression slope and R²."""
    features = {}
    
    for w in windows:
        slopes = []
        r_squared = []
        
        for date in prices.index[w:]:
            window_prices = prices.loc[:date].tail(w)
            x = np.arange(w)
            
            slope_series = {}
            r2_series = {}
            
            for col in prices.columns:
                y = window_prices[col].values
                if np.isnan(y).sum() < w // 2:
                    valid = ~np.isnan(y)
                    if valid.sum() > 2:
                        x_valid = x[valid]
                        y_valid = y[valid]
                        coef = np.polyfit(x_valid, y_valid, 1)
                        slope_series[col] = coef[0] / y_valid.mean() if y_valid.mean() != 0 else np.nan
                        
                        y_pred = np.polyval(coef, x_valid)
                        ss_res = ((y_valid - y_pred) ** 2).sum()
                        ss_tot = ((y_valid - y_valid.mean()) ** 2).sum()
                        r2_series[col] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
                    else:
                        slope_series[col] = np.nan
                        r2_series[col] = np.nan
                else:
                    slope_series[col] = np.nan
                    r2_series[col] = np.nan
            
            slopes.append(pd.Series(slope_series, name=date))
            r_squared.append(pd.Series(r2_series, name=date))
        
        features[f'trend_slope_{w}d'] = pd.DataFrame(slopes)
        features[f'trend_r2_{w}d'] = pd.DataFrame(r_squared)
    
    return features


def compute_momentum_consistency(prices: pd.DataFrame, window: int = 60, subperiods: int = 5) -> Dict[str, pd.DataFrame]:
    """Fraction of subperiods with positive returns."""
    subwindow = window // subperiods
    consistency = []
    
    for i in range(subperiods):
        ret = prices.pct_change(subwindow, fill_method=None).shift(i * subwindow)
        consistency.append((ret > 0).astype(float))
    
    return {'mom_consistency': pd.concat(consistency, axis=1).mean(axis=1)}
