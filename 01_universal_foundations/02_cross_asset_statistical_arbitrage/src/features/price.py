import pandas as pd
import numpy as np
from typing import Dict


def compute_returns_features(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Multi-horizon return features."""
    features = {}
    for w in windows:
        features[f'ret_{w}d'] = prices.pct_change(w, fill_method=None)
        features[f'logret_{w}d'] = np.log(prices / prices.shift(w))
    return features


def compute_price_levels(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Price relative to historical levels."""
    features = {}
    for w in windows:
        roll_max = prices.rolling(w).max()
        roll_min = prices.rolling(w).min()
        features[f'pct_from_high_{w}d'] = (prices - roll_max) / roll_max
        features[f'pct_from_low_{w}d'] = (prices - roll_min) / roll_min
        features[f'price_range_{w}d'] = (roll_max - roll_min) / roll_min
    return features


def compute_moving_averages(prices: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """MA crossovers and deviations."""
    features = {}
    for w in windows:
        ma = prices.rolling(w).mean()
        features[f'price_to_ma_{w}d'] = prices / ma - 1
    
    if len(windows) >= 2:
        for i in range(len(windows) - 1):
            w1, w2 = windows[i], windows[i + 1]
            ma1 = prices.rolling(w1).mean()
            ma2 = prices.rolling(w2).mean()
            features[f'ma_cross_{w1}_{w2}'] = ma1 / ma2 - 1
    
    return features


def compute_gap_features(open_prices: pd.DataFrame, close_prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Overnight gap features."""
    prev_close = close_prices.shift(1)
    gap = (open_prices - prev_close) / prev_close
    return {'overnight_gap': gap}
