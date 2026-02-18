import pandas as pd
import numpy as np
from typing import Dict


def compute_volume_features(volumes: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Volume relative to historical average."""
    features = {}
    for w in windows:
        avg_vol = volumes.rolling(w).mean()
        features[f'vol_ratio_{w}d'] = volumes / avg_vol
        features[f'vol_trend_{w}d'] = avg_vol / avg_vol.shift(w) - 1
    return features


def compute_dollar_volume(prices: pd.DataFrame, volumes: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Dollar volume features."""
    dollar_vol = prices * volumes
    features = {}
    for w in windows:
        features[f'dollar_vol_{w}d'] = dollar_vol.rolling(w).mean()
    return features


def compute_volume_price_correlation(returns: pd.DataFrame, volumes: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Rolling correlation between returns and volume."""
    features = {}
    for w in windows:
        corr = returns.rolling(w).corr(volumes)
        features[f'vol_price_corr_{w}d'] = corr
    return features


def compute_amihud_illiquidity(returns: pd.DataFrame, volumes: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Amihud illiquidity measure."""
    features = {}
    illiq = returns.abs() / volumes
    for w in windows:
        features[f'amihud_{w}d'] = illiq.rolling(w).mean()
    return features


def compute_volume_momentum(volumes: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Volume momentum."""
    features = {}
    for w in windows:
        features[f'vol_mom_{w}d'] = volumes.pct_change(w)
    return features
