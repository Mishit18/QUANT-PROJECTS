import pandas as pd
import numpy as np
from typing import Dict


def compute_realized_vol(returns: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Rolling realized volatility."""
    features = {}
    for w in windows:
        features[f'vol_{w}d'] = returns.rolling(w).std() * np.sqrt(252)
    return features


def compute_parkinson_vol(high: pd.DataFrame, low: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Parkinson high-low volatility estimator."""
    features = {}
    hl_ratio = np.log(high / low) ** 2
    for w in windows:
        features[f'parkinson_vol_{w}d'] = np.sqrt(hl_ratio.rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)
    return features


def compute_garman_klass_vol(open_p: pd.DataFrame, high: pd.DataFrame, 
                             low: pd.DataFrame, close: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Garman-Klass volatility estimator."""
    features = {}
    
    hl = np.log(high / low) ** 2
    co = np.log(close / open_p) ** 2
    
    for w in windows:
        gk = 0.5 * hl - (2 * np.log(2) - 1) * co
        features[f'gk_vol_{w}d'] = np.sqrt(gk.rolling(w).mean()) * np.sqrt(252)
    
    return features


def compute_vol_of_vol(returns: pd.DataFrame, vol_window: int = 20, volvol_window: int = 60) -> pd.DataFrame:
    """Volatility of volatility."""
    vol = returns.rolling(vol_window).std()
    volvol = vol.rolling(volvol_window).std()
    return volvol


def compute_downside_vol(returns: pd.DataFrame, windows: list) -> Dict[str, pd.DataFrame]:
    """Downside semi-deviation."""
    features = {}
    for w in windows:
        downside = returns.clip(upper=0)
        features[f'downside_vol_{w}d'] = downside.rolling(w).std() * np.sqrt(252)
    return features


def compute_vol_regime(returns: pd.DataFrame, short_window: int = 20, long_window: int = 120) -> pd.DataFrame:
    """Short vol / long vol ratio."""
    short_vol = returns.rolling(short_window).std()
    long_vol = returns.rolling(long_window).std()
    return short_vol / long_vol
