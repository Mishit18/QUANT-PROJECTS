import pandas as pd
import numpy as np
from typing import Optional, Tuple


def compute_forward_returns(prices: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Forward returns over horizon."""
    return prices.pct_change(horizon, fill_method=None).shift(-horizon)


def cross_sectional_rank(returns: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
    """Rank returns cross-sectionally per date, scaled to [-scale, +scale]."""
    def rank_row(row):
        valid = row.notna()
        if valid.sum() < 2:
            return pd.Series(np.nan, index=row.index)
        ranks = row[valid].rank(pct=True)
        scaled = 2 * scale * (ranks - 0.5)
        result = pd.Series(np.nan, index=row.index)
        result[valid] = scaled
        return result
    
    return returns.apply(rank_row, axis=1)


def volatility_scaled_returns(returns: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Scale returns by trailing realized volatility."""
    vol = returns.rolling(vol_window).std()
    return returns / vol.replace(0, np.nan)


def residualized_returns(returns: pd.DataFrame, market_returns: pd.Series, 
                        window: int = 60) -> pd.DataFrame:
    """Market-neutral returns via rolling beta residualization."""
    residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for col in returns.columns:
        asset_ret = returns[col]
        for i in range(window, len(returns)):
            y = asset_ret.iloc[i-window:i]
            x = market_returns.iloc[i-window:i]
            
            valid = y.notna() & x.notna()
            if valid.sum() < window // 2:
                continue
            
            cov = np.cov(x[valid], y[valid])[0, 1]
            var = np.var(x[valid])
            beta = cov / var if var > 1e-8 else 0
            
            residuals.iloc[i, residuals.columns.get_loc(col)] = asset_ret.iloc[i] - beta * market_returns.iloc[i]
    
    return residuals


def construct_targets(prices: pd.DataFrame, config: dict, 
                     market_returns: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, dict]:
    """Construct prediction targets based on config."""
    horizon = config['targets'].get('horizon', 1)
    method = config['targets'].get('method', 'rank')
    rank_scale = config['targets'].get('rank_scale', 0.5)
    vol_window = config['targets'].get('vol_window', 20)
    
    forward_returns = compute_forward_returns(prices, horizon)
    
    if method == 'rank':
        targets = cross_sectional_rank(forward_returns, scale=rank_scale)
    elif method == 'vol_scaled':
        targets = volatility_scaled_returns(forward_returns, vol_window)
    elif method == 'residualized':
        if market_returns is None:
            market_returns = forward_returns.mean(axis=1)
        targets = residualized_returns(forward_returns, market_returns)
    elif method == 'raw':
        targets = forward_returns
    else:
        raise ValueError(f"Unknown target method: {method}")
    
    # Validation statistics
    stats = {
        'method': method,
        'horizon': horizon,
        'n_dates': len(targets),
        'n_assets': targets.shape[1],
        'mean_cs_std': targets.std(axis=1).mean(),
        'mean_cs_valid': targets.notna().sum(axis=1).mean(),
        'target_min': targets.min().min(),
        'target_max': targets.max().max(),
        'target_mean': targets.mean().mean(),
        'target_std': targets.std().std()
    }
    
    return targets, stats


def validate_target_alignment(features: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Validate feature-target alignment."""
    feat_dates = features.index.get_level_values(0).unique()
    targ_dates = targets.index if targets.index.nlevels == 1 else targets.index.get_level_values(0).unique()
    
    common_dates = feat_dates.intersection(targ_dates)
    
    validation = {
        'feature_dates': len(feat_dates),
        'target_dates': len(targ_dates),
        'common_dates': len(common_dates),
        'alignment_pct': len(common_dates) / len(feat_dates) * 100 if len(feat_dates) > 0 else 0
    }
    
    return validation
