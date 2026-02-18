import pandas as pd
import numpy as np


def compute_transaction_costs(weights_prev: pd.Series, weights_curr: pd.Series, 
                              tcost_bps: float = 7.5) -> float:
    """Linear transaction cost model."""
    common = weights_prev.index.intersection(weights_curr.index)
    
    w_prev = weights_prev[common].fillna(0)
    w_curr = weights_curr[common].fillna(0)
    
    turnover = (w_prev - w_curr).abs().sum() / 2
    return turnover * tcost_bps / 10000


def compute_slippage(weights_prev: pd.Series, weights_curr: pd.Series,
                    volumes: pd.Series, slippage_bps: float = 2.5) -> float:
    """Volume-dependent slippage."""
    common = weights_prev.index.intersection(weights_curr.index).intersection(volumes.index)
    
    w_prev = weights_prev[common].fillna(0)
    w_curr = weights_curr[common].fillna(0)
    vol = volumes[common].fillna(volumes[common].median())
    
    trade_size = (w_prev - w_curr).abs()
    
    vol_normalized = vol / vol.median()
    slippage_factor = 1.0 / np.sqrt(vol_normalized)
    
    total_slippage = (trade_size * slippage_factor).sum() * slippage_bps / 10000
    return total_slippage


def compute_market_impact(trade_size: float, adv: float, participation_rate: float = 0.1) -> float:
    """Square-root market impact model."""
    if adv <= 0:
        return 0.0
    
    pct_adv = trade_size / adv
    impact_bps = 10 * np.sqrt(pct_adv / participation_rate)
    
    return impact_bps / 10000


def compute_total_costs(weights_prev: pd.Series, weights_curr: pd.Series,
                       volumes: pd.Series, tcost_bps: float = 7.5, 
                       slippage_bps: float = 2.5) -> float:
    """Combined transaction costs and slippage."""
    tc = compute_transaction_costs(weights_prev, weights_curr, tcost_bps)
    slip = compute_slippage(weights_prev, weights_curr, volumes, slippage_bps)
    return tc + slip
