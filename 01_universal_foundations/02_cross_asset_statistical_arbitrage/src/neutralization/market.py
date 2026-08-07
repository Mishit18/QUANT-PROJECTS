import pandas as pd
import numpy as np
from typing import Tuple


def compute_market_beta(returns: pd.DataFrame, market_returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling market beta estimation."""
    aligned_market = market_returns.reindex(returns.index)
    min_periods = max(window // 2, 20)
    market_var = aligned_market.rolling(window, min_periods=min_periods).var()
    cov = returns.rolling(window, min_periods=min_periods).cov(aligned_market)
    betas = cov.div(market_var.replace(0, np.nan), axis=0)
    return betas.astype(float)


def neutralize_market_beta(alpha: pd.DataFrame, returns: pd.DataFrame, 
                          market_returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Remove market beta exposure from alpha."""
    betas = compute_market_beta(returns, market_returns, window)
    
    neutralized = alpha.copy()
    for date in alpha.index:
        if date in betas.index:
            beta_row = betas.loc[date]
            alpha_row = alpha.loc[date]
            
            valid = alpha_row.notna() & beta_row.notna()
            if valid.sum() > 10:
                beta = beta_row[valid].astype(float)
                denom = float(beta @ beta)
                if denom <= 1e-12:
                    continue
                market_exposure = float(alpha_row[valid] @ beta) / denom
                neutralized.loc[date] = alpha_row - beta_row * market_exposure
    
    return neutralized


def compute_market_neutral_weights(alpha: pd.DataFrame, betas: pd.DataFrame) -> pd.DataFrame:
    """Construct beta-neutral portfolio weights."""
    weights = pd.DataFrame(index=alpha.index, columns=alpha.columns)
    
    for date in alpha.index:
        if date in betas.index:
            alpha_row = alpha.loc[date]
            beta_row = betas.loc[date]
            
            valid = alpha_row.notna() & beta_row.notna()
            if valid.sum() > 10:
                a = alpha_row[valid].values
                b = beta_row[valid].values
                
                w = a - (a @ b) / (b @ b) * b
                w = w / np.abs(w).sum()
                
                weights.loc[date, valid] = w
    
    return weights
