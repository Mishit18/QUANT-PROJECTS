import pandas as pd
import numpy as np
from typing import Tuple


def compute_market_beta(returns: pd.DataFrame, market_returns: pd.Series, window: int = 252) -> pd.DataFrame:
    """Rolling market beta estimation."""
    betas = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for col in returns.columns:
        asset_ret = returns[col]
        valid = asset_ret.notna() & market_returns.notna()
        
        for i in range(window, len(returns)):
            window_slice = slice(i - window, i)
            y = asset_ret.iloc[window_slice][valid.iloc[window_slice]]
            x = market_returns.iloc[window_slice][valid.iloc[window_slice]]
            
            if len(y) > window // 2:
                cov = np.cov(x, y)[0, 1]
                var = np.var(x)
                betas.iloc[i, betas.columns.get_loc(col)] = cov / var if var > 0 else np.nan
    
    return betas


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
                market_exposure = (alpha_row[valid] * beta_row[valid]).sum() / beta_row[valid].abs().sum()
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
