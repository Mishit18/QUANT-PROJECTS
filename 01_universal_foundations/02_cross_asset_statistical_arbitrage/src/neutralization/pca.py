import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple


def compute_pca_factors(returns: pd.DataFrame, n_components: int = 10, window: int = 252) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling PCA factor extraction."""
    factors = []
    loadings_list = []
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i - window:i]
        valid_cols = window_returns.notna().sum() > window * 0.8
        clean_returns = window_returns.loc[:, valid_cols].fillna(0)
        
        if clean_returns.shape[1] > n_components:
            pca = PCA(n_components=n_components)
            pca.fit(clean_returns)
            
            current_return = returns.iloc[i]
            factor_exposure = pca.transform(current_return[valid_cols].fillna(0).values.reshape(1, -1))
            
            factor_series = pd.Series(factor_exposure[0], 
                                     index=[f'PC{j+1}' for j in range(n_components)],
                                     name=returns.index[i])
            factors.append(factor_series)
            
            loadings = pd.DataFrame(pca.components_.T, 
                                   index=clean_returns.columns,
                                   columns=[f'PC{j+1}' for j in range(n_components)])
            loadings['date'] = returns.index[i]
            loadings_list.append(loadings)
    
    factor_df = pd.DataFrame(factors)
    return factor_df, loadings_list


def neutralize_pca_factors(alpha: pd.DataFrame, returns: pd.DataFrame, 
                           n_factors: int = 10, window: int = 252) -> pd.DataFrame:
    """Remove PCA factor exposure from alpha."""
    if len(alpha) == 0:
        return alpha
    
    neutralized = alpha.copy()
    
    for i in range(window, len(returns)):
        date = returns.index[i]
        if date not in alpha.index:
            continue
            
        window_returns = returns.iloc[i - window:i]
        valid_cols = window_returns.notna().sum() > window * 0.8
        common_cols = valid_cols[valid_cols].index.intersection(alpha.columns)
        
        if len(common_cols) < n_factors:
            continue
        
        clean_returns = window_returns[common_cols].fillna(0)
        
        if clean_returns.shape[1] > n_factors:
            pca = PCA(n_components=n_factors)
            pca.fit(clean_returns)
            
            alpha_row = alpha.loc[date, common_cols].fillna(0)
            
            for j in range(n_factors):
                loading = pca.components_[j]
                factor_exposure = (alpha_row.values * loading).sum()
                neutralized.loc[date, common_cols] -= factor_exposure * loading
    
    return neutralized


def compute_factor_exposures(weights: pd.DataFrame, returns: pd.DataFrame, 
                             n_factors: int = 10, window: int = 252) -> pd.DataFrame:
    """Calculate portfolio factor exposures."""
    exposures = []
    
    for i in range(window, len(returns)):
        date = returns.index[i]
        if date not in weights.index:
            continue
            
        window_returns = returns.iloc[i - window:i]
        valid_cols = window_returns.notna().sum() > window * 0.8
        clean_returns = window_returns.loc[:, valid_cols].fillna(0)
        
        if clean_returns.shape[1] > n_factors:
            pca = PCA(n_components=n_factors)
            pca.fit(clean_returns)
            
            weight_row = weights.loc[date, valid_cols].fillna(0)
            
            factor_exp = {}
            for j in range(n_factors):
                loading = pca.components_[j]
                factor_exp[f'PC{j+1}'] = (weight_row.values * loading).sum()
            
            exposures.append(pd.Series(factor_exp, name=date))
    
    return pd.DataFrame(exposures)
