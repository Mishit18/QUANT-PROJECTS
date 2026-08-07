import pandas as pd
import numpy as np
from typing import Dict


def neutralize_sector(alpha: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Remove sector exposure from alpha."""
    neutralized = alpha.copy()
    sector_series = pd.Series(sector_map).reindex(alpha.columns)

    for sector in sector_series.dropna().unique():
        sector_tickers = sector_series[sector_series == sector].index.intersection(alpha.columns)
        if len(sector_tickers) > 1:
            sector_alpha = alpha[sector_tickers]
            neutralized[sector_tickers] = sector_alpha.sub(sector_alpha.mean(axis=1), axis=0)

    return neutralized


def compute_sector_neutral_weights(alpha: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Construct sector-neutral portfolio weights."""
    sector_series = pd.Series(sector_map)
    weights = pd.DataFrame(index=alpha.index, columns=alpha.columns)
    
    for date in alpha.index:
        alpha_row = alpha.loc[date]
        weight_row = pd.Series(0.0, index=alpha_row.index)
        
        for sector in sector_series.unique():
            sector_tickers = sector_series[sector_series == sector].index
            sector_tickers = sector_tickers.intersection(alpha_row.index)
            
            if len(sector_tickers) > 1:
                sector_alpha = alpha_row[sector_tickers]
                valid = sector_alpha.notna()
                
                if valid.sum() > 1:
                    sector_weights = sector_alpha[valid] - sector_alpha[valid].mean()
                    sector_weights = sector_weights / sector_weights.abs().sum()
                    weight_row[sector_tickers[valid]] = sector_weights
        
        if weight_row.abs().sum() > 0:
            weights.loc[date] = weight_row / weight_row.abs().sum()
    
    return weights


def compute_sector_exposures(weights: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Calculate net sector exposures."""
    sector_series = pd.Series(sector_map)
    exposures = pd.DataFrame(index=weights.index, columns=sector_series.unique())
    
    for date in weights.index:
        weight_row = weights.loc[date]
        for sector in sector_series.unique():
            sector_tickers = sector_series[sector_series == sector].index
            sector_tickers = sector_tickers.intersection(weight_row.index)
            exposures.loc[date, sector] = weight_row[sector_tickers].sum()
    
    return exposures
