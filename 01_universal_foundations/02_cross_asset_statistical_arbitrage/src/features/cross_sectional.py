import pandas as pd
import numpy as np
from typing import Dict
from ..utils.helpers import rank_normalize, zscore, demean


def apply_cross_sectional_transforms(features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Apply rank, z-score, demean to all features."""
    transformed = {}
    
    for name, df in features.items():
        transformed[f'{name}_rank'] = df.apply(rank_normalize, axis=1)
        transformed[f'{name}_z'] = df.apply(zscore, axis=1)
        transformed[f'{name}_demean'] = df.apply(demean, axis=1)
    
    return transformed


def compute_cross_sectional_dispersion(features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Cross-sectional dispersion metrics."""
    dispersion = {}
    
    for name, df in features.items():
        dispersion[f'{name}_cs_std'] = df.std(axis=1).to_frame()
        dispersion[f'{name}_cs_iqr'] = (df.quantile(0.75, axis=1) - df.quantile(0.25, axis=1)).to_frame()
    
    return dispersion


def compute_relative_strength(feature: pd.DataFrame, quantiles: list = [0.2, 0.8]) -> pd.DataFrame:
    """Relative position in cross-sectional distribution."""
    def percentile_rank(row):
        valid = row.notna()
        if valid.sum() < 10:
            return pd.Series(np.nan, index=row.index)
        return row[valid].rank(pct=True).reindex(row.index)
    
    return feature.apply(percentile_rank, axis=1)


def compute_sector_relative(feature: pd.DataFrame, sector_map: Dict[str, str]) -> pd.DataFrame:
    """Feature relative to sector mean."""
    sector_df = pd.Series(sector_map)
    
    def sector_demean(row):
        result = pd.Series(np.nan, index=row.index)
        for sector in sector_df.unique():
            sector_tickers = sector_df[sector_df == sector].index
            sector_tickers = sector_tickers.intersection(row.index)
            if len(sector_tickers) > 1:
                sector_mean = row[sector_tickers].mean()
                result[sector_tickers] = row[sector_tickers] - sector_mean
        return result
    
    return feature.apply(sector_demean, axis=1)


def compute_industry_momentum(returns: pd.DataFrame, sector_map: Dict[str, str], window: int = 20) -> pd.DataFrame:
    """Industry momentum spillover."""
    sector_df = pd.Series(sector_map)
    industry_mom = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for sector in sector_df.unique():
        sector_tickers = sector_df[sector_df == sector].index.intersection(returns.columns)
        if len(sector_tickers) > 1:
            sector_ret = returns[sector_tickers].mean(axis=1)
            sector_mom = sector_ret.rolling(window).mean()
            industry_mom[sector_tickers] = sector_mom.values[:, np.newaxis]
    
    return industry_mom
