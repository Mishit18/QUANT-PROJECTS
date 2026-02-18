import pandas as pd
import numpy as np
from typing import Dict, Optional
from .market import neutralize_market_beta
from .sector import neutralize_sector
from .pca import neutralize_pca_factors


class RiskNeutralizer:
    def __init__(self, config: dict):
        self.config = config
        self.market_neutral = config['neutralization']['market_beta']
        self.sector_neutral = config['neutralization']['sector']
        self.pca_factors = config['neutralization']['pca_factors']
        
    def neutralize(self, alpha: pd.DataFrame, returns: pd.DataFrame,
                   market_returns: Optional[pd.Series] = None,
                   sector_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Apply full neutralization pipeline."""
        neutralized = alpha.copy()
        
        if self.market_neutral and market_returns is not None:
            neutralized = neutralize_market_beta(neutralized, returns, market_returns)
        
        if self.sector_neutral and sector_map is not None:
            neutralized = neutralize_sector(neutralized, sector_map)
        
        if self.pca_factors > 0:
            neutralized = neutralize_pca_factors(neutralized, returns, self.pca_factors)
        
        return neutralized
    
    def compute_residual_returns(self, returns: pd.DataFrame,
                                market_returns: Optional[pd.Series] = None,
                                sector_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Compute risk-adjusted residual returns."""
        residuals = returns.copy()
        
        if self.market_neutral and market_returns is not None:
            from .market import compute_market_beta
            betas = compute_market_beta(returns, market_returns)
            for date in returns.index:
                if date in betas.index and date in market_returns.index:
                    residuals.loc[date] = returns.loc[date] - betas.loc[date] * market_returns.loc[date]
        
        if self.sector_neutral and sector_map is not None:
            sector_series = pd.Series(sector_map)
            for date in residuals.index:
                for sector in sector_series.unique():
                    sector_tickers = sector_series[sector_series == sector].index
                    sector_tickers = sector_tickers.intersection(residuals.columns)
                    if len(sector_tickers) > 1:
                        sector_mean = residuals.loc[date, sector_tickers].mean()
                        residuals.loc[date, sector_tickers] -= sector_mean
        
        return residuals
