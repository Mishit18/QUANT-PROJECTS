import pandas as pd
import numpy as np
from typing import Optional


class Portfolio:
    """Long/short portfolio construction."""
    
    def __init__(self, leverage: float = 1.0, long_short: bool = True):
        self.leverage = leverage
        self.long_short = long_short
        self.weights = []
        
    def construct_weights(self, alpha: pd.Series, method: str = 'rank') -> pd.Series:
        """Convert alpha to portfolio weights."""
        valid = alpha.notna()
        if valid.sum() < 10:
            return pd.Series(0.0, index=alpha.index)
        
        if method == 'rank':
            weights = alpha[valid].rank(pct=True) - 0.5
        elif method == 'zscore':
            mean = alpha[valid].mean()
            std = alpha[valid].std()
            weights = (alpha[valid] - mean) / std if std > 0 else alpha[valid] * 0
        elif method == 'raw':
            weights = alpha[valid]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if not self.long_short:
            weights = weights.clip(lower=0)
        
        if weights.abs().sum() > 0:
            weights = weights / weights.abs().sum() * self.leverage
        
        result = pd.Series(0.0, index=alpha.index)
        result[valid] = weights
        
        return result
    
    def apply_constraints(self, weights: pd.Series, max_weight: float = 0.05,
                         max_long: Optional[float] = None, 
                         max_short: Optional[float] = None) -> pd.Series:
        """Apply position limits."""
        weights = weights.clip(-max_weight, max_weight)
        
        if max_long is not None:
            long_weights = weights[weights > 0]
            if long_weights.sum() > max_long:
                weights[weights > 0] *= max_long / long_weights.sum()
        
        if max_short is not None:
            short_weights = weights[weights < 0]
            if short_weights.abs().sum() > max_short:
                weights[weights < 0] *= max_short / short_weights.abs().sum()
        
        return weights
    
    def rebalance(self, target_weights: pd.Series, current_weights: pd.Series,
                 threshold: float = 0.01) -> pd.Series:
        """Rebalance only if drift exceeds threshold."""
        diff = (target_weights - current_weights).abs()
        
        if diff.max() < threshold:
            return current_weights
        
        return target_weights
