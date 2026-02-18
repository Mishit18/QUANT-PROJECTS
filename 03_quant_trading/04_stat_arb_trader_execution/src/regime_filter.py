"""
Regime detection for binary risk gating.

Purpose: Enforce OU stationarity assumptions by preventing new positions
in volatile regimes where mean-reversion dynamics are unreliable.

This is NOT return optimization - it enforces model validity.
Volatile regimes violate the stationarity assumption underlying OU models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler


class RegimeFilter:
    """
    HMM-based regime detection for binary risk gating.
    
    Purpose: Enforce OU stationarity assumptions by blocking new positions
    in volatile regimes. Existing positions are allowed to exit naturally.
    
    Rationale: OU mean-reversion assumes stationary dynamics. High volatility
    regimes violate this assumption, making parameter estimates unreliable.
    """
    
    def __init__(self, n_regimes: int = 2, random_state: int = 42):
        """
        Args:
            n_regimes: 2 (simple: calm/volatile) or 3 (calm/normal/volatile)
        """
        self.n_regimes = n_regimes
        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type='full',
            n_iter=100,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.fitted = False
        self.regime_labels = {}
    
    def fit(self, spread: pd.Series, returns: pd.Series = None) -> 'RegimeFilter':
        """
        Fit HMM to spread volatility (primary feature).
        
        Args:
            spread: Spread series
            returns: Optional market returns (for correlation)
        """
        # Simple features: spread volatility is what matters
        features = pd.DataFrame(index=spread.index)
        
        # Rolling volatility (20-day)
        features['spread_vol'] = spread.pct_change().rolling(20).std()
        
        # Spread absolute level
        features['spread_abs'] = spread.abs()
        
        # Spread change
        features['spread_change'] = spread.diff().abs()
        
        features = features.dropna()
        
        if len(features) < 100:
            self.fitted = False
            return self
        
        # Fit HMM
        X = self.scaler.fit_transform(features.values)
        self.model.fit(X)
        
        # Label regimes by volatility
        states = self.model.predict(X)
        
        regime_vols = {}
        for regime in range(self.n_regimes):
            mask = states == regime
            regime_vols[regime] = features.loc[mask, 'spread_vol'].mean()
        
        # Sort by volatility
        sorted_regimes = sorted(regime_vols.items(), key=lambda x: x[1])
        
        if self.n_regimes == 2:
            self.regime_labels = {
                sorted_regimes[0][0]: 'calm',
                sorted_regimes[1][0]: 'volatile'
            }
        else:  # 3 regimes
            self.regime_labels = {
                sorted_regimes[0][0]: 'calm',
                sorted_regimes[1][0]: 'normal',
                sorted_regimes[2][0]: 'volatile'
            }
        
        self.fitted = True
        return self
    
    def predict_gate(self, spread: pd.Series) -> pd.Series:
        """
        Return binary gate for each time point.
        
        Binary gating enforces OU stationarity assumptions:
        - Volatile regime: Block new positions (gate = 0)
        - Calm/normal regime: Allow positions (gate = 1)
        
        This is NOT return optimization. It enforces model validity.
        OU parameters estimated in calm regimes are unreliable in volatile regimes.
        
        Returns:
            Series of binary gates (0 = block new positions, 1 = allow)
        """
        if not self.fitted:
            return pd.Series(1, index=spread.index)
        
        # Prepare features
        features = pd.DataFrame(index=spread.index)
        features['spread_vol'] = spread.pct_change().rolling(20).std()
        features['spread_abs'] = spread.abs()
        features['spread_change'] = spread.diff().abs()
        
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        X = self.scaler.transform(features.values)
        states = self.model.predict(X)
        
        # Map states to binary gates
        # Rationale: Volatile regimes violate stationarity assumption
        gates = np.ones(len(states), dtype=int)
        
        for i, state in enumerate(states):
            label = self.regime_labels.get(state, 'normal')
            
            if label == 'volatile':
                gates[i] = 0  # Block new positions in volatile regime
            else:
                gates[i] = 1  # Allow positions in calm/normal regimes
        
        return pd.Series(gates, index=spread.index)
    
    def predict_multiplier(self, spread: pd.Series) -> pd.Series:
        """
        Return regime-based position multipliers (legacy method for backward compatibility).
        
        For single-pair mode, uses scaling approach:
        - Calm regime: 1.3x multiplier
        - Normal regime: 1.0x multiplier
        - Volatile regime: 0.4x multiplier
        
        Note: Portfolio mode uses predict_gate() for binary gating instead.
        
        Returns:
            Series of position multipliers
        """
        if not self.fitted:
            return pd.Series(1.0, index=spread.index)
        
        # Prepare features
        features = pd.DataFrame(index=spread.index)
        features['spread_vol'] = spread.pct_change().rolling(20).std()
        features['spread_abs'] = spread.abs()
        features['spread_change'] = spread.diff().abs()
        
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        X = self.scaler.transform(features.values)
        states = self.model.predict(X)
        
        # Map states to multipliers
        multipliers = np.ones(len(states))
        
        for i, state in enumerate(states):
            label = self.regime_labels.get(state, 'normal')
            
            if label == 'calm':
                multipliers[i] = 1.3  # Increase size in calm regimes
            elif label == 'normal':
                multipliers[i] = 1.0  # Normal size
            elif label == 'volatile':
                multipliers[i] = 0.4  # Reduce size in volatile regimes
        
        return pd.Series(multipliers, index=spread.index)
    
    def get_regime_stats(self, spread: pd.Series) -> pd.DataFrame:
        """
        Get regime statistics for diagnostics.
        """
        if not self.fitted:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=spread.index)
        features['spread_vol'] = spread.pct_change().rolling(20).std()
        features['spread_abs'] = spread.abs()
        features['spread_change'] = spread.diff().abs()
        features = features.fillna(method='bfill').fillna(method='ffill')
        
        X = self.scaler.transform(features.values)
        states = self.model.predict(X)
        
        stats = []
        for regime in range(self.n_regimes):
            mask = states == regime
            label = self.regime_labels.get(regime, f'regime_{regime}')
            
            stats.append({
                'regime': label,
                'frequency': mask.sum() / len(states),
                'mean_vol': features.loc[mask, 'spread_vol'].mean(),
                'mean_abs_spread': features.loc[mask, 'spread_abs'].mean()
            })
        
        return pd.DataFrame(stats)
