"""
Regime-based feature engineering and signal conditioning.

Combines Kalman-filtered latent states with HMM regime probabilities
to create regime-aware features for forecasting and trading.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from src.kalman_filter import KalmanFilter
from src.hmm_regimes import GaussianHMM
from src.utils import realized_volatility


class RegimeFeatureEngine:
    """
    Feature engineering combining latent states and regime information.
    """
    
    def __init__(self, 
                 kalman_filter: Optional[KalmanFilter] = None,
                 hmm: Optional[GaussianHMM] = None):
        """
        Initialize feature engine.
        
        Parameters
        ----------
        kalman_filter : KalmanFilter, optional
            Fitted Kalman filter
        hmm : GaussianHMM, optional
            Fitted HMM
        """
        self.kf = kalman_filter
        self.hmm = hmm
    
    def create_regime_features(self, 
                              returns: np.ndarray,
                              regime_probs: np.ndarray,
                              filtered_states: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Create comprehensive regime-based features.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
        regime_probs : np.ndarray
            Regime probabilities (n_samples, n_regimes)
        filtered_states : np.ndarray, optional
            Kalman-filtered states
            
        Returns
        -------
        pd.DataFrame
            Feature matrix
        """
        n_regimes = regime_probs.shape[1]
        features = pd.DataFrame()
        
        # Raw regime probabilities
        for k in range(n_regimes):
            features[f'regime_{k}_prob'] = regime_probs[:, k]
        
        # Regime uncertainty (entropy)
        epsilon = 1e-10
        entropy = -np.sum(regime_probs * np.log(regime_probs + epsilon), axis=1)
        features['regime_entropy'] = entropy
        
        # Dominant regime
        features['dominant_regime'] = np.argmax(regime_probs, axis=1)
        
        # Regime transition indicators
        regime_changes = np.diff(features['dominant_regime'].values, prepend=features['dominant_regime'].iloc[0])
        features['regime_changed'] = (regime_changes != 0).astype(int)
        
        # Regime persistence (time since last change)
        persistence = np.zeros(len(returns))
        counter = 0
        for i in range(len(returns)):
            if features['regime_changed'].iloc[i]:
                counter = 0
            else:
                counter += 1
            persistence[i] = counter
        features['regime_persistence'] = persistence
        
        # Regime-conditional statistics
        for k in range(n_regimes):
            # Weighted returns by regime probability
            features[f'regime_{k}_weighted_return'] = returns * regime_probs[:, k]
            
            # Regime-conditional volatility
            window = 20
            weighted_vol = np.zeros(len(returns))
            for i in range(window, len(returns)):
                weighted_returns = returns[i-window:i] * regime_probs[i-window:i, k]
                weighted_vol[i] = np.std(weighted_returns) * np.sqrt(252)
            features[f'regime_{k}_volatility'] = weighted_vol
        
        # Kalman filter features
        if filtered_states is not None:
            if filtered_states.ndim == 1:
                features['kf_state'] = filtered_states
            else:
                for i in range(filtered_states.shape[1]):
                    features[f'kf_state_{i}'] = filtered_states[:, i]
            
            # Regime-conditioned Kalman states
            for k in range(n_regimes):
                if filtered_states.ndim == 1:
                    features[f'regime_{k}_kf_state'] = filtered_states * regime_probs[:, k]
                else:
                    features[f'regime_{k}_kf_state_0'] = filtered_states[:, 0] * regime_probs[:, k]
        
        # Realized volatility
        rv = realized_volatility(returns, window=20)
        features['realized_vol'] = rv
        
        # Regime-relative volatility
        for k in range(n_regimes):
            features[f'regime_{k}_vol_ratio'] = features['realized_vol'] / (features[f'regime_{k}_volatility'] + 1e-10)
        
        return features
    
    def regime_conditional_moments(self, 
                                   returns: np.ndarray,
                                   regime_probs: np.ndarray,
                                   window: int = 60) -> Dict[str, np.ndarray]:
        """
        Calculate regime-conditional moments (mean, variance, skewness, kurtosis).
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
        regime_probs : np.ndarray
            Regime probabilities
        window : int
            Rolling window size
            
        Returns
        -------
        dict
            Regime-conditional moments
        """
        n_regimes = regime_probs.shape[1]
        n_samples = len(returns)
        
        moments = {
            'mean': np.zeros((n_samples, n_regimes)),
            'variance': np.zeros((n_samples, n_regimes)),
            'skewness': np.zeros((n_samples, n_regimes)),
            'kurtosis': np.zeros((n_samples, n_regimes))
        }
        
        for t in range(window, n_samples):
            window_returns = returns[t-window:t]
            window_probs = regime_probs[t-window:t]
            
            for k in range(n_regimes):
                weights = window_probs[:, k]
                weights = weights / (weights.sum() + 1e-10)
                
                # Weighted mean
                mean = np.sum(weights * window_returns)
                moments['mean'][t, k] = mean
                
                # Weighted variance
                variance = np.sum(weights * (window_returns - mean) ** 2)
                moments['variance'][t, k] = variance
                
                # Weighted skewness
                if variance > 0:
                    skewness = np.sum(weights * ((window_returns - mean) / np.sqrt(variance)) ** 3)
                    moments['skewness'][t, k] = skewness
                
                # Weighted kurtosis
                if variance > 0:
                    kurtosis = np.sum(weights * ((window_returns - mean) / np.sqrt(variance)) ** 4)
                    moments['kurtosis'][t, k] = kurtosis
        
        return moments
    
    def regime_transition_features(self, regime_probs: np.ndarray) -> pd.DataFrame:
        """
        Create features based on regime transition dynamics.
        
        Parameters
        ----------
        regime_probs : np.ndarray
            Regime probabilities
            
        Returns
        -------
        pd.DataFrame
            Transition features
        """
        n_regimes = regime_probs.shape[1]
        features = pd.DataFrame()
        
        # Probability changes (regime momentum)
        for k in range(n_regimes):
            prob_change = np.diff(regime_probs[:, k], prepend=regime_probs[0, k])
            features[f'regime_{k}_prob_change'] = prob_change
            
            # Smoothed probability change
            features[f'regime_{k}_prob_ma'] = pd.Series(regime_probs[:, k]).rolling(5).mean().values
        
        # Regime transition probability (sum of off-diagonal probabilities)
        for k in range(n_regimes):
            transition_prob = 1 - regime_probs[:, k]
            features[f'regime_{k}_transition_prob'] = transition_prob
        
        # Expected regime in next period
        # E[s_{t+1} | y_{1:t}] = sum_k k * P(s_{t+1} = k | y_{1:t})
        features['expected_next_regime'] = np.sum(
            regime_probs * np.arange(n_regimes), axis=1
        )
        
        return features
    
    def create_regime_indicators(self, 
                                 regime_probs: np.ndarray,
                                 threshold: float = 0.7) -> pd.DataFrame:
        """
        Create binary regime indicators based on probability threshold.
        
        Parameters
        ----------
        regime_probs : np.ndarray
            Regime probabilities
        threshold : float
            Probability threshold for regime assignment
            
        Returns
        -------
        pd.DataFrame
            Binary regime indicators
        """
        n_regimes = regime_probs.shape[1]
        indicators = pd.DataFrame()
        
        for k in range(n_regimes):
            indicators[f'in_regime_{k}'] = (regime_probs[:, k] > threshold).astype(int)
        
        # High confidence indicator (any regime above threshold)
        indicators['high_confidence'] = (regime_probs.max(axis=1) > threshold).astype(int)
        
        # Low confidence indicator (no regime above threshold)
        indicators['low_confidence'] = (regime_probs.max(axis=1) < threshold).astype(int)
        
        return indicators


def combine_kalman_hmm_features(returns: np.ndarray,
                                kf: KalmanFilter,
                                hmm: GaussianHMM) -> pd.DataFrame:
    """
    Convenience function to create combined Kalman-HMM features.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    kf : KalmanFilter
        Fitted Kalman filter
    hmm : GaussianHMM
        Fitted HMM
        
    Returns
    -------
    pd.DataFrame
        Combined feature matrix
    """
    # Get filtered states
    filtered_states = kf.filtered_states
    
    # Get regime probabilities
    regime_probs = hmm.predict_proba(returns)
    
    # Create feature engine
    engine = RegimeFeatureEngine(kf, hmm)
    
    # Generate features
    features = engine.create_regime_features(returns, regime_probs, filtered_states)
    transition_features = engine.regime_transition_features(regime_probs)
    indicators = engine.create_regime_indicators(regime_probs)
    
    # Combine all features
    combined = pd.concat([features, transition_features, indicators], axis=1)
    
    return combined


if __name__ == '__main__':
    # Test regime features
    from src.data_loader import generate_synthetic_data
    from src.state_space_models import LocalLevelModel
    
    print("Testing regime feature engineering...")
    
    # Generate data
    data = generate_synthetic_data(n_samples=500, seed=42)
    returns = data['returns'].iloc[:, 0].values
    
    # Fit models
    model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
    kf = KalmanFilter(model)
    kf.filter(returns)
    
    hmm = GaussianHMM(n_regimes=3, random_state=42)
    hmm.fit(returns)
    
    # Create features
    features = combine_kalman_hmm_features(returns, kf, hmm)
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()[:10]}...")
    print(f"\nSample features:")
    print(features.head())
