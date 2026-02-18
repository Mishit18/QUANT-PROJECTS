"""
Trading signal generation using Kalman-filtered states and regime information.

Implements regime-aware signals that adapt to market conditions:
- Trend-following in low-volatility regimes
- Mean-reversion in high-volatility regimes
- Risk-off during crisis regimes
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from src.kalman_filter import KalmanFilter
from src.hmm_regimes import GaussianHMM
from src.regime_features import RegimeFeatureEngine


class BaseSignal:
    """Base class for trading signals."""
    
    def generate(self, data: np.ndarray) -> np.ndarray:
        """
        Generate trading signals.
        
        Parameters
        ----------
        data : np.ndarray
            Input data
            
        Returns
        -------
        np.ndarray
            Signal values (typically -1 to 1)
        """
        raise NotImplementedError


class KalmanTrendSignal(BaseSignal):
    """
    Trend-following signal based on Kalman-filtered trend.
    
    Long when price above trend, short when below.
    """
    
    def __init__(self, kf: KalmanFilter, lookback: int = 20):
        """
        Initialize trend signal.
        
        Parameters
        ----------
        kf : KalmanFilter
            Fitted Kalman filter
        lookback : int
            Lookback for signal smoothing
        """
        self.kf = kf
        self.lookback = lookback
    
    def generate(self, returns: np.ndarray) -> np.ndarray:
        """Generate trend signal."""
        if self.kf.filtered_states is None:
            self.kf.filter(returns)
        
        # Extract trend (first state component)
        if self.kf.filtered_states.ndim == 1:
            trend = self.kf.filtered_states
        else:
            trend = self.kf.filtered_states[:, 0]
        
        # Signal: sign of trend
        signal = np.sign(trend)
        
        # Smooth signal
        signal = pd.Series(signal).rolling(self.lookback, min_periods=1).mean().values
        
        return np.clip(signal, -1, 1)


class RegimeAwareSignal(BaseSignal):
    """
    Regime-aware signal that adapts strategy based on market regime.
    
    - Low-vol regime: Trend-following
    - High-vol regime: Mean-reversion
    - Crisis regime: Risk-off (flat or defensive)
    """
    
    def __init__(self, 
                 kf: KalmanFilter,
                 hmm: GaussianHMM,
                 regime_strategies: Optional[Dict[int, str]] = None):
        """
        Initialize regime-aware signal.
        
        Parameters
        ----------
        kf : KalmanFilter
            Fitted Kalman filter
        hmm : GaussianHMM
            Fitted HMM
        regime_strategies : dict, optional
            Mapping from regime index to strategy type
            {'trend', 'mean_reversion', 'risk_off'}
        """
        self.kf = kf
        self.hmm = hmm
        
        # Default regime strategies (assumes 3 regimes)
        if regime_strategies is None:
            regime_strategies = {
                0: 'trend',           # Low-vol trending
                1: 'mean_reversion',  # High-vol mean-reverting
                2: 'risk_off'         # Crisis
            }
        self.regime_strategies = regime_strategies
    
    def generate(self, returns: np.ndarray) -> np.ndarray:
        """
        Generate regime-aware signal.
        
        PRODUCTION HARDENING:
        - Validates regime probabilities
        - Checks signal bounds
        - Ensures no NaN propagation
        
        Raises
        ------
        RuntimeError
            If signal generation produces invalid values
        """
        # Validate input
        if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
            raise ValueError("Input returns contain NaN/Inf values")
        
        # Get regime probabilities
        regime_probs = self.hmm.predict_proba(returns)
        n_regimes = regime_probs.shape[1]
        
        # Validate regime probabilities
        if np.any(np.isnan(regime_probs)) or np.any(np.isinf(regime_probs)):
            raise RuntimeError("Regime probabilities contain NaN/Inf")
        
        if not np.allclose(regime_probs.sum(axis=1), 1.0, atol=1e-3):
            raise RuntimeError(f"Regime probabilities don't sum to 1: range [{regime_probs.sum(axis=1).min():.6f}, {regime_probs.sum(axis=1).max():.6f}]")
        
        # Get Kalman trend
        if self.kf.filtered_states is None:
            self.kf.filter(returns)
        
        if self.kf.filtered_states.ndim == 1:
            trend = self.kf.filtered_states
        else:
            trend = self.kf.filtered_states[:, 0]
        
        # Validate trend
        if np.any(np.isnan(trend)) or np.any(np.isinf(trend)):
            raise RuntimeError("Kalman trend contains NaN/Inf")
        
        # Generate regime-specific signals
        signals = np.zeros((len(returns), n_regimes))
        
        for k in range(n_regimes):
            strategy = self.regime_strategies.get(k, 'risk_off')
            
            if strategy == 'trend':
                # Trend-following: follow Kalman trend
                signals[:, k] = np.sign(trend)
            
            elif strategy == 'mean_reversion':
                # Mean-reversion: fade the trend
                signals[:, k] = -np.sign(trend)
            
            elif strategy == 'risk_off':
                # Risk-off: flat position
                signals[:, k] = 0.0
        
        # Combine signals weighted by regime probabilities
        combined_signal = np.sum(regime_probs * signals, axis=1)
        
        # Validate combined signal
        if np.any(np.isnan(combined_signal)) or np.any(np.isinf(combined_signal)):
            raise RuntimeError("Combined signal contains NaN/Inf")
        
        # Smooth signal
        combined_signal = pd.Series(combined_signal).rolling(5, min_periods=1).mean().values
        
        # Final validation
        if np.any(np.isnan(combined_signal)) or np.any(np.isinf(combined_signal)):
            raise RuntimeError("Smoothed signal contains NaN/Inf")
        
        # Clip to valid range
        combined_signal = np.clip(combined_signal, -1, 1)
        
        return combined_signal


class VolatilityScaledSignal(BaseSignal):
    """
    Signal that scales position size by inverse volatility.
    
    Reduces exposure in high-volatility periods.
    """
    
    def __init__(self, 
                 base_signal: BaseSignal,
                 kf_volatility: Optional[KalmanFilter] = None,
                 target_vol: float = 0.15):
        """
        Initialize volatility-scaled signal.
        
        Parameters
        ----------
        base_signal : BaseSignal
            Base signal generator
        kf_volatility : KalmanFilter, optional
            Kalman filter for latent volatility
        target_vol : float
            Target annualized volatility
        """
        self.base_signal = base_signal
        self.kf_volatility = kf_volatility
        self.target_vol = target_vol
    
    def generate(self, returns: np.ndarray) -> np.ndarray:
        """Generate volatility-scaled signal."""
        # Base signal
        base = self.base_signal.generate(returns)
        
        # Estimate volatility
        if self.kf_volatility is not None and self.kf_volatility.filtered_states is not None:
            # Use Kalman-filtered volatility
            vol = np.exp(self.kf_volatility.filtered_states.flatten() / 2)
        else:
            # Use rolling realized volatility
            window = 20
            vol_series = pd.Series(returns).rolling(window, min_periods=1).std()
            # Fill NaN values (backward fill then forward fill)
            vol_series = vol_series.bfill().ffill().fillna(0.01)
            vol = vol_series.values * np.sqrt(252)
            # Ensure no zero volatility
            vol = np.maximum(vol, 0.01)
        
        # Volatility scaling factor
        vol_scale = self.target_vol / (vol + 1e-10)
        vol_scale = np.clip(vol_scale, 0.1, 3.0)  # Limit scaling
        
        # Scaled signal
        scaled_signal = base * vol_scale
        
        return np.clip(scaled_signal, -1, 1)


class DynamicBetaSignal(BaseSignal):
    """
    Market-timing signal based on dynamic beta estimation.
    
    Increases market exposure when beta is favorable.
    """
    
    def __init__(self, 
                 kf_beta: KalmanFilter,
                 market_returns: np.ndarray,
                 beta_threshold: float = 1.0):
        """
        Initialize dynamic beta signal.
        
        Parameters
        ----------
        kf_beta : KalmanFilter
            Kalman filter for dynamic beta
        market_returns : np.ndarray
            Market return series
        beta_threshold : float
            Beta threshold for position sizing
        """
        self.kf_beta = kf_beta
        self.market_returns = market_returns
        self.beta_threshold = beta_threshold
    
    def generate(self, returns: np.ndarray) -> np.ndarray:
        """Generate dynamic beta signal."""
        # Get dynamic beta estimates
        if self.kf_beta.filtered_states is None:
            raise ValueError("Kalman filter must be fitted first")
        
        # Extract beta (assuming it's the first or second state)
        if self.kf_beta.filtered_states.ndim == 1:
            beta = self.kf_beta.filtered_states
        else:
            # Assume beta is second state (after intercept)
            beta = self.kf_beta.filtered_states[:, 1] if self.kf_beta.filtered_states.shape[1] > 1 else self.kf_beta.filtered_states[:, 0]
        
        # Market timing: follow market when beta is high
        market_signal = np.sign(self.market_returns)
        
        # Scale by beta relative to threshold
        beta_scale = beta / self.beta_threshold
        beta_scale = np.clip(beta_scale, 0, 2)
        
        signal = market_signal * beta_scale
        
        return np.clip(signal, -1, 1)


class CompositeSignal(BaseSignal):
    """
    Composite signal combining multiple sub-signals.
    """
    
    def __init__(self, 
                 signals: list,
                 weights: Optional[np.ndarray] = None):
        """
        Initialize composite signal.
        
        Parameters
        ----------
        signals : list
            List of BaseSignal objects
        weights : np.ndarray, optional
            Signal weights (default: equal weight)
        """
        self.signals = signals
        
        if weights is None:
            weights = np.ones(len(signals)) / len(signals)
        self.weights = weights / weights.sum()
    
    def generate(self, returns: np.ndarray) -> np.ndarray:
        """Generate composite signal."""
        combined = np.zeros(len(returns))
        
        for signal, weight in zip(self.signals, self.weights):
            combined += weight * signal.generate(returns)
        
        return np.clip(combined, -1, 1)


class ThresholdSignal(BaseSignal):
    """
    Apply threshold to continuous signal to create discrete positions.
    """
    
    def __init__(self, 
                 base_signal: BaseSignal,
                 long_threshold: float = 0.3,
                 short_threshold: float = -0.3):
        """
        Initialize threshold signal.
        
        Parameters
        ----------
        base_signal : BaseSignal
            Base signal generator
        long_threshold : float
            Threshold for long position
        short_threshold : float
            Threshold for short position
        """
        self.base_signal = base_signal
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
    
    def generate(self, returns: np.ndarray) -> np.ndarray:
        """Generate threshold signal."""
        base = self.base_signal.generate(returns)
        
        signal = np.zeros_like(base)
        signal[base > self.long_threshold] = 1.0
        signal[base < self.short_threshold] = -1.0
        
        return signal


def create_regime_aware_strategy(returns: np.ndarray,
                                 kf: KalmanFilter,
                                 hmm: GaussianHMM,
                                 vol_target: float = 0.15) -> np.ndarray:
    """
    Convenience function to create complete regime-aware strategy.
    
    PRODUCTION HARDENING:
    - Validates all inputs
    - Ensures signal quality
    - Explicit error handling
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    kf : KalmanFilter
        Fitted Kalman filter
    hmm : GaussianHMM
        Fitted HMM
    vol_target : float
        Target volatility
        
    Returns
    -------
    np.ndarray
        Trading signals
        
    Raises
    ------
    ValueError
        If inputs are invalid
    RuntimeError
        If signal generation fails
    """
    # Input validation
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns, dtype=np.float64)
    
    if np.any(np.isnan(returns)):
        raise ValueError(f"Returns contain {np.sum(np.isnan(returns))} NaN values")
    
    if np.any(np.isinf(returns)):
        raise ValueError(f"Returns contain {np.sum(np.isinf(returns))} Inf values")
    
    if len(returns) < 50:
        raise ValueError(f"Insufficient data: {len(returns)} samples (need at least 50)")
    
    if not kf.is_fitted:
        raise ValueError("Kalman filter must be fitted before signal generation")
    
    if not hmm.is_fitted:
        raise ValueError("HMM must be fitted before signal generation")
    
    if vol_target <= 0 or vol_target > 1:
        raise ValueError(f"Invalid vol_target: {vol_target} (must be in (0, 1])")
    
    try:
        # Base regime-aware signal
        base_signal = RegimeAwareSignal(kf, hmm)
        
        # Add volatility scaling
        vol_signal = VolatilityScaledSignal(base_signal, target_vol=vol_target)
        
        # Generate signals
        signals = vol_signal.generate(returns)
        
        # Final validation
        if np.any(np.isnan(signals)):
            raise RuntimeError(f"Generated signals contain {np.sum(np.isnan(signals))} NaN values")
        
        if np.any(np.isinf(signals)):
            raise RuntimeError(f"Generated signals contain {np.sum(np.isinf(signals))} Inf values")
        
        if not np.all((signals >= -1.0) & (signals <= 1.0)):
            raise RuntimeError(f"Signals outside [-1, 1] range: [{signals.min():.3f}, {signals.max():.3f}]")
        
        # Check for constant signals (potential issue)
        if np.std(signals) < 1e-6:
            import warnings
            warnings.warn("Generated signals have very low variance - may be constant")
        
        return signals
        
    except Exception as e:
        raise RuntimeError(f"Signal generation failed: {e}") from e


if __name__ == '__main__':
    # Test signals
    from src.data_loader import generate_synthetic_data
    from src.state_space_models import LocalLevelModel
    
    print("Testing signal generation...")
    
    # Generate data
    data = generate_synthetic_data(n_samples=500, seed=42)
    returns = data['returns'].iloc[:, 0].values
    
    # Fit models
    model = LocalLevelModel(observation_variance=1.0, state_variance=0.1)
    kf = KalmanFilter(model)
    kf.filter(returns)
    
    hmm = GaussianHMM(n_regimes=3, random_state=42)
    hmm.fit(returns)
    
    # Generate signals
    signals = create_regime_aware_strategy(returns, kf, hmm)
    
    print(f"\nSignal shape: {signals.shape}")
    print(f"Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"Signal mean: {signals.mean():.3f}")
    print(f"Long positions: {(signals > 0).sum()}")
    print(f"Short positions: {(signals < 0).sum()}")
    print(f"Flat positions: {(signals == 0).sum()}")
