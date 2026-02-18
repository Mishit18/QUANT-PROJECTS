"""
Kalman Confidence-Aware Execution Control
==========================================

Makes Kalman Filter outputs ACTIONABLE by using state uncertainty
as a confidence measure.

High beta uncertainty → Reduce exposure or suppress trades
This is NOT smoothing - this is using uncertainty as information.

Author: Senior Quant Researcher
Purpose: Execution control based on model confidence
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class KalmanConfidenceGate:
    """
    Uses Kalman Filter state uncertainty to gate trade execution.
    
    LOGIC:
    - High P_t (state covariance) → Low confidence → Reduce/suppress trades
    - Rapid beta changes → Regime shift → Suppress trades
    - Stable beta → High confidence → Allow trades
    
    This is DEFENSIVE - we trade only when model is confident.
    """
    
    def __init__(self,
                 uncertainty_threshold=0.1,
                 beta_volatility_threshold=0.05,
                 lookback_window=60):
        """
        Initialize Kalman confidence gate.
        
        Parameters:
        -----------
        uncertainty_threshold : float
            Maximum acceptable state covariance (P_t)
        beta_volatility_threshold : float
            Maximum acceptable rolling std of beta
        lookback_window : int
            Window for computing beta volatility
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.beta_volatility_threshold = beta_volatility_threshold
        self.lookback_window = lookback_window
        
        # Track statistics
        self.high_confidence_count = 0
        self.low_confidence_count = 0
    
    def compute_beta_volatility(self, beta_series):
        """
        Compute rolling volatility of hedge ratio.
        
        High volatility indicates regime instability.
        """
        if len(beta_series) < self.lookback_window:
            return np.nan
        
        rolling_std = beta_series.rolling(self.lookback_window).std()
        return rolling_std.iloc[-1]
    
    def is_confident(self, state_covariance, beta_series, date=None):
        """
        Determine if Kalman Filter is confident enough to trade.
        
        Returns:
        --------
        dict with:
            - confident: bool
            - reasons: list of failure reasons
            - diagnostics: dict of metrics
        """
        reasons = []
        diagnostics = {}
        
        # Test 1: State uncertainty
        diagnostics['state_covariance'] = {
            'value': state_covariance,
            'threshold': self.uncertainty_threshold,
            'pass': state_covariance < self.uncertainty_threshold
        }
        
        if state_covariance >= self.uncertainty_threshold:
            reasons.append(f"High state uncertainty (P={state_covariance:.4f})")
        
        # Test 2: Beta volatility
        beta_vol = self.compute_beta_volatility(beta_series)
        diagnostics['beta_volatility'] = {
            'value': beta_vol,
            'threshold': self.beta_volatility_threshold,
            'pass': beta_vol < self.beta_volatility_threshold if not np.isnan(beta_vol) else False
        }
        
        if np.isnan(beta_vol):
            reasons.append("Insufficient data for beta volatility")
        elif beta_vol >= self.beta_volatility_threshold:
            reasons.append(f"High beta volatility (std={beta_vol:.4f})")
        
        # DECISION: Both tests must pass
        confident = (
            state_covariance < self.uncertainty_threshold and
            not np.isnan(beta_vol) and
            beta_vol < self.beta_volatility_threshold
        )
        
        # Track statistics
        if confident:
            self.high_confidence_count += 1
        else:
            self.low_confidence_count += 1
        
        # Log decision
        if not confident:
            logger.info(f"Low Kalman confidence at {date}: {', '.join(reasons)}")
        
        return {
            'confident': confident,
            'reasons': reasons,
            'diagnostics': diagnostics,
            'date': date
        }
    
    def get_statistics(self):
        """Get confidence gate statistics."""
        total = self.high_confidence_count + self.low_confidence_count
        if total == 0:
            return {
                'total_decisions': 0,
                'high_confidence': 0,
                'low_confidence': 0,
                'low_confidence_rate': 0.0
            }
        
        return {
            'total_decisions': total,
            'high_confidence': self.high_confidence_count,
            'low_confidence': self.low_confidence_count,
            'low_confidence_rate': self.low_confidence_count / total
        }


def apply_kalman_confidence_filter(signals, kalman_results, confidence_gate):
    """
    Apply Kalman confidence filter to trading signals.
    
    This suppresses trades when Kalman Filter uncertainty is high.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Trading signals with 'position' column
    kalman_results : pd.DataFrame
        Kalman Filter results with 'beta' and 'P' columns
    confidence_gate : KalmanConfidenceGate
        Configured confidence gate
    
    Returns:
    --------
    pd.DataFrame : Modified signals with confidence filter applied
    dict : Confidence gate statistics
    """
    logger.info("Applying Kalman confidence filter to trading signals...")
    
    # Create copy
    filtered_signals = signals.copy()
    filtered_signals['kalman_confident'] = False
    filtered_signals['kalman_reasons'] = ''
    
    # Align indices
    common_idx = signals.index.intersection(kalman_results.index)
    
    confidence_decisions = []
    
    for idx in common_idx:
        # Get Kalman state at this point
        state_cov = kalman_results.loc[idx, 'P']
        
        # Get beta history up to this point (no lookahead)
        beta_history = kalman_results.loc[:idx, 'beta']
        
        # Test confidence
        decision = confidence_gate.is_confident(
            state_cov,
            beta_history,
            date=idx
        )
        
        confidence_decisions.append(decision)
        filtered_signals.loc[idx, 'kalman_confident'] = decision['confident']
        filtered_signals.loc[idx, 'kalman_reasons'] = '; '.join(decision['reasons'])
        
        # Suppress position if not confident
        if not decision['confident']:
            filtered_signals.loc[idx, 'position'] = 0
    
    # Get statistics
    stats = confidence_gate.get_statistics()
    
    logger.info(f"Kalman confidence filter applied:")
    logger.info(f"  - Total decisions: {stats['total_decisions']}")
    logger.info(f"  - High confidence: {stats['high_confidence']}")
    logger.info(f"  - Low confidence: {stats['low_confidence']}")
    logger.info(f"  - Low confidence rate: {stats['low_confidence_rate']:.1%}")
    
    return filtered_signals, stats, confidence_decisions


def compute_confidence_adjusted_position_size(base_position, state_covariance, max_covariance=1.0):
    """
    Scale position size based on Kalman confidence.
    
    This is an ALTERNATIVE to binary suppression - gradually reduce exposure
    as uncertainty increases.
    
    Parameters:
    -----------
    base_position : float
        Desired position size
    state_covariance : float
        Current Kalman state covariance
    max_covariance : float
        Maximum covariance for full position
    
    Returns:
    --------
    float : Adjusted position size
    """
    if state_covariance >= max_covariance:
        return 0.0
    
    # Linear scaling: full position at P=0, zero position at P=max_covariance
    confidence_factor = 1.0 - (state_covariance / max_covariance)
    
    return base_position * confidence_factor
