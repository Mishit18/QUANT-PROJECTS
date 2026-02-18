"""
Spread construction and OU parameter estimation.
Enforces quality thresholds - weak mean reversion = no trade.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.optimize import minimize


class SpreadModel:
    """
    OU process with strict quality enforcement.
    
    Edge hypothesis: Only trade spreads with strong, fast mean reversion.
    """
    
    def __init__(self, min_r_squared: float = 0.15, 
                 min_half_life: float = 3.0,
                 max_half_life: float = 40.0):
        """
        Args:
            min_r_squared: Minimum OU fit quality (reject weak mean reversion)
            min_half_life: Minimum days (too fast = noise)
            max_half_life: Maximum days (too slow = no edge)
        """
        # PRODUCTION FIX: Defensive type casting
        try:
            self.min_r_squared = float(min_r_squared)
            self.min_half_life = float(min_half_life)
            self.max_half_life = float(max_half_life)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"SpreadModel parameters must be numeric. Error: {e}"
            )
        
        # Validation
        if not (0 < self.min_r_squared < 1):
            raise ValueError(f"min_r_squared must be in (0, 1), got {self.min_r_squared}")
        if self.min_half_life <= 0 or self.max_half_life <= 0:
            raise ValueError("Half-life bounds must be positive")
        if self.min_half_life >= self.max_half_life:
            raise ValueError("min_half_life must be < max_half_life")
        
        self.theta = None
        self.mu = None
        self.sigma = None
        self.half_life = None
        self.r_squared = None
        self.is_valid = False
        self.rejection_reason = None
    
    def fit(self, spread: pd.Series) -> Dict:
        """
        Fit OU model with quality checks.
        
        Returns:
            dict with parameters and validation status
        """
        spread_clean = spread.dropna()
        
        if len(spread_clean) < 100:
            self.is_valid = False
            self.rejection_reason = "Insufficient data"
            return self._get_params()
        
        # OLS estimation
        X_t = spread_clean.values[:-1]
        X_t1 = spread_clean.values[1:]
        dX = X_t1 - X_t
        
        # Regression: dX = a + b * X_t
        X_matrix = np.column_stack([np.ones(len(X_t)), X_t])
        
        try:
            beta = np.linalg.lstsq(X_matrix, dX, rcond=None)[0]
        except:
            self.is_valid = False
            self.rejection_reason = "OLS failed"
            return self._get_params()
        
        a, b = beta[0], beta[1]
        
        # Convert to OU parameters
        if b >= 0:
            self.is_valid = False
            self.rejection_reason = "No mean reversion (b >= 0)"
            return self._get_params()
        
        self.theta = -b
        self.mu = a / self.theta if self.theta > 0 else spread_clean.mean()
        
        # Estimate sigma from residuals
        residuals = dX - (a + b * X_t)
        self.sigma = np.std(residuals)
        
        # Half-life
        self.half_life = np.log(2) / self.theta
        
        # R-squared (critical quality metric)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((dX - np.mean(dX))**2)
        self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Validation
        self._validate()
        
        return self._get_params()
    
    def _validate(self):
        """Enforce quality thresholds with controlled flexibility for strong fits."""
        
        # Check R²
        if self.r_squared < self.min_r_squared:
            self.is_valid = False
            self.rejection_reason = f"Weak OU fit (R²={self.r_squared:.3f} < {self.min_r_squared})"
            return
        
        # Check half-life bounds with flexibility for strong fits
        # If R² is very strong (>0.25), allow faster mean reversion
        strong_r2_threshold = 0.25
        is_strong_fit = self.r_squared >= strong_r2_threshold
        
        # Check half-life bounds (with flexibility for strong fits)
        effective_min_hl = self.min_half_life
        if is_strong_fit:
            # Allow half-life down to 1.5 days if R² > 0.25
            effective_min_hl = max(1.5, self.min_half_life * 0.5)
        
        if self.half_life < effective_min_hl:
            if is_strong_fit:
                # Allow but flag
                self.is_valid = True
                self.rejection_reason = f"Fast mean reversion (HL={self.half_life:.1f}d) but strong R²={self.r_squared:.3f}"
                return
            else:
                self.is_valid = False
                self.rejection_reason = f"Half-life too short ({self.half_life:.1f} < {effective_min_hl})"
                return
        
        if self.half_life > self.max_half_life:
            self.is_valid = False
            self.rejection_reason = f"Half-life too long ({self.half_life:.1f} > {self.max_half_life})"
            return
        
        # Check theta (mean reversion speed)
        if self.theta < 0.02:  # Very slow mean reversion
            self.is_valid = False
            self.rejection_reason = f"Mean reversion too slow (theta={self.theta:.4f})"
            return
        
        self.is_valid = True
        self.rejection_reason = None
    
    def _get_params(self) -> Dict:
        """Return parameter dict."""
        return {
            'theta': self.theta,
            'mu': self.mu,
            'sigma': self.sigma,
            'half_life': self.half_life,
            'r_squared': self.r_squared,
            'is_valid': self.is_valid,
            'rejection_reason': self.rejection_reason
        }
    
    def get_equilibrium_std(self) -> float:
        """
        Long-run standard deviation of spread.
        Used for z-score normalization.
        """
        if not self.is_valid or self.theta is None:
            return np.nan
        
        return self.sigma / np.sqrt(2 * self.theta)
    
    def expected_reversion_time(self, current_z: float, target_z: float = 0.0) -> float:
        """
        Expected time for z-score to revert from current to target.
        
        Used for time-based exits.
        """
        if not self.is_valid or abs(current_z) <= abs(target_z):
            return 0.0
        
        # Time for expected value to reach target
        ratio = abs(target_z) / abs(current_z)
        if ratio >= 1:
            return 0.0
        
        return -np.log(ratio) / self.theta
    
    def position_size_multiplier(self) -> float:
        """
        Scale position by OU quality.
        
        Better fit = more confidence = larger position.
        Penalize very fast mean reversion (likely noise).
        """
        if not self.is_valid:
            return 0.0
        
        # Scale by R² (0.15 -> 0.5x, 0.30 -> 1.0x, 0.45+ -> 1.5x)
        r2_factor = np.clip((self.r_squared - 0.15) / 0.15, 0.5, 1.5)
        
        # Scale by half-life (penalize very fast reversion)
        # HL < 2 days: likely noise, reduce size
        # HL 5-20 days: sweet spot, full size
        # HL > 30 days: slow, reduce size
        if self.half_life < 2.0:
            hl_factor = 0.3  # Very conservative on fast reversion
        elif self.half_life < 5.0:
            hl_factor = 0.6  # Still cautious
        elif self.half_life <= 20.0:
            hl_factor = 1.0  # Sweet spot
        elif self.half_life <= 30.0:
            hl_factor = 0.8  # Getting slow
        else:
            hl_factor = 0.5  # Very slow
        
        return r2_factor * hl_factor
