"""
Spread metrics and mean-reversion analytics.

Implements Ornstein-Uhlenbeck process estimation, half-life computation,
and z-score normalization for mean-reverting spreads.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import logging

from utils import setup_logging, compute_zscore


logger = setup_logging()


class SpreadAnalyzer:
    """
    Analyzes mean-reversion properties of spread series.
    
    Implements:
    - Ornstein-Uhlenbeck process estimation
    - Half-life computation
    - Z-score normalization
    - Hurst exponent calculation
    """
    
    def __init__(self):
        """Initialize spread analyzer."""
        self.ou_params = None
    
    def estimate_ou_parameters(self, spread: pd.Series) -> Dict:
        """
        Estimate Ornstein-Uhlenbeck process parameters.
        
        The OU process: dS_t = θ(μ - S_t)dt + σdW_t
        
        Discrete approximation: ΔS_t = θ(μ - S_{t-1})Δt + σ√Δt * ε_t
        
        Rearranging: ΔS_t = α + β*S_{t-1} + ε_t
        Where: θ = -β/Δt, μ = -α/β
        
        Args:
            spread: Spread time series
            
        Returns:
            Dictionary with OU parameters
        """
        logger.info("Estimating Ornstein-Uhlenbeck parameters")
        
        # Remove NaN values
        spread_clean = spread.dropna()
        
        if len(spread_clean) < 30:
            raise ValueError("Insufficient data for OU estimation")
        
        # Compute differences
        delta_spread = spread_clean.diff().dropna()
        lagged_spread = spread_clean.shift(1).dropna()
        
        # Align
        common_idx = delta_spread.index.intersection(lagged_spread.index)
        delta_spread = delta_spread.loc[common_idx]
        lagged_spread = lagged_spread.loc[common_idx]
        
        # OLS regression: ΔS_t = α + β*S_{t-1} + ε_t
        X = add_constant(lagged_spread)
        model = OLS(delta_spread, X).fit()
        
        alpha = model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0]
        beta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
        
        # Time step (assuming daily data)
        dt = 1.0
        
        # OU parameters
        theta = -beta / dt  # Mean reversion speed
        mu = -alpha / beta if beta != 0 else 0  # Long-term mean
        sigma = model.resid.std() / np.sqrt(dt)  # Volatility
        
        # Half-life: time for spread to revert halfway to mean
        # Half-life = ln(2) / θ
        if theta > 0:
            half_life = np.log(2) / theta
        else:
            half_life = np.inf
        
        # R-squared
        r_squared = model.rsquared
        
        self.ou_params = {
            'theta': theta,
            'mu': mu,
            'sigma': sigma,
            'half_life': half_life,
            'alpha': alpha,
            'beta': beta,
            'r_squared': r_squared,
            'std_error_theta': (model.bse.iloc[1] if hasattr(model.bse, 'iloc') else model.bse[1]) / dt,
            'p_value_beta': model.pvalues.iloc[1] if hasattr(model.pvalues, 'iloc') else model.pvalues[1]
        }
        
        logger.info(
            f"OU parameters: θ={theta:.4f}, μ={mu:.4f}, σ={sigma:.4f}, "
            f"half-life={half_life:.2f} days"
        )
        
        return self.ou_params
    
    def compute_half_life_ar1(self, spread: pd.Series) -> float:
        """
        Compute half-life using AR(1) model.
        
        Alternative method: S_t = φ*S_{t-1} + ε_t
        Half-life = -ln(2) / ln(φ)
        
        Args:
            spread: Spread time series
            
        Returns:
            Half-life in periods
        """
        spread_clean = spread.dropna()
        
        if len(spread_clean) < 30:
            raise ValueError("Insufficient data for AR(1) estimation")
        
        # Lag spread
        lagged_spread = spread_clean.shift(1).dropna()
        current_spread = spread_clean.loc[lagged_spread.index]
        
        # OLS regression
        X = add_constant(lagged_spread)
        model = OLS(current_spread, X).fit()
        
        phi = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
        
        # Half-life
        if 0 < phi < 1:
            half_life = -np.log(2) / np.log(phi)
        else:
            half_life = np.inf
        
        logger.info(f"AR(1) half-life: {half_life:.2f} days (φ={phi:.4f})")
        
        return half_life
    
    def compute_rolling_half_life(self,
                                  spread: pd.Series,
                                  window: int = 60,
                                  method: str = 'ou') -> pd.Series:
        """
        Compute rolling half-life over time.
        
        Args:
            spread: Spread time series
            window: Rolling window size
            method: 'ou' or 'ar1'
            
        Returns:
            Series with rolling half-life
        """
        logger.info(f"Computing rolling half-life (window={window}, method={method})")
        
        half_lives = []
        
        for i in range(window, len(spread) + 1):
            window_spread = spread.iloc[i-window:i]
            
            try:
                if method == 'ou':
                    params = self.estimate_ou_parameters(window_spread)
                    hl = params['half_life']
                elif method == 'ar1':
                    hl = self.compute_half_life_ar1(window_spread)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Cap at reasonable value
                hl = min(hl, 1000)
                
            except:
                hl = np.nan
            
            half_lives.append(hl)
        
        # Create series with proper index
        half_life_series = pd.Series(
            half_lives,
            index=spread.index[window:],
            name='half_life'
        )
        
        return half_life_series
    
    def compute_hurst_exponent(self, spread: pd.Series, max_lag: int = 20) -> float:
        """
        Compute Hurst exponent to assess mean-reversion strength.
        
        H < 0.5: Mean-reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        
        Args:
            spread: Spread time series
            max_lag: Maximum lag for R/S analysis
            
        Returns:
            Hurst exponent
        """
        spread_clean = spread.dropna().values
        
        if len(spread_clean) < max_lag * 2:
            raise ValueError("Insufficient data for Hurst exponent")
        
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            # Divide series into chunks
            n_chunks = len(spread_clean) // lag
            
            if n_chunks == 0:
                continue
            
            # Compute R/S for each chunk
            rs_values = []
            
            for i in range(n_chunks):
                chunk = spread_clean[i*lag:(i+1)*lag]
                
                # Mean-adjusted series
                mean_adj = chunk - np.mean(chunk)
                
                # Cumulative sum
                cumsum = np.cumsum(mean_adj)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(chunk, ddof=1)
                
                if S > 0:
                    rs_values.append(R / S)
            
            if len(rs_values) > 0:
                tau.append(np.mean(rs_values))
        
        # Fit log(R/S) = H * log(lag) + const
        log_lags = np.log(list(lags[:len(tau)]))
        log_tau = np.log(tau)
        
        # Linear regression
        coeffs = np.polyfit(log_lags, log_tau, 1)
        hurst = coeffs[0]
        
        logger.info(f"Hurst exponent: {hurst:.4f}")
        
        return hurst
    
    def compute_zscore_series(self,
                            spread: pd.Series,
                            window: int = 60,
                            min_periods: Optional[int] = None) -> pd.Series:
        """
        Compute rolling z-score of spread.
        
        Args:
            spread: Spread time series
            window: Rolling window for mean and std
            min_periods: Minimum observations required
            
        Returns:
            Z-score series
        """
        zscore = compute_zscore(spread, window, min_periods)
        return zscore
    
    def compute_spread_statistics(self, spread: pd.Series) -> Dict:
        """
        Compute comprehensive spread statistics.
        
        Args:
            spread: Spread time series
            
        Returns:
            Dictionary with spread statistics
        """
        spread_clean = spread.dropna()
        
        stats_dict = {
            'mean': spread_clean.mean(),
            'std': spread_clean.std(),
            'min': spread_clean.min(),
            'max': spread_clean.max(),
            'median': spread_clean.median(),
            'skewness': spread_clean.skew(),
            'kurtosis': spread_clean.kurtosis(),
            'adf_statistic': None,
            'n_observations': len(spread_clean)
        }
        
        # Add ADF test
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(spread_clean, maxlag=20, regression='c')
            stats_dict['adf_statistic'] = adf_result[0]
            stats_dict['adf_pvalue'] = adf_result[1]
        except:
            pass
        
        # Add OU parameters if available
        if self.ou_params is not None:
            stats_dict.update({
                'theta': self.ou_params['theta'],
                'mu': self.ou_params['mu'],
                'sigma': self.ou_params['sigma'],
                'half_life': self.ou_params['half_life']
            })
        
        return stats_dict
    
    def detect_regime_changes(self,
                            spread: pd.Series,
                            window: int = 60,
                            threshold: float = 2.0) -> pd.Series:
        """
        Detect regime changes in spread behavior.
        
        Uses rolling volatility to identify regime shifts.
        
        Args:
            spread: Spread time series
            window: Rolling window for volatility
            threshold: Threshold for regime change (in std devs)
            
        Returns:
            Series indicating regime changes
        """
        # Compute rolling volatility
        rolling_vol = spread.rolling(window=window).std()
        
        # Compute z-score of volatility
        vol_mean = rolling_vol.rolling(window=window*2).mean()
        vol_std = rolling_vol.rolling(window=window*2).std()
        vol_zscore = (rolling_vol - vol_mean) / (vol_std + 1e-8)
        
        # Flag regime changes
        regime_change = (vol_zscore.abs() > threshold).astype(int)
        
        return regime_change
    
    def compute_crossing_frequency(self,
                                  spread: pd.Series,
                                  threshold: float = 0.0) -> Dict:
        """
        Compute frequency of spread crossing a threshold (e.g., mean).
        
        Args:
            spread: Spread time series
            threshold: Threshold value
            
        Returns:
            Dictionary with crossing statistics
        """
        spread_clean = spread.dropna()
        
        # Detect crossings
        above = (spread_clean > threshold).astype(int)
        crossings = above.diff().abs()
        
        n_crossings = crossings.sum()
        n_days = len(spread_clean)
        
        # Average time between crossings
        if n_crossings > 0:
            avg_time_between = n_days / n_crossings
        else:
            avg_time_between = np.inf
        
        crossing_stats = {
            'n_crossings': n_crossings,
            'crossing_frequency': n_crossings / n_days,
            'avg_time_between_crossings': avg_time_between
        }
        
        return crossing_stats
    
    def compute_bollinger_bands(self,
                               spread: pd.Series,
                               window: int = 60,
                               n_std: float = 2.0) -> pd.DataFrame:
        """
        Compute Bollinger Bands for spread.
        
        Args:
            spread: Spread time series
            window: Rolling window
            n_std: Number of standard deviations
            
        Returns:
            DataFrame with upper, middle, and lower bands
        """
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        bands = pd.DataFrame({
            'spread': spread,
            'middle': rolling_mean,
            'upper': rolling_mean + n_std * rolling_std,
            'lower': rolling_mean - n_std * rolling_std
        })
        
        return bands
