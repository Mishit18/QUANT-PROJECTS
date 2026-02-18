"""
Regime-Aware Trading Enforcement
=================================

Converts regime diagnostics into HARD trading constraints.

This module implements systematic go/no-go logic based on statistical tests.
Trades are SUPPRESSED when cointegration assumptions fail.

This is NOT optimization - this is risk management.

Author: Senior Quant Researcher
Purpose: Capital preservation through regime awareness
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import logging

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Systematic regime filter for statistical arbitrage.
    
    A regime is TRADEABLE only if ALL conditions are met:
    1. Cointegration test passes (Johansen)
    2. Spread is stationary (ADF)
    3. Half-life is economically meaningful (5-60 days)
    4. Spread variance is stable (not exploding)
    
    This is DEFENSIVE - we trade ONLY when assumptions hold.
    """
    
    def __init__(self, 
                 lookback_window=252,
                 min_half_life=5,
                 max_half_life=60,
                 variance_explosion_threshold=3.0,
                 johansen_significance=0.05,
                 adf_significance=0.05):
        """
        Initialize regime filter.
        
        Parameters:
        -----------
        lookback_window : int
            Rolling window for regime tests (days)
        min_half_life : float
            Minimum acceptable half-life (days)
        max_half_life : float
            Maximum acceptable half-life (days)
        variance_explosion_threshold : float
            Max variance relative to historical mean
        johansen_significance : float
            Significance level for Johansen test
        adf_significance : float
            Significance level for ADF test
        """
        self.lookback_window = lookback_window
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.variance_explosion_threshold = variance_explosion_threshold
        self.johansen_significance = johansen_significance
        self.adf_significance = adf_significance
        
        # Track regime state
        self.regime_history = []
        self.suppressed_trades = 0
        self.allowed_trades = 0
    
    def test_cointegration(self, price1, price2):
        """
        Test if pair is currently cointegrated using Johansen test.
        
        Returns:
        --------
        bool : True if cointegrated, False otherwise
        """
        try:
            data = pd.DataFrame({'p1': price1, 'p2': price2})
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            trace_stat = result.lr1[0]  # Test for r=0
            critical_value = result.cvt[0, 1]  # 5% critical value
            
            is_cointegrated = trace_stat > critical_value
            
            return is_cointegrated, trace_stat, critical_value
        except:
            return False, np.nan, np.nan
    
    def test_stationarity(self, spread):
        """
        Test if spread is stationary using ADF test.
        
        Returns:
        --------
        bool : True if stationary, False otherwise
        """
        try:
            result = adfuller(spread, maxlag=20, regression='c')
            adf_stat = result[0]
            p_value = result[1]
            critical_value = result[4]['5%']
            
            is_stationary = p_value < self.adf_significance
            
            return is_stationary, adf_stat, p_value, critical_value
        except:
            return False, np.nan, np.nan, np.nan
    
    def compute_half_life(self, spread):
        """
        Compute mean-reversion half-life.
        
        Returns:
        --------
        float : Half-life in days
        """
        try:
            spread_clean = spread.dropna()
            delta_spread = spread_clean.diff().dropna()
            lagged_spread = spread_clean.shift(1).dropna()
            
            common_idx = delta_spread.index.intersection(lagged_spread.index)
            delta_spread = delta_spread.loc[common_idx]
            lagged_spread = lagged_spread.loc[common_idx]
            
            X = add_constant(lagged_spread)
            model = OLS(delta_spread, X).fit()
            
            beta = model.params.iloc[1] if hasattr(model.params, 'iloc') else model.params[1]
            theta = -beta
            
            if theta > 0:
                half_life = np.log(2) / theta
            else:
                half_life = np.inf
            
            return half_life
        except:
            return np.inf
    
    def test_variance_stability(self, spread, historical_variance):
        """
        Test if spread variance is stable (not exploding).
        
        Returns:
        --------
        bool : True if stable, False if exploding
        """
        try:
            current_variance = spread.var()
            
            if historical_variance == 0 or np.isnan(historical_variance):
                return True
            
            variance_ratio = current_variance / historical_variance
            is_stable = variance_ratio < self.variance_explosion_threshold
            
            return is_stable, variance_ratio
        except:
            return False, np.nan
    
    def is_regime_tradeable(self, price1, price2, spread, date=None):
        """
        Determine if current regime allows trading.
        
        This is the CORE DECISION FUNCTION.
        
        Returns:
        --------
        dict with:
            - tradeable: bool
            - reasons: list of failure reasons
            - diagnostics: dict of test results
        """
        reasons = []
        diagnostics = {}
        
        # Use rolling window
        if len(price1) < self.lookback_window:
            reasons.append("Insufficient data for regime tests")
            return {
                'tradeable': False,
                'reasons': reasons,
                'diagnostics': diagnostics,
                'date': date
            }
        
        window_price1 = price1.iloc[-self.lookback_window:]
        window_price2 = price2.iloc[-self.lookback_window:]
        window_spread = spread.iloc[-self.lookback_window:]
        
        # Test 1: Cointegration
        is_coint, trace_stat, coint_crit = self.test_cointegration(window_price1, window_price2)
        diagnostics['cointegration'] = {
            'pass': is_coint,
            'trace_stat': trace_stat,
            'critical_value': coint_crit
        }
        if not is_coint:
            reasons.append(f"Cointegration test fails (trace={trace_stat:.2f} < crit={coint_crit:.2f})")
        
        # Test 2: Stationarity
        is_stat, adf_stat, adf_p, adf_crit = self.test_stationarity(window_spread)
        diagnostics['stationarity'] = {
            'pass': is_stat,
            'adf_stat': adf_stat,
            'p_value': adf_p,
            'critical_value': adf_crit
        }
        if not is_stat:
            reasons.append(f"Spread non-stationary (ADF p-value={adf_p:.4f})")
        
        # Test 3: Half-life
        half_life = self.compute_half_life(window_spread)
        is_hl_valid = self.min_half_life <= half_life <= self.max_half_life
        diagnostics['half_life'] = {
            'pass': is_hl_valid,
            'value': half_life,
            'min': self.min_half_life,
            'max': self.max_half_life
        }
        if not is_hl_valid:
            reasons.append(f"Half-life out of bounds ({half_life:.1f} days)")
        
        # Test 4: Variance stability
        historical_var = spread.iloc[:-60].var() if len(spread) > 60 else spread.var()
        is_var_stable, var_ratio = self.test_variance_stability(window_spread, historical_var)
        diagnostics['variance'] = {
            'pass': is_var_stable,
            'ratio': var_ratio,
            'threshold': self.variance_explosion_threshold
        }
        if not is_var_stable:
            reasons.append(f"Variance explosion (ratio={var_ratio:.2f})")
        
        # DECISION: ALL tests must pass
        tradeable = is_coint and is_stat and is_hl_valid and is_var_stable
        
        # Track statistics
        if tradeable:
            self.allowed_trades += 1
        else:
            self.suppressed_trades += 1
        
        # Log decision
        if not tradeable:
            logger.info(f"Trade SUPPRESSED at {date}: {', '.join(reasons)}")
        
        return {
            'tradeable': tradeable,
            'reasons': reasons,
            'diagnostics': diagnostics,
            'date': date
        }
    
    def get_statistics(self):
        """Get regime filter statistics."""
        total = self.allowed_trades + self.suppressed_trades
        if total == 0:
            return {
                'total_decisions': 0,
                'trades_allowed': 0,
                'trades_suppressed': 0,
                'suppression_rate': 0.0
            }
        
        return {
            'total_decisions': total,
            'trades_allowed': self.allowed_trades,
            'trades_suppressed': self.suppressed_trades,
            'suppression_rate': self.suppressed_trades / total
        }


def apply_regime_filter_to_signals(signals, price1, price2, spread, regime_filter):
    """
    Apply regime filter to trading signals.
    
    This modifies signals DataFrame in-place to suppress trades during
    unstable regimes.
    
    Parameters:
    -----------
    signals : pd.DataFrame
        Trading signals with 'position' column
    price1, price2 : pd.Series
        Price series for the pair
    spread : pd.Series
        Spread time series
    regime_filter : RegimeFilter
        Configured regime filter
    
    Returns:
    --------
    pd.DataFrame : Modified signals with regime filter applied
    dict : Regime filter statistics
    """
    logger.info("Applying regime filter to trading signals...")
    
    # Create copy to avoid modifying original
    filtered_signals = signals.copy()
    filtered_signals['regime_tradeable'] = False
    filtered_signals['regime_reasons'] = ''
    
    # Track regime state for each signal
    regime_decisions = []
    
    for idx in filtered_signals.index:
        # Get data up to current point (no lookahead)
        current_price1 = price1.loc[:idx]
        current_price2 = price2.loc[:idx]
        current_spread = spread.loc[:idx]
        
        # Test regime
        decision = regime_filter.is_regime_tradeable(
            current_price1, 
            current_price2, 
            current_spread,
            date=idx
        )
        
        regime_decisions.append(decision)
        filtered_signals.loc[idx, 'regime_tradeable'] = decision['tradeable']
        filtered_signals.loc[idx, 'regime_reasons'] = '; '.join(decision['reasons'])
        
        # Suppress position if regime is not tradeable
        if not decision['tradeable']:
            filtered_signals.loc[idx, 'position'] = 0
    
    # Get statistics
    stats = regime_filter.get_statistics()
    
    logger.info(f"Regime filter applied:")
    logger.info(f"  - Total decisions: {stats['total_decisions']}")
    logger.info(f"  - Trades allowed: {stats['trades_allowed']}")
    logger.info(f"  - Trades suppressed: {stats['trades_suppressed']}")
    logger.info(f"  - Suppression rate: {stats['suppression_rate']:.1%}")
    
    return filtered_signals, stats, regime_decisions
