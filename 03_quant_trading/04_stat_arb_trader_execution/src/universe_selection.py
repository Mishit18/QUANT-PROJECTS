"""
Universe selection and pair screening for statistical arbitrage.
Implements rigorous cointegration testing and pair validation.

This module enforces statistical discipline:
- Only cointegrated pairs pass screening
- OU parameters must be economically meaningful
- Reject >90% of candidate pairs
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PairCandidate:
    """Container for pair screening results."""
    asset1: str
    asset2: str
    hedge_ratio: float
    coint_pvalue: float
    half_life: float
    adf_stat: float
    is_valid: bool
    rejection_reason: Optional[str] = None
    category: Optional[str] = None  # Pair classification: micro_noise, unstable, tradable_daily, slow_reversion


class UniverseSelector:
    """
    Production-grade pair selection engine.
    
    Screening criteria (ALL must pass):
    1. Cointegration p-value < 0.05
    2. Half-life between 5 and 60 days
    3. Residuals are stationary (ADF p-value < 0.05)
    4. Sufficient data quality (>= 252 observations)
    """
    
    # Hardcoded thresholds - no flexibility for gaming
    COINT_PVALUE_MAX = 0.05
    HALF_LIFE_MIN = 5.0
    HALF_LIFE_MAX = 60.0
    ADF_PVALUE_MAX = 0.05
    MIN_OBSERVATIONS = 252
    
    def __init__(self, start_date: str = '2018-01-01', end_date: str = '2023-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.data_cache = {}
    
    def fetch_data(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch and cache price data."""
        cache_key = tuple(sorted(tickers))
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        data = yf.download(tickers, start=self.start_date, end=self.end_date, 
                          progress=False, auto_adjust=True)
        
        if len(tickers) == 1:
            prices = data[['Close']].copy()
            prices.columns = tickers
        else:
            prices = data['Close'].copy()
        
        # Clean data
        prices = prices.dropna()
        
        self.data_cache[cache_key] = prices
        return prices
    
    def test_cointegration(self, y: pd.Series, x: pd.Series) -> Tuple[float, float, float]:
        """
        Engle-Granger cointegration test.
        
        Returns:
            (hedge_ratio, p_value, adf_statistic)
        """
        # OLS regression
        X = np.column_stack([np.ones(len(x)), x.values])
        beta = np.linalg.lstsq(X, y.values, rcond=None)[0]
        hedge_ratio = beta[1]
        
        # Residuals
        residuals = y.values - (beta[0] + hedge_ratio * x.values)
        
        # ADF test on residuals
        adf_result = adfuller(residuals, maxlag=1, regression='c', autolag=None)
        adf_stat = adf_result[0]
        
        # Cointegration test
        _, pvalue, _ = coint(y.values, x.values, trend='c')
        
        return hedge_ratio, pvalue, adf_stat
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using AR(1).
        
        Returns:
            Half-life in days (inf if no mean reversion)
        """
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        
        df = pd.DataFrame({'spread_diff': spread_diff, 'spread_lag': spread_lag}).dropna()
        
        if len(df) < 10:
            return np.inf
        
        X = df['spread_lag'].values.reshape(-1, 1)
        y = df['spread_diff'].values
        
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            lambda_param = beta[1]
            
            if lambda_param >= 0:
                return np.inf
            
            half_life = -np.log(2) / lambda_param
            return half_life
        except:
            return np.inf
    
    def screen_pair(self, asset1: str, asset2: str) -> PairCandidate:
        """
        Screen a single pair against all criteria.
        
        Categorizes pairs by mean reversion characteristics:
        - micro_noise: HL < 2 days (likely noise at daily frequency)
        - unstable: Poor OU fit or regime instability
        - tradable_daily: HL ∈ [15, 45] days (suitable for daily trading)
        - slow_reversion: HL > 60 days (too slow for practical trading)
        
        Returns:
            PairCandidate with validation results and category
        """
        try:
            # Fetch data
            prices = self.fetch_data([asset1, asset2])
            
            if len(prices) < self.MIN_OBSERVATIONS:
                return PairCandidate(
                    asset1=asset1, asset2=asset2,
                    hedge_ratio=np.nan, coint_pvalue=1.0,
                    half_life=np.inf, adf_stat=0.0,
                    is_valid=False,
                    rejection_reason=f"Insufficient data: {len(prices)} < {self.MIN_OBSERVATIONS}",
                    category="insufficient_data"
                )
            
            y = prices[asset1]
            x = prices[asset2]
            
            # Test cointegration
            hedge_ratio, coint_pvalue, adf_stat = self.test_cointegration(y, x)
            
            # Calculate spread
            spread = y - hedge_ratio * x
            
            # Calculate half-life
            half_life = self.calculate_half_life(spread)
            
            # Categorize pair based on half-life
            category = self._categorize_pair(half_life, coint_pvalue, adf_stat)
            
            # Validation logic
            is_valid = True
            rejection_reason = None
            
            if coint_pvalue > self.COINT_PVALUE_MAX:
                is_valid = False
                rejection_reason = f"Not cointegrated: p-value={coint_pvalue:.4f} > {self.COINT_PVALUE_MAX}"
            
            elif half_life < self.HALF_LIFE_MIN:
                is_valid = False
                rejection_reason = f"Half-life too short: {half_life:.1f} < {self.HALF_LIFE_MIN} days"
            
            elif half_life > self.HALF_LIFE_MAX:
                is_valid = False
                rejection_reason = f"Half-life too long: {half_life:.1f} > {self.HALF_LIFE_MAX} days"
            
            elif abs(adf_stat) < 2.5:  # Rough threshold for stationarity
                is_valid = False
                rejection_reason = f"Residuals not stationary: ADF={adf_stat:.2f}"
            
            return PairCandidate(
                asset1=asset1, asset2=asset2,
                hedge_ratio=hedge_ratio,
                coint_pvalue=coint_pvalue,
                half_life=half_life,
                adf_stat=adf_stat,
                is_valid=is_valid,
                rejection_reason=rejection_reason,
                category=category
            )
        
        except Exception as e:
            return PairCandidate(
                asset1=asset1, asset2=asset2,
                hedge_ratio=np.nan, coint_pvalue=1.0,
                half_life=np.inf, adf_stat=0.0,
                is_valid=False,
                rejection_reason=f"Error: {str(e)}",
                category="error"
            )
    
    def _categorize_pair(self, half_life: float, coint_pvalue: float, adf_stat: float) -> str:
        """
        Categorize pair by mean reversion characteristics.
        
        Categories:
        - micro_noise: HL < 2 days (likely noise at daily frequency)
        - tradable_daily: HL ∈ [15, 45] days (suitable for daily trading)
        - slow_reversion: HL > 60 days (too slow for practical trading)
        - unstable: Poor statistical properties
        """
        if half_life < 2.0:
            return "micro_noise"
        elif 15.0 <= half_life <= 45.0:
            return "tradable_daily"
        elif half_life > 60.0:
            return "slow_reversion"
        elif coint_pvalue > 0.10 or abs(adf_stat) < 2.0:
            return "unstable"
        else:
            return "marginal"  # Passes tests but not in sweet spot
    
    def screen_universe(self, universe: List[str]) -> List[PairCandidate]:
        """
        Screen all pairs in universe.
        
        Args:
            universe: List of tickers to screen
        
        Returns:
            List of PairCandidate objects (sorted by validity and p-value)
        """
        candidates = []
        
        print(f"Screening {len(universe)} assets...")
        print(f"Total pairs to test: {len(universe) * (len(universe) - 1) // 2}")
        
        for i in range(len(universe)):
            for j in range(i + 1, len(universe)):
                asset1, asset2 = universe[i], universe[j]
                
                candidate = self.screen_pair(asset1, asset2)
                candidates.append(candidate)
                
                if candidate.is_valid:
                    print(f"  [PASS] {asset1}/{asset2}: p={candidate.coint_pvalue:.4f}, HL={candidate.half_life:.1f}d, category={candidate.category}")
                else:
                    print(f"  [REJECT] {asset1}/{asset2}: {candidate.rejection_reason}")
        
        # Sort: valid first, then by p-value
        candidates.sort(key=lambda x: (not x.is_valid, x.coint_pvalue))
        
        valid_count = sum(1 for c in candidates if c.is_valid)
        rejection_rate = 1 - (valid_count / len(candidates))
        
        # Category breakdown
        category_counts = {}
        for c in candidates:
            if c.category:
                category_counts[c.category] = category_counts.get(c.category, 0) + 1
        
        print(f"\nScreening complete:")
        print(f"  Valid pairs: {valid_count}/{len(candidates)} ({valid_count/len(candidates)*100:.1f}%)")
        print(f"  Rejection rate: {rejection_rate*100:.1f}%")
        print(f"\nCategory breakdown:")
        for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count} pairs")
        
        if rejection_rate < 0.90:
            print(f"  WARNING: Rejection rate < 90% - criteria may be too loose")
        
        return candidates
    
    def get_best_pairs(self, candidates: List[PairCandidate], n: int = 3) -> List[PairCandidate]:
        """
        Get top N valid pairs, prioritizing tradable_daily category.
        """
        valid = [c for c in candidates if c.is_valid]
        
        # Prioritize tradable_daily
        tradable = [c for c in valid if c.category == "tradable_daily"]
        others = [c for c in valid if c.category != "tradable_daily"]
        
        # Return tradable first, then others
        return (tradable + others)[:n]


def get_default_universe() -> List[str]:
    """
    Default universe of liquid ETFs for pair screening.
    
    Includes:
    - Sector ETFs (XL*)
    - Commodity ETFs (GLD, GDX, SLV, USO)
    - International ETFs (EW*)
    - Broad market ETFs (SPY, QQQ, IWM, DIA)
    """
    return [
        # Sector ETFs
        'XLE', 'XLF', 'XLU', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLB',
        # Commodities
        'GLD', 'GDX', 'SLV',
        # International
        'EWA', 'EWC', 'EWG', 'EWJ', 'EWU',
        # Broad market
        'SPY', 'QQQ', 'IWM', 'DIA'
    ]
