"""
Stationarity testing module.

Implements Augmented Dickey-Fuller (ADF) and KPSS tests for unit root testing.
Critical for validating cointegration assumptions and spread stationarity.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from statsmodels.tsa.stattools import adfuller, kpss
import logging

from utils import setup_logging


logger = setup_logging()


class StationarityTester:
    """
    Performs stationarity tests on time series data.
    
    Implements:
    - Augmented Dickey-Fuller (ADF) test
    - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize stationarity tester.
        
        Args:
            significance_level: Significance level for hypothesis tests
        """
        self.significance_level = significance_level
    
    def adf_test(self,
                series: pd.Series,
                maxlag: Optional[int] = None,
                regression: str = 'c',
                autolag: str = 'AIC') -> Dict:
        """
        Perform Augmented Dickey-Fuller test.
        
        The ADF test tests the null hypothesis that a unit root is present.
        - H0: Series has a unit root (non-stationary)
        - H1: Series is stationary
        
        Args:
            series: Time series to test
            maxlag: Maximum lag to use (None for automatic selection)
            regression: Regression type ('c': constant, 'ct': constant+trend, 'ctt': constant+trend+trend^2, 'n': no constant/trend)
            autolag: Method for lag selection ('AIC', 'BIC', 't-stat', None)
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            raise ValueError("Insufficient data for ADF test")
        
        # Perform ADF test
        result = adfuller(
            series_clean,
            maxlag=maxlag,
            regression=regression,
            autolag=autolag
        )
        
        adf_statistic = result[0]
        p_value = result[1]
        used_lag = result[2]
        n_obs = result[3]
        critical_values = result[4]
        
        # Determine stationarity
        is_stationary = p_value < self.significance_level
        
        test_results = {
            'test': 'ADF',
            'statistic': adf_statistic,
            'p_value': p_value,
            'used_lag': used_lag,
            'n_obs': n_obs,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'regression': regression,
            'interpretation': self._interpret_adf(adf_statistic, p_value, critical_values)
        }
        
        logger.info(
            f"ADF Test: statistic={adf_statistic:.4f}, p-value={p_value:.4f}, "
            f"stationary={is_stationary}"
        )
        
        return test_results
    
    def _interpret_adf(self,
                      statistic: float,
                      p_value: float,
                      critical_values: Dict) -> str:
        """
        Interpret ADF test results.
        
        Args:
            statistic: ADF test statistic
            p_value: P-value
            critical_values: Critical values at different significance levels
            
        Returns:
            Interpretation string
        """
        if p_value < 0.01:
            return "Strong evidence against unit root (stationary at 1% level)"
        elif p_value < 0.05:
            return "Evidence against unit root (stationary at 5% level)"
        elif p_value < 0.10:
            return "Weak evidence against unit root (stationary at 10% level)"
        else:
            return "Insufficient evidence against unit root (likely non-stationary)"
    
    def kpss_test(self,
                 series: pd.Series,
                 regression: str = 'c',
                 nlags: str = 'auto') -> Dict:
        """
        Perform KPSS test for stationarity.
        
        The KPSS test tests the null hypothesis that the series is stationary.
        - H0: Series is stationary
        - H1: Series has a unit root (non-stationary)
        
        Note: KPSS is complementary to ADF. Use both for robust conclusions.
        
        Args:
            series: Time series to test
            regression: Regression type ('c': constant, 'ct': constant+trend)
            nlags: Number of lags ('auto' for automatic selection)
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) < 10:
            raise ValueError("Insufficient data for KPSS test")
        
        # Perform KPSS test
        result = kpss(
            series_clean,
            regression=regression,
            nlags=nlags
        )
        
        kpss_statistic = result[0]
        p_value = result[1]
        used_lag = result[2]
        critical_values = result[3]
        
        # Determine stationarity (note: opposite interpretation from ADF)
        is_stationary = p_value > self.significance_level
        
        test_results = {
            'test': 'KPSS',
            'statistic': kpss_statistic,
            'p_value': p_value,
            'used_lag': used_lag,
            'critical_values': critical_values,
            'is_stationary': is_stationary,
            'regression': regression,
            'interpretation': self._interpret_kpss(kpss_statistic, p_value, critical_values)
        }
        
        logger.info(
            f"KPSS Test: statistic={kpss_statistic:.4f}, p-value={p_value:.4f}, "
            f"stationary={is_stationary}"
        )
        
        return test_results
    
    def _interpret_kpss(self,
                       statistic: float,
                       p_value: float,
                       critical_values: Dict) -> str:
        """
        Interpret KPSS test results.
        
        Args:
            statistic: KPSS test statistic
            p_value: P-value
            critical_values: Critical values at different significance levels
            
        Returns:
            Interpretation string
        """
        if p_value > 0.10:
            return "Strong evidence for stationarity"
        elif p_value > 0.05:
            return "Evidence for stationarity (at 5% level)"
        elif p_value > 0.01:
            return "Weak evidence for stationarity"
        else:
            return "Evidence against stationarity (likely non-stationary)"
    
    def combined_test(self,
                     series: pd.Series,
                     adf_params: Optional[Dict] = None,
                     kpss_params: Optional[Dict] = None) -> Dict:
        """
        Perform both ADF and KPSS tests for robust stationarity assessment.
        
        Interpretation guide:
        - ADF rejects H0 & KPSS fails to reject H0: Stationary
        - ADF fails to reject H0 & KPSS rejects H0: Non-stationary
        - Both reject: Difference-stationary (trend-stationary)
        - Both fail to reject: Inconclusive (need more data or different tests)
        
        Args:
            series: Time series to test
            adf_params: Parameters for ADF test
            kpss_params: Parameters for KPSS test
            
        Returns:
            Dictionary with combined test results
        """
        if adf_params is None:
            adf_params = {}
        if kpss_params is None:
            kpss_params = {}
        
        # Perform both tests
        adf_result = self.adf_test(series, **adf_params)
        kpss_result = self.kpss_test(series, **kpss_params)
        
        # Combined interpretation
        adf_stationary = adf_result['is_stationary']
        kpss_stationary = kpss_result['is_stationary']
        
        if adf_stationary and kpss_stationary:
            conclusion = "STATIONARY: Both tests agree"
            confidence = "High"
        elif not adf_stationary and not kpss_stationary:
            conclusion = "NON-STATIONARY: Both tests agree"
            confidence = "High"
        elif adf_stationary and not kpss_stationary:
            conclusion = "DIFFERENCE-STATIONARY: Tests disagree (possible trend)"
            confidence = "Medium"
        else:
            conclusion = "INCONCLUSIVE: Tests disagree (need more investigation)"
            confidence = "Low"
        
        combined_result = {
            'adf': adf_result,
            'kpss': kpss_result,
            'conclusion': conclusion,
            'confidence': confidence,
            'is_stationary': adf_stationary and kpss_stationary
        }
        
        logger.info(f"Combined Test: {conclusion} (confidence: {confidence})")
        
        return combined_result
    
    def test_multiple_series(self,
                           data: pd.DataFrame,
                           test_type: str = 'combined') -> pd.DataFrame:
        """
        Test stationarity for multiple time series.
        
        Args:
            data: DataFrame with multiple time series
            test_type: Type of test ('adf', 'kpss', 'combined')
            
        Returns:
            DataFrame with test results for each series
        """
        results = []
        
        for col in data.columns:
            logger.info(f"Testing stationarity for {col}")
            
            if test_type == 'adf':
                result = self.adf_test(data[col])
                results.append({
                    'series': col,
                    'test': 'ADF',
                    'statistic': result['statistic'],
                    'p_value': result['p_value'],
                    'is_stationary': result['is_stationary']
                })
            
            elif test_type == 'kpss':
                result = self.kpss_test(data[col])
                results.append({
                    'series': col,
                    'test': 'KPSS',
                    'statistic': result['statistic'],
                    'p_value': result['p_value'],
                    'is_stationary': result['is_stationary']
                })
            
            elif test_type == 'combined':
                result = self.combined_test(data[col])
                results.append({
                    'series': col,
                    'test': 'Combined',
                    'adf_statistic': result['adf']['statistic'],
                    'adf_p_value': result['adf']['p_value'],
                    'kpss_statistic': result['kpss']['statistic'],
                    'kpss_p_value': result['kpss']['p_value'],
                    'is_stationary': result['is_stationary'],
                    'conclusion': result['conclusion'],
                    'confidence': result['confidence']
                })
            
            else:
                raise ValueError(f"Unknown test type: {test_type}")
        
        return pd.DataFrame(results)
    
    def test_differenced_series(self,
                               series: pd.Series,
                               max_diff: int = 2) -> Dict:
        """
        Test stationarity of differenced series to determine integration order.
        
        Args:
            series: Time series to test
            max_diff: Maximum number of differences to test
            
        Returns:
            Dictionary with results for each differencing level
        """
        results = {}
        
        current_series = series.copy()
        
        for d in range(max_diff + 1):
            logger.info(f"Testing d={d} differenced series")
            
            result = self.combined_test(current_series)
            results[f'd={d}'] = result
            
            if result['is_stationary']:
                logger.info(f"Series is I({d})")
                break
            
            # Difference for next iteration
            current_series = current_series.diff().dropna()
        
        return results
