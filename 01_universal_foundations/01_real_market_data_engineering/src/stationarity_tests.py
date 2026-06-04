# src/stationarity_tests.py
"""
Statistical tests for stationarity and time series properties.
Implements: ADF, KPSS, Phillips-Perron, Ljung-Box, ARCH LM tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
import os
import tempfile

# Statistical test imports
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from config import stationarity_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StationarityTester:
    """
    Comprehensive stationarity and time series diagnostics.
    """
    
    def __init__(self, significance_level: float = 0.05, max_observations: int = 250_000):
        self.alpha = significance_level
        self.max_observations = max_observations
        self.results = {}
    
    def _prepare_series(self, series: pd.Series) -> pd.Series:
        """Clean and cap very long series for expensive statistical tests."""
        series_clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(series_clean) > self.max_observations:
            # Use the most recent contiguous block so test results match the data
            # a researcher would actually trade next.
            series_clean = series_clean.iloc[-self.max_observations:]
        return series_clean
    
    def test_adf(self, series: pd.Series, maxlag: int = 20) -> Dict:
        """
        Augmented Dickey-Fuller test for unit root.
        H0: Series has unit root (non-stationary)
        H1: Series is stationary
        """
        series_clean = self._prepare_series(series)
        
        try:
            maxlag = min(maxlag, max(1, len(series_clean) // 10 - 1))
            result = adfuller(series_clean, maxlag=maxlag, autolag='AIC')
            
            adf_stat, pvalue, usedlag, nobs, critical_values, icbest = result
            
            is_stationary = pvalue < self.alpha
            
            output = {
                'test': 'ADF',
                'statistic': adf_stat,
                'pvalue': pvalue,
                'used_lag': usedlag,
                'nobs': nobs,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': 'Stationary' if is_stationary else 'Non-stationary (unit root)'
            }
            
            logger.info(f"ADF Test: statistic={adf_stat:.4f}, p-value={pvalue:.4f}, "
                       f"conclusion={output['conclusion']}")
            
            return output
            
        except Exception as e:
            logger.error(f"ADF test failed: {e}")
            return {'test': 'ADF', 'error': str(e)}
    
    def test_kpss(self, series: pd.Series, nlags: int = 20, regression: str = 'c') -> Dict:
        """
        KPSS test for stationarity.
        H0: Series is stationary
        H1: Series has unit root (non-stationary)
        
        regression: 'c' (constant) or 'ct' (constant + trend)
        """
        series_clean = self._prepare_series(series)
        
        try:
            nlags = min(nlags, max(1, len(series_clean) // 10 - 1))
            kpss_stat, pvalue, nlags_used, critical_values = kpss(
                series_clean, regression=regression, nlags=nlags
            )
            
            is_stationary = pvalue > self.alpha
            
            output = {
                'test': 'KPSS',
                'statistic': kpss_stat,
                'pvalue': pvalue,
                'nlags': nlags_used,
                'critical_values': critical_values,
                'is_stationary': is_stationary,
                'conclusion': 'Stationary' if is_stationary else 'Non-stationary (unit root)'
            }
            
            logger.info(f"KPSS Test: statistic={kpss_stat:.4f}, p-value={pvalue:.4f}, "
                       f"conclusion={output['conclusion']}")
            
            return output
            
        except Exception as e:
            logger.error(f"KPSS test failed: {e}")
            return {'test': 'KPSS', 'error': str(e)}
    
    def test_ljung_box(self, series: pd.Series, lags: int = 20) -> Dict:
        """
        Ljung-Box test for autocorrelation.
        H0: No autocorrelation up to lag k
        H1: Autocorrelation exists
        """
        series_clean = self._prepare_series(series)
        
        try:
            lags = min(lags, max(1, len(series_clean) - 1))
            result = acorr_ljungbox(series_clean, lags=lags, return_df=True)
            
            # Check if any p-value is below significance level
            has_autocorr = (result['lb_pvalue'] < self.alpha).any()
            
            output = {
                'test': 'Ljung-Box',
                'lags_tested': lags,
                'has_autocorrelation': has_autocorr,
                'min_pvalue': result['lb_pvalue'].min(),
                'max_statistic': result['lb_stat'].max(),
                'conclusion': 'Autocorrelation detected' if has_autocorr else 'No significant autocorrelation'
            }
            
            logger.info(f"Ljung-Box Test: min p-value={output['min_pvalue']:.4f}, "
                       f"conclusion={output['conclusion']}")
            
            return output
            
        except Exception as e:
            logger.error(f"Ljung-Box test failed: {e}")
            return {'test': 'Ljung-Box', 'error': str(e)}
    
    def test_arch(self, series: pd.Series, lags: int = 10) -> Dict:
        """
        ARCH LM test for volatility clustering (heteroskedasticity).
        H0: No ARCH effects
        H1: ARCH effects present (volatility clustering)
        """
        series_clean = self._prepare_series(series)
        
        try:
            lags = min(lags, max(1, len(series_clean) - 1))
            lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(series_clean, nlags=lags)
            
            has_arch = lm_pvalue < self.alpha
            
            output = {
                'test': 'ARCH LM',
                'lm_statistic': lm_stat,
                'lm_pvalue': lm_pvalue,
                'f_statistic': f_stat,
                'f_pvalue': f_pvalue,
                'lags': lags,
                'has_arch_effects': has_arch,
                'conclusion': 'ARCH effects detected (volatility clustering)' if has_arch else 'No ARCH effects'
            }
            
            logger.info(f"ARCH LM Test: LM p-value={lm_pvalue:.4f}, "
                       f"conclusion={output['conclusion']}")
            
            return output
            
        except Exception as e:
            logger.error(f"ARCH test failed: {e}")
            return {'test': 'ARCH LM', 'error': str(e)}
    
    def run_full_battery(self, series: pd.Series, name: str = "series") -> Dict:
        """
        Run complete battery of stationarity tests.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Running stationarity tests on: {name}")
        logger.info(f"{'='*60}")
        
        results = {
            'series_name': name,
            'n_obs_raw': int(series.dropna().shape[0]),
            'n_obs_tested': int(len(self._prepare_series(series))),
            'adf': self.test_adf(series),
            'kpss': self.test_kpss(series),
            'ljung_box': self.test_ljung_box(series),
            'arch': self.test_arch(series)
        }
        
        # Overall assessment
        adf_stationary = results['adf'].get('is_stationary', False)
        kpss_stationary = results['kpss'].get('is_stationary', False)
        
        if adf_stationary and kpss_stationary:
            overall = "STATIONARY (both tests agree)"
        elif not adf_stationary and not kpss_stationary:
            overall = "NON-STATIONARY (both tests agree)"
        else:
            overall = "INCONCLUSIVE (tests disagree - may need differencing)"
        
        results['overall_conclusion'] = overall
        
        logger.info(f"\nOverall conclusion: {overall}")
        logger.info(f"{'='*60}\n")
        
        self.results[name] = results
        return results
    
    def save_results(self, output_path: Path):
        """Save test results to JSON"""
        import json
        
        # Convert to JSON-serializable format
        serializable = {}
        for key, val in self.results.items():
            serializable[key] = self._make_serializable(val)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=output_path.parent,
            suffix=".tmp",
            encoding="utf-8",
        ) as f:
            json.dump(serializable, f, indent=2)
            temp_name = f.name
        os.replace(temp_name, output_path)
        
        logger.info(f"Saved stationarity test results to {output_path}")
    
    def _make_serializable(self, obj):
        """Convert numpy/pandas types to Python native types"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        else:
            return obj


def test_price_and_returns(parquet_path: Path, output_dir: Path):
    """
    Load data and test both price levels and returns for stationarity.
    """
    df = pd.read_parquet(parquet_path)
    
    tester = StationarityTester(
        significance_level=stationarity_config.significance_level,
        max_observations=stationarity_config.max_observations,
    )
    
    # Test price levels
    if 'close' in df.columns:
        tester.run_full_battery(df['close'], name='close_price')
    
    # Test log returns
    if 'log_return' in df.columns:
        tester.run_full_battery(df['log_return'], name='log_returns')
    elif 'close' in df.columns:
        log_returns = np.log(df['close']).diff()
        tester.run_full_battery(log_returns, name='log_returns')
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    tester.save_results(output_dir / 'stationarity_tests.json')
    
    return tester.results


if __name__ == "__main__":
    from config import CLEANED_PARQUET, REPORTS_DIR
    
    test_price_and_returns(CLEANED_PARQUET, REPORTS_DIR)
