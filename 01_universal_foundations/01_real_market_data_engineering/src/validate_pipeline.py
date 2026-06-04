# src/validate_pipeline.py
"""
Validation script to verify pipeline outputs and data quality.
Run this after pipeline.py to ensure everything worked correctly.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

from config import (
    RAW_CSV, LOADED_PARQUET, CLEANED_PARQUET, STATIONARY_PARQUET,
    FEATURES_PARQUET, REPORTS_DIR
)
from utils import (
    load_parquet_safe, get_data_summary, check_data_quality,
    validate_ohlcv, print_pipeline_status
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineValidator:
    """
    Comprehensive pipeline validation.
    """
    
    def __init__(self):
        self.results = {
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': 0,
            'details': []
        }
    
    def test(self, name: str, condition: bool, error_msg: str = "", warning: bool = False):
        """
        Run a test and record result.
        """
        if condition:
            self.results['tests_passed'] += 1
            print_pipeline_status(name, 'success')
            self.results['details'].append({
                'test': name,
                'status': 'passed'
            })
        else:
            if warning:
                self.results['warnings'] += 1
                print_pipeline_status(name, 'warning', error_msg)
                self.results['details'].append({
                    'test': name,
                    'status': 'warning',
                    'message': error_msg
                })
            else:
                self.results['tests_failed'] += 1
                print_pipeline_status(name, 'error', error_msg)
                self.results['details'].append({
                    'test': name,
                    'status': 'failed',
                    'message': error_msg
                })
    
    def validate_file_exists(self, path: Path, name: str):
        """
        Check if file exists.
        """
        self.test(
            f"File exists: {name}",
            path.exists(),
            f"File not found: {path}"
        )
        return path.exists()
    
    def validate_loaded_data(self):
        """
        Validate loaded data.
        """
        logger.info("\n=== Validating Loaded Data ===")
        
        if not self.validate_file_exists(LOADED_PARQUET, "loaded data"):
            return
        
        df = load_parquet_safe(LOADED_PARQUET)
        
        # Check row count
        self.test(
            "Loaded data has rows",
            len(df) > 0,
            "DataFrame is empty"
        )
        
        # Check required columns
        required = ['open', 'high', 'low', 'close']
        has_required = all(col in df.columns for col in required)
        self.test(
            "Has OHLC columns",
            has_required,
            f"Missing columns: {[c for c in required if c not in df.columns]}"
        )
        
        # Check datetime index
        self.test(
            "Has DatetimeIndex",
            isinstance(df.index, pd.DatetimeIndex),
            "Index is not DatetimeIndex"
        )
        
        # Validate OHLCV consistency
        if has_required:
            validation = validate_ohlcv(df)
            self.test(
                "OHLCV data is consistent",
                validation['valid'],
                f"OHLCV issues: {validation.get('issues', [])}",
                warning=True
            )
    
    def validate_cleaned_data(self):
        """
        Validate cleaned data.
        """
        logger.info("\n=== Validating Cleaned Data ===")
        
        if not self.validate_file_exists(CLEANED_PARQUET, "cleaned data"):
            return
        
        df = load_parquet_safe(CLEANED_PARQUET)
        
        # Check flagging column exists
        self.test(
            "Has bad tick flags",
            'flag_bad_tick' in df.columns or 'flag_bad_tick_v2' in df.columns,
            "No flagging column found"
        )
        
        # Check outlier percentage
        if 'flag_bad_tick' in df.columns:
            outlier_pct = df['flag_bad_tick'].sum() / len(df) * 100
            self.test(
                "Outlier percentage reasonable (<5%)",
                outlier_pct < 5.0,
                f"Outlier percentage: {outlier_pct:.2f}%",
                warning=True
            )
        
        # Check missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        self.test(
            "Missing values reasonable (<10%)",
            missing_pct < 10.0,
            f"Missing values: {missing_pct:.2f}%",
            warning=True
        )
        
        # Check data quality
        quality = check_data_quality(df, max_missing_pct=10.0)
        self.test(
            "Data quality score >70",
            quality['quality_score'] > 70,
            f"Quality score: {quality['quality_score']}/100",
            warning=True
        )
    
    def validate_stationarity_tests(self):
        """
        Validate stationarity test results.
        """
        logger.info("\n=== Validating Stationarity Tests ===")
        
        test_file = REPORTS_DIR / 'stationarity_tests.json'
        if not self.validate_file_exists(test_file, "stationarity tests"):
            return
        
        with open(test_file, 'r') as f:
            results = json.load(f)
        
        # Check that tests were run
        self.test(
            "Stationarity tests completed",
            len(results) > 0,
            "No test results found"
        )
        
        # Check for returns stationarity
        if 'log_returns' in results:
            returns_result = results['log_returns']
            
            # ADF test
            if 'adf' in returns_result:
                adf_stationary = returns_result['adf'].get('is_stationary', False)
                self.test(
                    "Returns are stationary (ADF)",
                    adf_stationary,
                    "ADF test indicates non-stationarity",
                    warning=True
                )
            
            # KPSS test
            if 'kpss' in returns_result:
                kpss_stationary = returns_result['kpss'].get('is_stationary', False)
                self.test(
                    "Returns are stationary (KPSS)",
                    kpss_stationary,
                    "KPSS test indicates non-stationarity",
                    warning=True
                )
    
    def validate_stationary_data(self):
        """
        Validate drift-removed data.
        """
        logger.info("\n=== Validating Stationary Data ===")
        
        if not self.validate_file_exists(STATIONARY_PARQUET, "stationary data"):
            return
        
        df = load_parquet_safe(STATIONARY_PARQUET)
        
        # Check for decomposition columns
        has_trend = 'close_trend' in df.columns
        has_detrended = 'close_detrended' in df.columns
        
        self.test(
            "Has trend component",
            has_trend,
            "Trend column not found"
        )
        
        self.test(
            "Has detrended component",
            has_detrended,
            "Detrended column not found"
        )
        
        # Check detrended series properties
        if has_detrended:
            detrended = df['close_detrended'].dropna()
            
            # Mean should be close to zero
            mean_close_to_zero = abs(detrended.mean()) < 0.1 * detrended.std()
            self.test(
                "Detrended mean ≈ 0",
                mean_close_to_zero,
                f"Detrended mean: {detrended.mean():.6f}",
                warning=True
            )
    
    def validate_features(self):
        """
        Validate engineered features.
        """
        logger.info("\n=== Validating Features ===")
        
        if not self.validate_file_exists(FEATURES_PARQUET, "features"):
            return
        
        df = load_parquet_safe(FEATURES_PARQUET)
        
        # Check feature count
        n_features = len(df.columns)
        self.test(
            "Has sufficient features (>40)",
            n_features > 40,
            f"Only {n_features} features found"
        )
        
        # Check for key feature categories
        feature_categories = {
            'returns': ['log_return', 'simple_return'],
            'volatility': ['parkinson_vol', 'garman_klass_vol'],
            'momentum': ['momentum_', 'rsi_'],
            'seasonality': ['hour_sin', 'hour_cos']
        }
        volume_informative = bool(df.get('volume_is_informative', pd.Series([True])).iloc[0])
        if volume_informative:
            feature_categories['volume'] = ['volume_ma', 'vwap']
        
        for category, patterns in feature_categories.items():
            has_category = any(
                any(pattern in col for pattern in patterns)
                for col in df.columns
            )
            self.test(
                f"Has {category} features",
                has_category,
                f"No {category} features found",
                warning=True
            )
        
        self.test(
            "Volume feature decision recorded",
            'volume_is_informative' in df.columns,
            "No volume informativeness flag found",
            warning=True
        )
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        has_inf = False
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                has_inf = True
                break
        
        self.test(
            "No infinite values",
            not has_inf,
            "Found infinite values in features"
        )
        
        # Check feature variance
        zero_var_features = []
        volume_informative = bool(df.get('volume_is_informative', pd.Series([True])).iloc[0])
        for col in numeric_cols:
            if col == 'volume' and not volume_informative:
                continue
            if df[col].var() == 0:
                zero_var_features.append(col)
        
        self.test(
            "No zero-variance features",
            len(zero_var_features) == 0,
            f"Zero-variance features: {zero_var_features}",
            warning=True
        )
    
    def validate_backtest(self):
        """
        Validate backtest results.
        """
        logger.info("\n=== Validating Backtest Results ===")
        
        backtest_file = REPORTS_DIR / 'backtest_results.parquet'
        if not self.validate_file_exists(backtest_file, "backtest results"):
            return
        
        df = load_parquet_safe(backtest_file)
        
        # Check required columns
        required = ['equity', 'position', 'returns']
        has_required = all(col in df.columns for col in required)
        self.test(
            "Has required backtest columns",
            has_required,
            f"Missing: {[c for c in required if c not in df.columns]}"
        )
        
        if not has_required:
            return
        
        # Check equity is positive
        self.test(
            "Equity remains positive",
            (df['equity'] > 0).all(),
            "Equity went negative (bankruptcy)"
        )
        
        # Check returns are reasonable
        returns = df['returns'].dropna()
        max_return = returns.abs().max()
        self.test(
            "Returns are reasonable (<100%)",
            max_return < 1.0,
            f"Extreme return detected: {max_return:.2%}",
            warning=True
        )
    
    def validate_all(self):
        """
        Run all validation tests.
        """
        logger.info("\n" + "="*70)
        logger.info("PIPELINE VALIDATION")
        logger.info("="*70)
        
        self.validate_loaded_data()
        self.validate_cleaned_data()
        self.validate_stationarity_tests()
        self.validate_stationary_data()
        self.validate_features()
        self.validate_backtest()
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Tests Passed:  {self.results['tests_passed']}")
        logger.info(f"Tests Failed:  {self.results['tests_failed']}")
        logger.info(f"Warnings:      {self.results['warnings']}")
        
        total_tests = (self.results['tests_passed'] + 
                      self.results['tests_failed'] + 
                      self.results['warnings'])
        
        if total_tests > 0:
            success_rate = self.results['tests_passed'] / total_tests * 100
            logger.info(f"Success Rate:  {success_rate:.1f}%")
        
        if self.results['tests_failed'] == 0:
            logger.info("\nPIPELINE VALIDATION PASSED")
        else:
            logger.info("\nPIPELINE VALIDATION FAILED")
            logger.info("\nFailed tests:")
            for detail in self.results['details']:
                if detail['status'] == 'failed':
                    logger.info(f"  - {detail['test']}: {detail.get('message', '')}")
        
        logger.info("="*70 + "\n")
        
        # Save results
        results_file = REPORTS_DIR / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Validation results saved to {results_file}")
        
        return self.results['tests_failed'] == 0


if __name__ == "__main__":
    validator = PipelineValidator()
    success = validator.validate_all()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if success else 1)
