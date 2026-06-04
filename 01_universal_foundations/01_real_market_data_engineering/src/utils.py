# src/utils.py
"""
Utility functions for the market data engineering pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_parquet_safe(path: Path) -> pd.DataFrame:
    """
    Safely load parquet file with error handling.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    try:
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise


def save_parquet_safe(df: pd.DataFrame, path: Path):
    """
    Safely save DataFrame to parquet with error handling.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        logger.info(f"Saved {len(df)} rows to {path}")
    except Exception as e:
        logger.error(f"Failed to save to {path}: {e}")
        raise


def save_json(data: Dict, path: Path):
    """
    Save dictionary to JSON file.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise


def load_json(path: Path) -> Dict:
    """
    Load JSON file to dictionary.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate summary statistics for a DataFrame.
    """
    summary = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    
    if isinstance(df.index, pd.DatetimeIndex):
        summary['date_range'] = {
            'start': str(df.index.min()),
            'end': str(df.index.max()),
            'n_days': (df.index.max() - df.index.min()).days
        }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    return summary


def check_data_quality(df: pd.DataFrame, 
                       max_missing_pct: float = 5.0,
                       check_duplicates: bool = True) -> Dict:
    """
    Check data quality and return issues.
    """
    issues = []
    
    # Check missing values
    missing_pct = (df.isnull().sum() / len(df) * 100)
    high_missing = missing_pct[missing_pct > max_missing_pct]
    if len(high_missing) > 0:
        issues.append({
            'type': 'high_missing_values',
            'columns': high_missing.to_dict()
        })
    
    # Check duplicates
    if check_duplicates:
        n_duplicates = df.index.duplicated().sum()
        if n_duplicates > 0:
            issues.append({
                'type': 'duplicate_timestamps',
                'count': int(n_duplicates)
            })
    
    # Check infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            issues.append({
                'type': 'infinite_values',
                'column': col,
                'count': int(n_inf)
            })
    
    # Check constant columns
    for col in numeric_cols:
        if df[col].nunique() == 1:
            issues.append({
                'type': 'constant_column',
                'column': col,
                'value': df[col].iloc[0]
            })
    
    return {
        'n_issues': len(issues),
        'issues': issues,
        'quality_score': max(0, 100 - len(issues) * 10)
    }


def calculate_feature_importance_proxy(df: pd.DataFrame, 
                                       target_col: str = 'log_return',
                                       top_n: int = 20) -> pd.DataFrame:
    """
    Calculate simple feature importance using correlation with target.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found")
    
    # Get numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    # Calculate correlations
    correlations = {}
    for col in feature_cols:
        try:
            corr = df[col].corr(df[target_col])
            if not np.isnan(corr):
                correlations[col] = abs(corr)
        except:
            pass
    
    # Sort by absolute correlation
    importance_df = pd.DataFrame({
        'feature': list(correlations.keys()),
        'abs_correlation': list(correlations.values())
    }).sort_values('abs_correlation', ascending=False)
    
    return importance_df.head(top_n)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using z-score method.
    
    Returns boolean mask: True = outlier
    """
    mean = series.mean()
    std = series.std()
    z_scores = np.abs((series - mean) / std)
    return z_scores > threshold


def winsorize(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """
    Winsorize series by clipping extreme values.
    """
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    return series.clip(lower=lower_bound, upper=upper_bound)


def resample_ohlcv(df: pd.DataFrame, freq: str = '5T') -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.
    
    Args:
        df: DataFrame with OHLCV columns
        freq: Pandas frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour)
    
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    agg_dict = {}
    
    if 'open' in df.columns:
        agg_dict['open'] = 'first'
    if 'high' in df.columns:
        agg_dict['high'] = 'max'
    if 'low' in df.columns:
        agg_dict['low'] = 'min'
    if 'close' in df.columns:
        agg_dict['close'] = 'last'
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'
    
    resampled = df.resample(freq).agg(agg_dict)
    
    logger.info(f"Resampled from {len(df)} to {len(resampled)} rows (freq={freq})")
    
    return resampled


def calculate_returns_matrix(df: pd.DataFrame, 
                             price_col: str = 'close',
                             periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate returns for multiple periods.
    """
    returns_df = pd.DataFrame(index=df.index)
    
    for period in periods:
        returns_df[f'return_{period}'] = df[price_col].pct_change(period)
    
    return returns_df


def print_pipeline_status(step: str, status: str = "running", details: str = ""):
    """
    Print formatted pipeline status message.
    """
    symbols = {
        'running': 'RUN',
        'success': 'OK',
        'error': 'ERR',
        'warning': 'WARN'
    }
    
    symbol = symbols.get(status, 'INFO')
    
    if details:
        logger.info(f"{symbol} {step}: {details}")
    else:
        logger.info(f"{symbol} {step}")


def validate_ohlcv(df: pd.DataFrame) -> Dict:
    """
    Validate OHLCV data for consistency.
    """
    issues = []
    
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        return {
            'valid': False,
            'issues': [f"Missing columns: {missing_cols}"]
        }
    
    # Check: high >= low
    invalid_hl = (df['high'] < df['low']).sum()
    if invalid_hl > 0:
        issues.append(f"High < Low in {invalid_hl} rows")
    
    # Check: high >= open, close
    invalid_h_open = (df['high'] < df['open']).sum()
    invalid_h_close = (df['high'] < df['close']).sum()
    if invalid_h_open > 0:
        issues.append(f"High < Open in {invalid_h_open} rows")
    if invalid_h_close > 0:
        issues.append(f"High < Close in {invalid_h_close} rows")
    
    # Check: low <= open, close
    invalid_l_open = (df['low'] > df['open']).sum()
    invalid_l_close = (df['low'] > df['close']).sum()
    if invalid_l_open > 0:
        issues.append(f"Low > Open in {invalid_l_open} rows")
    if invalid_l_close > 0:
        issues.append(f"Low > Close in {invalid_l_close} rows")
    
    # Check: negative prices
    for col in required_cols:
        negative = (df[col] <= 0).sum()
        if negative > 0:
            issues.append(f"Negative/zero {col} in {negative} rows")
    
    return {
        'valid': len(issues) == 0,
        'n_issues': len(issues),
        'issues': issues
    }


if __name__ == "__main__":
    # Test utilities
    from config import CLEANED_PARQUET
    
    if CLEANED_PARQUET.exists():
        df = load_parquet_safe(CLEANED_PARQUET)
        
        print("\n=== Data Summary ===")
        summary = get_data_summary(df)
        print(f"Rows: {summary['n_rows']}")
        print(f"Columns: {summary['n_columns']}")
        
        print("\n=== Data Quality ===")
        quality = check_data_quality(df)
        print(f"Quality Score: {quality['quality_score']}/100")
        print(f"Issues: {quality['n_issues']}")
        
        print("\n=== OHLCV Validation ===")
        validation = validate_ohlcv(df)
        print(f"Valid: {validation['valid']}")
        if not validation['valid']:
            for issue in validation['issues']:
                print(f"  - {issue}")
