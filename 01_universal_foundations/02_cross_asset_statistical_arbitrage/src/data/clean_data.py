import pandas as pd
import numpy as np


def remove_outliers(df: pd.DataFrame, n_std: float = 5.0) -> pd.DataFrame:
    """Remove extreme cross-sectional outliers per date."""
    def clip_row(row):
        valid = row.notna()
        if valid.sum() < 10:
            return row
        mean = row[valid].mean()
        std = row[valid].std()
        lower = mean - n_std * std
        upper = mean + n_std * std
        return row.clip(lower, upper)
    
    return df.apply(clip_row, axis=1)


def handle_splits_dividends(prices: pd.DataFrame, adjustment_factors: pd.DataFrame) -> pd.DataFrame:
    """Apply adjustment factors for splits/dividends."""
    return prices * adjustment_factors


def detect_stale_prices(prices: pd.DataFrame, max_unchanged: int = 10) -> pd.DataFrame:
    """Flag stale prices (unchanged for N days)."""
    unchanged = (prices.diff() == 0).rolling(max_unchanged).sum()
    stale = unchanged >= max_unchanged
    return prices.where(~stale)


def remove_low_coverage(df: pd.DataFrame, min_coverage: float = 0.8) -> pd.DataFrame:
    """Drop dates with insufficient cross-sectional coverage."""
    coverage = df.notna().sum(axis=1) / len(df.columns)
    valid_dates = coverage >= min_coverage
    return df[valid_dates]


def align_timestamps(dfs: list) -> list:
    """Align multiple dataframes to common date index."""
    common_index = dfs[0].index
    for df in dfs[1:]:
        common_index = common_index.intersection(df.index)
    return [df.loc[common_index] for df in dfs]
