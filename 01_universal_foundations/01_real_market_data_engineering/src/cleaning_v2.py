# src/cleaning_v2.py
"""
Robust OHLC cleaning for minute-level market data.

The cleaner preserves raw prices in ``*_raw`` columns, flags suspicious rows,
repairs short gaps with bounded forward/backward fill, and enforces OHLC
invariants after imputation. This avoids the common failure mode where only
``close`` is patched while ``open/high/low`` remain inconsistent.
"""
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from config import CLEANED_PARQUET, LOADED_PARQUET, cleaning_config


PRICE_COLS = ("open", "high", "low", "close")
IN_PARQ = LOADED_PARQUET
OUT_PARQ = CLEANED_PARQUET


def hampel_filter_for_series(
    s: pd.Series,
    window_size: int = 5,
    n_sigmas: float = 3.0,
    min_scale: float = 1e-10,
) -> pd.Series:
    """
    Return a boolean mask where True means the observation is acceptable.

    A zero local MAD is common in index data with many unchanged ticks. Treating
    every nonzero move as an outlier in those windows is too aggressive, so the
    filter falls back to "accept" when the local robust scale is effectively
    zero.
    """
    s = pd.to_numeric(s, errors="coerce").astype(float)
    window = 2 * window_size + 1
    scale_factor = 1.4826

    roll_median = s.rolling(window=window, center=True, min_periods=1).median()
    mad = (s - roll_median).abs().rolling(window=window, center=True, min_periods=1).median()

    robust_scale = scale_factor * mad
    threshold = n_sigmas * robust_scale
    diff = (s - roll_median).abs()

    mask = (diff <= threshold) | (robust_scale <= min_scale)
    return mask.fillna(True)


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _ohlc_invalid_mask(df: pd.DataFrame) -> pd.Series:
    required = set(PRICE_COLS)
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing OHLC columns: {missing}")

    high_bound = df[list(PRICE_COLS)].max(axis=1)
    low_bound = df[list(PRICE_COLS)].min(axis=1)

    return (
        df[list(PRICE_COLS)].isna().any(axis=1)
        | (df[list(PRICE_COLS)] <= 0).any(axis=1)
        | (df["high"] < high_bound)
        | (df["low"] > low_bound)
    )


def _repair_ohlc(df: pd.DataFrame, bad_mask: pd.Series, ffill_limit: int, bfill_limit: int) -> pd.DataFrame:
    cleaned = df.copy()

    for col in PRICE_COLS:
        cleaned[f"{col}_raw"] = cleaned[col]
        series = cleaned[col].where(~bad_mask, np.nan)
        series = series.ffill(limit=ffill_limit).bfill(limit=bfill_limit)
        cleaned[col] = series.fillna(cleaned[f"{col}_raw"])

    # Enforce OHLC invariants after imputation. This is preferable to dropping
    # rows in a minute bar pipeline because calendar continuity matters.
    row_max = cleaned[list(PRICE_COLS)].max(axis=1)
    row_min = cleaned[list(PRICE_COLS)].min(axis=1)
    cleaned["high"] = row_max
    cleaned["low"] = row_min

    return cleaned


def clean_market_data(
    df: pd.DataFrame,
    price_window: int = cleaning_config.price_window,
    returns_window: int = cleaning_config.returns_window,
    price_nsigma: float = cleaning_config.price_nsigma,
    ret_nsigma: float = cleaning_config.returns_nsigma,
    ffill_limit: int = cleaning_config.ffill_limit,
    bfill_limit: int = cleaning_config.bfill_limit,
    min_scale: float = cleaning_config.min_robust_scale,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Clean a loaded OHLCV DataFrame and return ``(cleaned_df, summary)``.
    """
    df = _coerce_numeric(df, [*PRICE_COLS, "volume"])

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex before cleaning")

    raw_ohlc_invalid = _ohlc_invalid_mask(df)
    raw_log_return = np.log(df["close"].where(df["close"] > 0)).diff()

    price_mask = hampel_filter_for_series(
        df["close"].ffill(),
        window_size=price_window,
        n_sigmas=price_nsigma,
        min_scale=min_scale,
    )
    return_mask = hampel_filter_for_series(
        raw_log_return.fillna(0.0),
        window_size=returns_window,
        n_sigmas=ret_nsigma,
        min_scale=min_scale,
    )

    volume_is_informative = "volume" in df.columns and df["volume"].fillna(0).abs().sum() > 0
    if volume_is_informative:
        zero_volume = df["volume"].fillna(0) <= cleaning_config.min_volume_threshold
    else:
        zero_volume = pd.Series(False, index=df.index)

    df["flag_ohlc_invalid_raw"] = raw_ohlc_invalid
    df["flag_price_outlier"] = ~price_mask
    df["flag_return_outlier"] = ~return_mask
    df["flag_zero_volume"] = zero_volume
    df["flag_bad_tick"] = (
        df["flag_ohlc_invalid_raw"]
        | df["flag_price_outlier"]
        | df["flag_return_outlier"]
    )

    cleaned = _repair_ohlc(df, df["flag_bad_tick"], ffill_limit, bfill_limit)
    cleaned["close_clean"] = cleaned["close"]
    cleaned["log_return"] = np.log(cleaned["close"].where(cleaned["close"] > 0)).diff()
    cleaned["simple_return"] = cleaned["close"].pct_change()
    cleaned["abs_return"] = cleaned["log_return"].abs()
    cleaned["squared_return"] = cleaned["log_return"] ** 2
    cleaned["volume_is_informative"] = bool(volume_is_informative)

    post_ohlc_invalid = _ohlc_invalid_mask(cleaned)

    summary = {
        "total_rows": int(len(cleaned)),
        "raw_ohlc_invalid": int(raw_ohlc_invalid.sum()),
        "price_outliers": int((~price_mask).sum()),
        "return_outliers": int((~return_mask).sum()),
        "bad_ticks": int(cleaned["flag_bad_tick"].sum()),
        "bad_ticks_pct": float(cleaned["flag_bad_tick"].mean() * 100),
        "post_clean_ohlc_invalid": int(post_ohlc_invalid.sum()),
        "volume_is_informative": bool(volume_is_informative),
    }

    return cleaned, summary


def second_pass_clean(
    in_parquet: Path = IN_PARQ,
    out_parquet: Path = OUT_PARQ,
    price_window: int = cleaning_config.price_window,
    returns_window: int = cleaning_config.returns_window,
    price_nsigma: float = cleaning_config.price_nsigma,
    ret_nsigma: float = cleaning_config.returns_nsigma,
    ffill_limit: int = cleaning_config.ffill_limit,
):
    if not in_parquet.exists():
        raise FileNotFoundError(f"Input parquet missing: {in_parquet}")

    df = pd.read_parquet(in_parquet)
    cleaned, summary = clean_market_data(
        df,
        price_window=price_window,
        returns_window=returns_window,
        price_nsigma=price_nsigma,
        ret_nsigma=ret_nsigma,
        ffill_limit=ffill_limit,
    )

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_parquet)

    print("Second-pass cleaning summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Saved cleaned parquet to {out_parquet}")

    return cleaned


if __name__ == "__main__":
    second_pass_clean()
