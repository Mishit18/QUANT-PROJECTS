# src/cleaning.py
import pandas as pd
import numpy as np
from pathlib import Path

IN_PARQ = Path("data/cleaned/INDIA_VIX_loaded.parquet")
OUT_PARQ = Path("data/cleaned/INDIA_VIX_firstclean.parquet")

def hampel_filter_for_series(s: pd.Series, window_size:int=5, n_sigmas:float=3.0) -> pd.Series:
    """
    Returns boolean mask: True = value is OK, False = flagged as outlier.
    window_size is half-window on each side (so total window = 2*window_size+1).
    Uses rolling median + rolling MAD.
    """
    # ensure float series
    s = s.astype(float)
    k = 1.4826  # scale factor for MAD to approximate std
    window = 2*window_size + 1

    # rolling median
    roll_median = s.rolling(window=window, center=True, min_periods=1).median()

    # rolling MAD (median absolute deviation around rolling median)
    mad = (s - roll_median).abs().rolling(window=window, center=True, min_periods=1).median()

    # threshold
    threshold = n_sigmas * k * mad
    diff = (s - roll_median).abs()

    mask = diff <= threshold
    # keep NaNs as True for mask (we will handle missing separately)
    return mask.fillna(True)

def first_pass_clean(in_parquet: Path = IN_PARQ, out_parquet: Path = OUT_PARQ,
                     price_window:int=3, returns_window:int=5,
                     price_nsigma:float=4.0, ret_nsigma:float=6.0,
                     ffill_limit:int=3):
    if not in_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {in_parquet}. Run load_data.py first.")

    df = pd.read_parquet(in_parquet)

    # ensure numeric
    for col in ("open","high","low","close","volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # flag zero-volume rows (do not drop yet)
    df["flag_zero_volume"] = (df.get("volume", 0) == 0)

    # compute log returns on close
    df["log_return"] = np.log(df["close"]).diff()

    # Hampel on price and on returns
    # For price use a smaller window; for returns a wider window.
    close_series = df["close"].fillna(method="ffill")
    price_mask = hampel_filter_for_series(close_series, window_size=price_window, n_sigmas=price_nsigma)
    ret_mask = hampel_filter_for_series(df["log_return"].fillna(0), window_size=returns_window, n_sigmas=ret_nsigma)

    df["flag_bad_tick"] = ~(price_mask & ret_mask)

    # create a cleaned close column: set bad ticks to NaN and forward-fill small gaps
    df["close_clean"] = df["close"].where(~df["flag_bad_tick"], np.nan)
    df["close_clean"] = df["close_clean"].ffill(limit=ffill_limit)

    # Additional: create a 'close_imputed' that falls back to original close if ffill failed
    df["close_imputed"] = df["close_clean"].fillna(df["close"])

    # diagnostics
    n_total = len(df)
    n_bad = int(df["flag_bad_tick"].sum())
    n_zero = int(df["flag_zero_volume"].sum())
    print(f"Total rows: {n_total}; bad-ticks flagged: {n_bad}; zero-volume rows: {n_zero}")

    # save cleaned parquet
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    print(f"Saved first-clean parquet to {out_parquet}")

    return df

if __name__ == "__main__":
    first_pass_clean()
