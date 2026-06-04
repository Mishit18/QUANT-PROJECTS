import pandas as pd
from pathlib import Path

from config import RAW_CSV, LOADED_PARQUET, DATETIME_FORMAT


def _parse_datetimes(date_series: pd.Series) -> pd.Series:
    """
    Parse the raw feed timestamps.

    The source file is ISO-like ``YYYY-MM-DD HH:MM:SS``. Do not use
    ``dayfirst=True`` here: it silently drops dates where day > 12 and swaps
    month/day on ambiguous dates, which destroys the time series.
    """
    parsed = pd.to_datetime(date_series, format=DATETIME_FORMAT, errors="coerce")

    if parsed.isna().any():
        fallback = pd.to_datetime(date_series[parsed.isna()], errors="coerce")
        parsed.loc[parsed.isna()] = fallback

    return parsed


def load_market_data(path: Path = RAW_CSV, save_parquet: bool = True) -> pd.DataFrame:
    """
    Load INDIA VIX minute-level CSV,
    parse datetime (dayfirst format),
    convert OHLCV to numeric,
    set DatetimeIndex,
    run basic diagnostics,
    save parquet snapshot.
    """

    # --- Load CSV ---
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Check for required column: 'date' ---
    if "date" not in df.columns:
        raise ValueError("Column 'date' not found in CSV. Check file structure.")

    # --- Parse datetime ---
    df["datetime"] = _parse_datetimes(df["date"])

    # Drop rows where datetime failed to parse
    before = len(df)
    df = df.dropna(subset=["datetime"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped} rows due to invalid datetime parsing.")

    # --- Sort and set index ---
    df = df.sort_values("datetime")
    df = df.set_index("datetime")
    df = df[~df.index.duplicated(keep="last")]

    # --- Convert OHLCV to numeric ---
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Basic diagnostics ---
    print("\n===== BASIC DATA SUMMARY =====")
    print("Date range:", df.index.min(), "->", df.index.max())
    print("Total rows:", len(df))
    print("Duplicate timestamps:", df.index.duplicated().sum())
    print("Monotonic timestamp index:", df.index.is_monotonic_increasing)
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("================================\n")

    # --- Save parquet snapshot ---
    if save_parquet:
        LOADED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(LOADED_PARQUET)
        print(f"[OK] Saved loaded parquet to: {LOADED_PARQUET}")

    return df


if __name__ == "__main__":
    load_market_data()
