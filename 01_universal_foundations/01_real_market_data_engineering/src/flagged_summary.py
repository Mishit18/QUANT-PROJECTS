# src/flagged_summary.py
import pandas as pd
from pathlib import Path

PARQ = Path("data/cleaned/INDIA_VIX_secondclean.parquet")

def summary(parquet=PARQ):
    if not parquet.exists():
        raise FileNotFoundError("Run cleaning_v2.py first.")

    df = pd.read_parquet(parquet)
    n = len(df)
    flagged = df.get("flag_bad_tick_v2", pd.Series(False, index=df.index)).sum()
    pct = flagged / n * 100.0

    print("Total rows:", n)
    print("Flagged (v2):", int(flagged), f"({pct:.3f}%)")

    # monthly flagged counts
    if not df.empty:
        s = df.get("flag_bad_tick_v2", pd.Series(False, index=df.index)).astype(int)
        monthly = s.resample("M").sum()
        monthly_pct = monthly / df.resample("M").size() * 100
        print("\n--- Monthly flagged counts (last 12 months) ---")
        print(monthly.tail(12))

        print("\n--- Monthly flagged % (last 12 months) ---")
        print(monthly_pct.tail(12))

    # flagged interval histogram (gaps between flagged points)
    flagged_idx = df.index[df.get("flag_bad_tick_v2", False)]
    if len(flagged_idx) > 1:
        diffs = (flagged_idx.to_series().diff().dropna().dt.total_seconds() / 60.0)  # minutes
        print("\n--- Flagged-gap stats (minutes) ---")
        print("median:", diffs.median(), "mean:", diffs.mean(), "max:", diffs.max())
    else:
        print("\nNot enough flagged points to compute gaps.")

if __name__ == "__main__":
    summary()
