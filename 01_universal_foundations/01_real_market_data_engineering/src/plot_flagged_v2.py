# src/plot_flagged_v2.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

PARQ = Path("data/cleaned/INDIA_VIX_secondclean.parquet")
OUT = Path("reports/plots/flagged_ticks_v2.png")
OUT.parent.mkdir(parents=True, exist_ok=True)

def plot_flagged_v2(parquet=PARQ, outpath=OUT, last_days=60, max_points=8000):
    if not parquet.exists():
        raise FileNotFoundError("Run cleaning_v2.py first to produce secondclean parquet.")

    df = pd.read_parquet(parquet)

    # restrict to last N days for clarity
    if last_days is not None:
        end = df.index.max()
        start = end - pd.Timedelta(days=last_days)
        df = df.loc[start:end]
        print(f"[INFO] Using last {last_days} days: {start} -> {end} (rows: {len(df)})")

    # decimate if too many points
    n = len(df)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points, dtype=int)
        dfp = df.iloc[idx]
        print(f"[INFO] Decimated {n} -> {len(dfp)} points for plotting")
    else:
        dfp = df

    fig, ax = plt.subplots(1, 1, figsize=(14,5))
    ax.plot(dfp.index, dfp["close"], lw=0.6, label="close")
    flagged = dfp[dfp.get("flag_bad_tick_v2", False)]
    ax.scatter(flagged.index, flagged["close"], color="red", s=10, alpha=0.7, label="flagged v2")
    ax.set_title("Close price with v2 flagged ticks (red)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved plot: {outpath}")
    print(f"Flagged plotted: {len(flagged)} out of {len(dfp)} sampled points")

if __name__ == "__main__":
    plot_flagged_v2()
