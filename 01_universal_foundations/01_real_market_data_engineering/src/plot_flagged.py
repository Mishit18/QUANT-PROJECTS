# src/plot_flagged.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

IN_PARQ = Path("data/cleaned/INDIA_VIX_firstclean.parquet")
OUT_DIR = Path("reports/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "flagged_ticks.png"

def plot_flagged_sample(parquet_path: Path = IN_PARQ, out_path: Path = OUT_PATH,
                        max_points:int = 8000, last_days:int = 60):
    if not parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {parquet_path}. Run src/cleaning.py first.")

    df = pd.read_parquet(parquet_path)

    # if last_days specified, restrict to last N calendar days for clarity
    if last_days is not None:
        end = df.index.max()
        start = end - pd.Timedelta(days=last_days)
        df = df.loc[start:end]
        print(f"[INFO] plotting last {last_days} days: {start} -> {end} (rows: {len(df)})")

    # create a boolean mask of flagged ticks
    if "flag_bad_tick" not in df.columns:
        raise KeyError("flag_bad_tick column not found in firstclean parquet. Run first pass cleaning.")

    # Resample/decimate for plotting if too many points
    n = len(df)
    if n > max_points:
        idx_pos = np.linspace(0, n-1, max_points, dtype=int)
        df_plot = df.iloc[idx_pos]
        print(f"[INFO] decimated from {n} -> {len(df_plot)} points for plotting")
    else:
        df_plot = df

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(14,5))
    ax.plot(df_plot.index, df_plot["close"], lw=0.6, label="close")
    # overlay flagged points
    flagged = df_plot[df_plot["flag_bad_tick"]]
    ax.scatter(flagged.index, flagged["close"], color="red", s=10, label="flagged bad ticks", alpha=0.6)
    ax.set_title("Close price with flagged bad ticks (red)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved flagged ticks plot to: {out_path}")
    print(f"Flagged points plotted: {len(flagged)} out of {len(df_plot)} sampled points")

if __name__ == "__main__":
    plot_flagged_sample()
