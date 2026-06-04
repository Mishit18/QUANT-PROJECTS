# src/explore_data.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PARQUET = Path("data/cleaned/INDIA_VIX_loaded.parquet")
OUT_DIR = Path("reports/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def basic_exploration(parquet_path: Path = PARQUET):
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}. Run load_data.py first.")

    df = pd.read_parquet(parquet_path)
    pd.options.display.float_format = "{:.4f}".format

    print("\n--- HEAD ---")
    print(df.head(8))
    print("\n--- TAIL ---")
    print(df.tail(8))
    print("\n--- DESCRIBE (numeric) ---")
    print(df.describe())
    print("\n--- Missing values ---")
    print(df.isnull().sum())

    print("\n--- Index frequency hint ---")
    try:
        print("Inferred freq:", pd.infer_freq(df.index[:50]))
    except:
        print("Could not infer frequency.")

    # ---------- SAVE PLOT (no GUI) ----------
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df.index, df["close"], linewidth=0.6)
    axes[0].set_title("Close Price (minute)")

    if "volume" in df.columns:
        axes[1].bar(df.index, df["volume"], width=0.0005)
        axes[1].set_title("Volume (minute)")
    else:
        axes[1].text(0.1, 0.5, "No volume column", transform=axes[1].transAxes)

    oc = df["open"] - df["close"]
    axes[2].plot(df.index, oc, linewidth=0.6)
    axes[2].set_title("Open - Close (diagnostic)")

    plt.tight_layout()

    out_path = OUT_DIR / "exploration_close_volume_oc.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n[OK] Saved plot to: {out_path}")

if __name__ == "__main__":
    basic_exploration()
