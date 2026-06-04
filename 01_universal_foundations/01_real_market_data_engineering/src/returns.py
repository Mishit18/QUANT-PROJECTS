# src/returns.py
import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("data/cleaned/INDIA_VIX_secondclean.parquet")
OUT_PATH = Path("data/cleaned/INDIA_VIX_returns.parquet")

def compute_returns():
    df = pd.read_parquet(IN_PATH)

    price = df["close_clean_v2"]

    df["log_return"] = np.log(price).diff()
    df["abs_return"] = df["log_return"].abs()
    df["sq_return"] = df["log_return"] ** 2

    df = df.dropna(subset=["log_return"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH)

    print("Saved returns dataset:", OUT_PATH)
    print("Return mean:", df["log_return"].mean())
    print("Return std:", df["log_return"].std())

    return df

if __name__ == "__main__":
    compute_returns()
