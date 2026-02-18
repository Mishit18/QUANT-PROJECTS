import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_ohlcv(data_dir: Path, tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Load OHLCV for ticker list. Assumes CSV per ticker."""
    data = {}
    for ticker in tickers:
        path = data_dir / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=['date'], index_col='date')
            data[ticker] = df
    return data


def build_panel(ohlcv_dict: Dict[str, pd.DataFrame], field: str = 'close') -> pd.DataFrame:
    """Extract single field into wide panel."""
    panels = {}
    for ticker, df in ohlcv_dict.items():
        if field in df.columns:
            panels[ticker] = df[field]
    return pd.DataFrame(panels)


def forward_fill_limited(df: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    """Forward fill with max gap limit."""
    return df.fillna(method='ffill', limit=limit)


def filter_universe(prices: pd.DataFrame, volumes: pd.DataFrame,
                    min_price: float = 5.0, min_volume: float = 1e6) -> pd.DataFrame:
    """Apply liquidity filters."""
    price_valid = prices >= min_price
    volume_valid = volumes >= min_volume
    valid = price_valid & volume_valid
    return prices.where(valid)


def compute_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Simple returns over N periods."""
    return prices.pct_change(periods)


def compute_log_returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Log returns over N periods."""
    return np.log(prices / prices.shift(periods))
