import numpy as np
import pandas as pd

from src.data.targets import construct_targets, volatility_scaled_returns


def test_volatility_scaled_returns_use_trailing_volatility():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    forward = pd.DataFrame({"A": [0.01, 0.02, 0.03, 0.04, 0.05]}, index=dates)
    trailing = pd.DataFrame({"A": [0.05, 0.04, 0.03, 0.02, 0.01]}, index=dates)

    scaled = volatility_scaled_returns(forward, trailing, vol_window=3, horizon=1)

    expected_denominator = trailing["A"].iloc[0:3].std()
    expected = forward.loc[dates[2], "A"] / expected_denominator
    assert np.isclose(scaled.loc[dates[2], "A"], expected)


def test_construct_vol_scaled_targets_preserves_cross_sectional_coverage():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    columns = [f"A{i}" for i in range(12)]
    base = np.linspace(100, 120, len(dates))
    prices = pd.DataFrame(
        {col: base * (1 + i / 1000) for i, col in enumerate(columns)},
        index=dates,
    )

    config = {"targets": {"horizon": 5, "method": "vol_scaled", "vol_window": 10}}
    targets, stats = construct_targets(prices, config)

    valid_counts = targets.notna().sum(axis=1)
    assert valid_counts.max() == len(columns)
    assert stats["mean_cs_valid"] > len(columns) / 2
