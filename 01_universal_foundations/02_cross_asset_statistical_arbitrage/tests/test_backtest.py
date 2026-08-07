import numpy as np
import pandas as pd

from src.backtest.costs import compute_transaction_costs
from src.backtest.engine import BacktestEngine
from src.backtest.portfolio import Portfolio


def _config():
    return {
        "backtest": {
            "leverage": 1.0,
            "long_short": True,
            "tcost_bps": 0.0,
            "slippage_bps": 0.0,
            "rebalance_freq": 1,
        }
    }


def test_transaction_costs_include_entering_and_exiting_names():
    prev_weights = pd.Series({"A": 0.10})
    curr_weights = pd.Series({"B": -0.10})

    cost = compute_transaction_costs(prev_weights, curr_weights, tcost_bps=10_000)

    assert cost == 0.10


def test_backtest_uses_next_period_returns():
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    columns = [f"A{i}" for i in range(12)]
    alpha = pd.Series(np.arange(12, dtype=float), index=columns)
    predictions = pd.DataFrame([alpha, alpha, alpha], index=dates)
    returns = pd.DataFrame(0.0, index=dates, columns=columns)
    returns.iloc[1] = np.linspace(-0.02, 0.02, len(columns))
    returns.iloc[2] = -returns.iloc[1]
    volumes = pd.DataFrame(1_000_000.0, index=dates, columns=columns)

    engine = BacktestEngine(_config())
    result = engine.run(predictions, returns, volumes)

    expected_weights = Portfolio().construct_weights(alpha, method="rank")
    expected_first_return = (expected_weights * returns.iloc[1]).sum()
    assert result["returns"].iloc[0] == expected_first_return
