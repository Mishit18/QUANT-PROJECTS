from __future__ import annotations

import numpy as np
import pandas as pd


def shortfall_decomposition(
    fills: pd.DataFrame,
    arrival_price: float,
    side: str = "buy",
    decision_price: float | None = None,
) -> dict[str, float]:
    required = {"price", "quantity", "mid_price"}
    missing = required - set(fills.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    sign = 1.0 if side.lower() == "buy" else -1.0
    qty = fills["quantity"].to_numpy(dtype=float)
    total_qty = float(qty.sum())
    if total_qty <= 0:
        raise ValueError("total quantity must be positive")
    price = fills["price"].to_numpy(dtype=float)
    mid = fills["mid_price"].to_numpy(dtype=float)
    decision = arrival_price if decision_price is None else decision_price

    weighted_fill = float(np.sum(price * qty) / total_qty)
    implementation_shortfall_bps = sign * (weighted_fill - decision) / decision * 10000
    spread_cost_bps = float(np.sum(sign * (price - mid) / decision * qty) / total_qty * 10000)
    market_drift_bps = float(sign * (np.average(mid, weights=qty) - arrival_price) / decision * 10000)
    timing_cost_bps = implementation_shortfall_bps - spread_cost_bps - market_drift_bps
    return {
        "total_quantity": total_qty,
        "weighted_fill_price": weighted_fill,
        "implementation_shortfall_bps": implementation_shortfall_bps,
        "spread_cost_bps": spread_cost_bps,
        "market_drift_bps": market_drift_bps,
        "timing_residual_bps": timing_cost_bps,
    }


def participation_guardrail(
    fills: pd.DataFrame,
    market_volume_col: str = "market_volume",
    quantity_col: str = "quantity",
    max_participation: float = 0.15,
) -> pd.DataFrame:
    required = {market_volume_col, quantity_col}
    missing = required - set(fills.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = fills.copy()
    out["participation_rate"] = out[quantity_col] / out[market_volume_col].replace(0, np.nan)
    out["participation_breach"] = out["participation_rate"] > max_participation
    return out

