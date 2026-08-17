from __future__ import annotations

import math

import numpy as np
import pandas as pd


def quote_fill_probability(distance_from_mid_ticks: float, order_arrival_intensity: float, horizon_seconds: float) -> float:
    if distance_from_mid_ticks < 0 or order_arrival_intensity < 0 or horizon_seconds < 0:
        raise ValueError("inputs must be non-negative")
    adjusted_intensity = order_arrival_intensity * math.exp(-0.35 * distance_from_mid_ticks)
    return float(np.clip(1 - math.exp(-adjusted_intensity * horizon_seconds), 0, 1))


def toxic_flow_stress(spread_bps: float, adverse_move_bps: float, fill_probability: float, fee_bps: float = 0.2) -> dict[str, float | str]:
    expected_spread_capture = spread_bps * fill_probability
    expected_adverse_cost = adverse_move_bps * fill_probability
    net_edge = expected_spread_capture - expected_adverse_cost - fee_bps
    return {
        "expected_spread_capture_bps": expected_spread_capture,
        "expected_adverse_cost_bps": expected_adverse_cost,
        "net_edge_bps": net_edge,
        "verdict": "quote" if net_edge > 0 else "widen_or_skip",
    }


def spread_compression_scenarios(base_spread_bps: float, adverse_move_bps: float) -> pd.DataFrame:
    rows = []
    for compression in [0.0, 0.25, 0.50, 0.75]:
        spread = base_spread_bps * (1 - compression)
        fill_prob = quote_fill_probability(max(spread / 2, 0), 0.8, 1.0)
        stress = toxic_flow_stress(spread, adverse_move_bps, fill_prob)
        rows.append({"compression": compression, "spread_bps": spread, "fill_probability": fill_prob, **stress})
    return pd.DataFrame(rows)

