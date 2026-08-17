from __future__ import annotations

import numpy as np
import pandas as pd


def queue_fill_probability(
    queue_ahead: float,
    trade_through_rate: float,
    cancel_rate: float,
    horizon_events: int,
) -> float:
    if queue_ahead < 0 or trade_through_rate < 0 or cancel_rate < 0 or horizon_events < 0:
        raise ValueError("queue, rates, and horizon must be non-negative")
    effective_depletion = (trade_through_rate + cancel_rate) * horizon_events
    if queue_ahead == 0:
        return 1.0
    return float(np.clip(1 - np.exp(-effective_depletion / max(queue_ahead, 1e-9)), 0, 1))


def latency_sensitivity_table(
    signals: pd.DataFrame,
    edge_col: str = "expected_edge_bps",
    half_spread_bps: float = 1.0,
    latencies_events: tuple[int, ...] = (0, 1, 3, 5, 10),
    decay_per_event: float = 0.08,
) -> pd.DataFrame:
    if edge_col not in signals.columns:
        raise ValueError(f"missing column: {edge_col}")
    base_edge = float(signals[edge_col].mean())
    rows = []
    for latency in latencies_events:
        decayed_edge = base_edge * np.exp(-decay_per_event * latency)
        net_edge = decayed_edge - half_spread_bps
        rows.append(
            {
                "latency_events": latency,
                "gross_edge_bps": decayed_edge,
                "net_edge_after_half_spread_bps": net_edge,
                "tradable": bool(net_edge > 0),
            }
        )
    return pd.DataFrame(rows)


def adverse_selection_report(trades: pd.DataFrame) -> dict[str, float]:
    required = {"side", "fill_price", "mid_after_fill"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    signed = np.where(trades["side"].str.lower().eq("buy"), 1.0, -1.0)
    adverse_move = signed * (trades["mid_after_fill"] - trades["fill_price"])
    return {
        "trades": int(len(trades)),
        "mean_adverse_selection_bps": float(np.mean(adverse_move / trades["fill_price"] * 10000)),
        "toxic_fill_share": float(np.mean(adverse_move < 0)),
        "p10_adverse_selection_bps": float(np.percentile(adverse_move / trades["fill_price"] * 10000, 10)),
    }

