from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_stability_score(p_values: pd.Series, half_lives: pd.Series, max_half_life: float = 60.0) -> pd.DataFrame:
    if len(p_values) != len(half_lives):
        raise ValueError("p-values and half-lives must have equal length")
    out = pd.DataFrame({"p_value": p_values, "half_life": half_lives})
    out["cointegration_pass"] = out["p_value"] < 0.05
    out["half_life_pass"] = out["half_life"].between(2, max_half_life)
    out["stable_window"] = out["cointegration_pass"] & out["half_life_pass"]
    out["rolling_stability_6m"] = out["stable_window"].rolling(6, min_periods=1).mean()
    return out


def borrow_short_constraint(signal: pd.Series, borrow_available: pd.Series, hard_to_borrow_fee_bps: pd.Series) -> pd.DataFrame:
    aligned = pd.concat([signal.rename("signal"), borrow_available.rename("borrow_available"), hard_to_borrow_fee_bps.rename("borrow_fee_bps")], axis=1).dropna()
    aligned["tradable_signal"] = aligned["signal"]
    aligned.loc[(aligned["signal"] < 0) & (~aligned["borrow_available"].astype(bool)), "tradable_signal"] = 0
    aligned["fee_adjusted_edge_bps"] = np.where(aligned["tradable_signal"] < 0, -aligned["borrow_fee_bps"], 0)
    return aligned


def paper_trade_readiness(stability: pd.DataFrame, min_stable_share: float = 0.60) -> dict[str, float | str]:
    stable_share = float(stability["stable_window"].mean())
    recent_stability = float(stability["rolling_stability_6m"].iloc[-1])
    verdict = "paper_trade" if stable_share >= min_stable_share and recent_stability >= min_stable_share else "research_only"
    return {"stable_window_share": stable_share, "recent_stability": recent_stability, "verdict": verdict}

