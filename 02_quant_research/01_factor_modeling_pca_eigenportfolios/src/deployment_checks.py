from __future__ import annotations

import numpy as np
import pandas as pd


def turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    numeric = weights.select_dtypes(include=[np.number])
    return numeric.diff().abs().sum(axis=1).fillna(0.0)


def factor_capacity_proxy(weights: pd.DataFrame, adv: pd.Series, max_adv_share: float = 0.05) -> pd.DataFrame:
    numeric = weights.select_dtypes(include=[np.number])
    rows = []
    for factor in numeric.columns:
        dollar_weight = numeric[factor].abs()
        aligned_adv = adv.reindex(dollar_weight.index).fillna(adv.median())
        capacity = (aligned_adv * max_adv_share / dollar_weight.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).min()
        rows.append({"factor": factor, "capacity_proxy_notional": float(capacity), "max_adv_share": max_adv_share})
    return pd.DataFrame(rows)


def deploy_gate_table(performance: pd.DataFrame) -> pd.DataFrame:
    required = {"factor", "sharpe_ci_05", "max_drawdown", "hit_rate"}
    missing = required - set(performance.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = performance.copy()
    out["deploy_gate"] = np.where(
        (out["sharpe_ci_05"] > 0) & (out["max_drawdown"] > -0.25) & (out["hit_rate"] > 0.50),
        "research_further",
        "reject_or_hedge_only",
    )
    return out

