from __future__ import annotations

import numpy as np
import pandas as pd


def capacity_cost_grid(gross_edge_bps: float, daily_turnover: float, capital_grid: list[float], cost_bps_grid: list[float]) -> pd.DataFrame:
    rows = []
    for capital in capital_grid:
        for cost_bps in cost_bps_grid:
            net_edge_bps = gross_edge_bps - cost_bps * daily_turnover
            rows.append(
                {
                    "capital": capital,
                    "cost_bps": cost_bps,
                    "daily_turnover": daily_turnover,
                    "net_edge_bps": net_edge_bps,
                    "annualized_net_pnl": capital * net_edge_bps / 10000 * 252,
                    "deployable": net_edge_bps > 0,
                }
            )
    return pd.DataFrame(rows)


def deploy_reject_gate(ic_mean: float, ic_ir: float, turnover: float, net_return: float, max_turnover: float = 3.0) -> dict[str, float | str | list[str]]:
    reasons = []
    if ic_mean <= 0:
        reasons.append("non_positive_ic")
    if ic_ir < 0.20:
        reasons.append("weak_ic_ir")
    if turnover > max_turnover:
        reasons.append("turnover_too_high")
    if net_return <= 0:
        reasons.append("negative_net_return")
    return {"verdict": "deploy_candidate" if not reasons else "reject", "reasons": reasons, "ic_mean": ic_mean, "ic_ir": ic_ir, "turnover": turnover, "net_return": net_return}

