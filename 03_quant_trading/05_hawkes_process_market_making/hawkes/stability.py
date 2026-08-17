from __future__ import annotations

import numpy as np
import pandas as pd


def branching_stability(alpha: np.ndarray, beta: np.ndarray | None = None) -> dict[str, float | str]:
    matrix = np.asarray(alpha, dtype=float)
    if beta is not None:
        matrix = matrix / np.maximum(np.asarray(beta, dtype=float), 1e-9)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    return {
        "spectral_radius": spectral_radius,
        "stability_margin": 1.0 - spectral_radius,
        "verdict": "stable" if spectral_radius < 1 else "explosive",
    }


def calibration_stability_grid(alpha_samples: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for idx, alpha in enumerate(alpha_samples):
        check = branching_stability(alpha)
        rows.append({"window": idx, **check})
    out = pd.DataFrame(rows)
    out["rolling_min_margin"] = out["stability_margin"].rolling(3, min_periods=1).min()
    return out


def hawkes_failure_modes(stability: pd.DataFrame, ks_p_values: pd.Series) -> dict[str, float | str]:
    unstable_share = float((stability["verdict"] != "stable").mean())
    poor_residual_share = float((ks_p_values < 0.05).mean())
    if unstable_share > 0 or poor_residual_share > 0.5:
        verdict = "research_only"
    elif poor_residual_share > 0.25:
        verdict = "trade_with_limits"
    else:
        verdict = "model_usable"
    return {"unstable_window_share": unstable_share, "poor_residual_share": poor_residual_share, "verdict": verdict}

