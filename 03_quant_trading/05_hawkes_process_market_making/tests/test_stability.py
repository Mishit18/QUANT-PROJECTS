import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hawkes.stability import branching_stability, calibration_stability_grid, hawkes_failure_modes


def test_hawkes_stability_and_failure_modes():
    alpha = np.array([[0.2, 0.1], [0.05, 0.3]])
    check = branching_stability(alpha)
    grid = calibration_stability_grid([alpha, alpha * 1.2, alpha * 0.8])
    modes = hawkes_failure_modes(grid, pd.Series([0.2, 0.1, 0.03]))

    assert check["verdict"] == "stable"
    assert "rolling_min_margin" in grid.columns
    assert modes["verdict"] in {"research_only", "trade_with_limits", "model_usable"}

