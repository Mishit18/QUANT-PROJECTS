import numpy as np

from src.data.fi2010 import build_lob_features


def test_build_lob_features_uses_all_ten_levels() -> None:
    levels = np.zeros((4, 40), dtype=np.float32)
    for level in range(10):
        offset = level * 4
        levels[:, offset] = 101 + level
        levels[:, offset + 1] = 10 + level
        levels[:, offset + 2] = 100 - level
        levels[:, offset + 3] = 20 + level
    features = build_lob_features(levels)
    assert features.shape == (4, 57)
    assert {"spread_l1", "microprice_l1", "depth_imbalance_10"} <= set(features.columns)
    assert np.isfinite(features.to_numpy()).all()


def test_build_lob_features_rejects_wrong_shape() -> None:
    try:
        build_lob_features(np.zeros((4, 39), dtype=np.float32))
    except ValueError as error:
        assert "40" in str(error)
    else:
        raise AssertionError("Expected wrong-shaped FI-2010 input to fail")
