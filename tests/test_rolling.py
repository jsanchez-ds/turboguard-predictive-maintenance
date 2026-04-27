"""Tests for rolling features."""

import numpy as np
import pandas as pd

from turboguard.features.rolling import _slope, add_rolling_features


def test_slope_is_one_on_arange():
    assert _slope(np.arange(10, dtype=float)) == 1.0


def test_slope_is_zero_on_constant():
    assert _slope(np.full(5, 7.0)) == 0.0


def test_rolling_features_shape_and_grouping():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_1": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
        }
    )
    out = add_rolling_features(df, sensor_cols=["sensor_1"], windows=(2,))
    # Original cols + 5 new cols (mean/std/min/max/slope) for one sensor / one window.
    assert set(out.columns) >= {
        "sensor_1_mean_2",
        "sensor_1_std_2",
        "sensor_1_min_2",
        "sensor_1_max_2",
        "sensor_1_slope_2",
    }
    # Engines must not leak into each other.
    eng1_max = out.loc[out.unit_id == 1, "sensor_1_max_2"].max()
    eng2_min = out.loc[out.unit_id == 2, "sensor_1_min_2"].min()
    assert eng1_max < eng2_min
