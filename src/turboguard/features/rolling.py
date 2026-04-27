"""Rolling-window features over sensor signals.

Per-engine rolling statistics (mean, std, min, max, slope) are the bread-and-butter
features for RUL regression on C-MAPSS — they capture the degradation trend without
needing deep learning.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    """Linear-regression slope over a rolling window."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    # Closed-form slope of a simple linear regression.
    x_mean = x.mean()
    y_mean = values.mean()
    num = ((x - x_mean) * (values - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return float(num / den) if den != 0 else 0.0


def add_rolling_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    windows: Iterable[int] = (5, 15, 30),
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Add per-engine rolling mean/std/min/max/slope features.

    Parameters
    ----------
    df : DataFrame
        Long-format sensor table with one row per (unit_id, cycle).
    sensor_cols : iterable of str
        Sensor column names to compute features for.
    windows : iterable of int
        Rolling window sizes (in cycles).
    group_col : str
        Column identifying each engine.
    """
    out = df.sort_values([group_col, "cycle"]).copy()
    for w in windows:
        grouped = out.groupby(group_col, sort=False)[list(sensor_cols)]
        roll = grouped.rolling(window=w, min_periods=1)
        out[[f"{c}_mean_{w}" for c in sensor_cols]] = roll.mean().reset_index(level=0, drop=True)
        out[[f"{c}_std_{w}" for c in sensor_cols]] = (
            roll.std().fillna(0.0).reset_index(level=0, drop=True)
        )
        out[[f"{c}_min_{w}" for c in sensor_cols]] = roll.min().reset_index(level=0, drop=True)
        out[[f"{c}_max_{w}" for c in sensor_cols]] = roll.max().reset_index(level=0, drop=True)
        # Slope is more expensive — apply per group.
        for c in sensor_cols:
            out[f"{c}_slope_{w}"] = (
                out.groupby(group_col, sort=False)[c]
                .rolling(window=w, min_periods=2)
                .apply(_slope, raw=True)
                .reset_index(level=0, drop=True)
                .fillna(0.0)
            )
    return out
