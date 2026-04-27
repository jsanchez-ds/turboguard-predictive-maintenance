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

    Builds every new column once into a single ``pd.DataFrame`` and then concatenates
    in a single shot — this avoids the ``PerformanceWarning: DataFrame is highly
    fragmented`` that fires when each new column is assigned individually.
    """
    sensor_cols = list(sensor_cols)
    base = df.sort_values([group_col, "cycle"]).reset_index(drop=True)

    new_frames: list[pd.DataFrame] = []
    for w in windows:
        grouped = base.groupby(group_col, sort=False)[sensor_cols]
        roll = grouped.rolling(window=w, min_periods=1)
        mean = roll.mean().reset_index(level=0, drop=True)
        std = roll.std().fillna(0.0).reset_index(level=0, drop=True)
        rmin = roll.min().reset_index(level=0, drop=True)
        rmax = roll.max().reset_index(level=0, drop=True)

        slopes = pd.DataFrame(
            {
                f"{c}_slope_{w}": (
                    base.groupby(group_col, sort=False)[c]
                    .rolling(window=w, min_periods=2)
                    .apply(_slope, raw=True)
                    .reset_index(level=0, drop=True)
                    .fillna(0.0)
                    .to_numpy()
                )
                for c in sensor_cols
            },
            index=base.index,
        )

        mean.columns = [f"{c}_mean_{w}" for c in sensor_cols]
        std.columns = [f"{c}_std_{w}" for c in sensor_cols]
        rmin.columns = [f"{c}_min_{w}" for c in sensor_cols]
        rmax.columns = [f"{c}_max_{w}" for c in sensor_cols]
        new_frames.extend([mean, std, rmin, rmax, slopes])

    return pd.concat([base, *new_frames], axis=1)
