"""Online change-point features for sensor signals.

We use a lightweight CUSUM-style detector: cycles since the last point at
which the rolling mean crossed a multiple of the rolling standard deviation
relative to the historical mean. This is a standard early-warning signal in
predictive maintenance — degradation typically manifests as a sustained
mean-shift in a few sensors before the engine actually fails.

The features per sensor are:
  - {sensor}_cps_{w}    : cycles since the last detected change-point
  - {sensor}_cum_dev_{w}: cumulative absolute deviation since that point
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _cusum_change_points(values: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """Boolean array marking cycles where the rolling mean has shifted.

    A point is flagged when the rolling-`window` mean lies more than
    `threshold` historical standard deviations away from the historical mean.
    """
    n = len(values)
    if n < window + 2:
        return np.zeros(n, dtype=bool)
    flags = np.zeros(n, dtype=bool)
    for i in range(window, n):
        history = values[: i - window + 1]
        if history.size < 2:
            continue
        hist_mean = history.mean()
        hist_std = history.std()
        if hist_std < 1e-9:
            continue
        recent_mean = values[i - window + 1 : i + 1].mean()
        if abs(recent_mean - hist_mean) > threshold * hist_std:
            flags[i] = True
    return flags


def add_changepoint_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    window: int = 15,
    threshold: float = 2.5,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Append per-engine change-point features for each sensor."""
    sensor_cols = list(sensor_cols)
    out = df.sort_values([group_col, "cycle"]).reset_index(drop=True).copy()
    n = len(out)
    cps = {c: np.zeros(n) for c in sensor_cols}
    cum_dev = {c: np.zeros(n) for c in sensor_cols}

    for _, idx in out.groupby(group_col, sort=False).indices.items():
        for col in sensor_cols:
            series = out[col].values[idx]
            flags = _cusum_change_points(series, window=window, threshold=threshold)
            last_cp = -1
            cumulative = 0.0
            ref_mean = series[: max(window, 1)].mean()
            for j, pos in enumerate(idx):
                if flags[j]:
                    last_cp = j
                    cumulative = 0.0
                cps[col][pos] = (j - last_cp) if last_cp >= 0 else j
                cumulative += abs(series[j] - ref_mean)
                cum_dev[col][pos] = cumulative

    for col in sensor_cols:
        out[f"{col}_cps_{window}"] = cps[col]
        out[f"{col}_cum_dev_{window}"] = cum_dev[col]
    return out
