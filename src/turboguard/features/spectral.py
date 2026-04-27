"""Frequency-domain (FFT) features over a rolling window.

For each cycle of each engine, we look at the last `window` cycles of a sensor
and compute the energy distributed across coarse frequency bands. Sensor
"nervousness" — the ratio of high-frequency to low-frequency energy — is a
known leading indicator of mechanical degradation in C-MAPSS literature.

The features per (sensor, window) are:
  - {sensor}_fft_low_{w}     : sum of |FFT|^2 in the low-frequency band
  - {sensor}_fft_mid_{w}     : sum of |FFT|^2 in the mid-frequency band
  - {sensor}_fft_high_{w}    : sum of |FFT|^2 in the high-frequency band
  - {sensor}_fft_hi_lo_{w}   : high / (low + 1e-9)  — degradation proxy
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _fft_band_energy(values: np.ndarray) -> tuple[float, float, float]:
    """Return (low, mid, high) FFT energy bands of a 1-D signal.

    Bands are split into thirds of the positive-frequency spectrum.
    Constant signals return zero energy in every band.
    """
    n = len(values)
    if n < 4:
        return 0.0, 0.0, 0.0
    x = values - values.mean()
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    # Drop the DC term (we already mean-centered).
    spectrum = spectrum[1:]
    if spectrum.size == 0:
        return 0.0, 0.0, 0.0
    third = max(1, spectrum.size // 3)
    low = float(spectrum[:third].sum())
    mid = float(spectrum[third : 2 * third].sum())
    high = float(spectrum[2 * third :].sum())
    return low, mid, high


def add_spectral_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    window: int = 30,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Add rolling FFT band-energy features per engine.

    Slow path: a single Python loop per (engine, sensor, cycle). C-MAPSS is
    small enough (low six-figure rows total) that this is fine and keeps the
    code transparent.
    """
    sensor_cols = list(sensor_cols)
    out = df.sort_values([group_col, "cycle"]).reset_index(drop=True).copy()

    n = len(out)
    feat_low = {c: np.zeros(n) for c in sensor_cols}
    feat_mid = {c: np.zeros(n) for c in sensor_cols}
    feat_high = {c: np.zeros(n) for c in sensor_cols}

    for _, idx in out.groupby(group_col, sort=False).indices.items():
        for col in sensor_cols:
            series = out[col].values[idx]
            for j, pos in enumerate(idx):
                start = max(0, j - window + 1)
                low, mid, high = _fft_band_energy(series[start : j + 1])
                feat_low[col][pos] = low
                feat_mid[col][pos] = mid
                feat_high[col][pos] = high

    for col in sensor_cols:
        out[f"{col}_fft_low_{window}"] = feat_low[col]
        out[f"{col}_fft_mid_{window}"] = feat_mid[col]
        out[f"{col}_fft_high_{window}"] = feat_high[col]
        out[f"{col}_fft_hi_lo_{window}"] = feat_high[col] / (feat_low[col] + 1e-9)
    return out
