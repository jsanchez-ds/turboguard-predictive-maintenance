"""Tests for spectral, change-point, and pipeline feature modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turboguard.features.changepoint import _cusum_change_points, add_changepoint_features
from turboguard.features.pipeline import (
    FeatureConfig,
    build_features,
    find_constant_sensors,
    load_gold,
    save_gold,
)
from turboguard.features.spectral import _fft_band_energy, add_spectral_features


# ----- spectral ---------------------------------------------------------------


def test_fft_constant_signal_has_zero_energy():
    low, mid, high = _fft_band_energy(np.full(64, 3.7))
    assert low == 0.0 and mid == 0.0 and high == 0.0


def test_fft_low_band_dominates_for_slow_signal():
    # A slowly-varying signal (period = full window) puts all energy in the low band.
    n = 240
    t = np.arange(n)
    slow = np.sin(2 * np.pi * t / n)  # exactly one full cycle across the window.
    low, mid, high = _fft_band_energy(slow)
    assert low > mid
    assert low > high


def test_fft_high_band_dominates_for_fast_signal():
    # Near-Nyquist oscillation (period = 2 samples) lives entirely in the high band.
    n = 240
    t = np.arange(n)
    fast = np.sin(np.pi * t)  # frequency = Nyquist (1/2 sample rate).
    low, mid, high = _fft_band_energy(fast)
    assert high > low
    assert high > mid


# ----- change-point -----------------------------------------------------------


def test_cusum_flags_after_mean_shift():
    rng = np.random.default_rng(1)
    pre = rng.normal(loc=0.0, scale=1.0, size=120)
    post = rng.normal(loc=10.0, scale=1.0, size=80)
    series = np.concatenate([pre, post])
    flags = _cusum_change_points(series, window=15, threshold=2.5)
    # The shift starts at index 120 — at least one flag should land in the post region.
    assert flags[120:].sum() > 0
    # The very early region should have nothing.
    assert flags[:30].sum() == 0


def test_changepoint_features_shape():
    df = pd.DataFrame(
        {
            "unit_id": [1] * 50 + [2] * 50,
            "cycle": list(range(1, 51)) * 2,
            "sensor_1": np.concatenate(
                [np.zeros(40), np.ones(10) * 5, np.zeros(40), np.ones(10) * 5]
            ),
        }
    )
    out = add_changepoint_features(df, sensor_cols=["sensor_1"], window=5, threshold=2.0)
    assert {"sensor_1_cps_5", "sensor_1_cum_dev_5"} <= set(out.columns)
    assert len(out) == len(df)


# ----- pipeline ---------------------------------------------------------------


def _toy_cmapss(n_engines: int = 4, cycles_per_engine: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for uid in range(1, n_engines + 1):
        for c in range(1, cycles_per_engine + 1):
            rows.append(
                {
                    "unit_id": uid,
                    "cycle": c,
                    "sensor_1": float(rng.normal(loc=c * 0.05, scale=0.5)),  # drifting
                    "sensor_2": 1.0,  # constant — should be pruned
                    "sensor_3": float(rng.normal(loc=0.0, scale=1.0)),  # noise only
                }
            )
    return pd.DataFrame(rows)


def test_find_constant_sensors():
    df = _toy_cmapss()
    const = find_constant_sensors(df, ["sensor_1", "sensor_2", "sensor_3"])
    assert const == ["sensor_2"]


def test_build_features_drops_constants_and_adds_columns():
    df = _toy_cmapss()
    cfg = FeatureConfig(rolling_windows=(5,), fft_window=10, cusum_window=5)
    gold, kept = build_features(df, config=cfg)
    assert "sensor_2" not in kept
    # Original cols + features for the two surviving sensors.
    assert any(col.startswith("sensor_1_mean_5") for col in gold.columns)
    assert any(col.endswith("_fft_hi_lo_10") for col in gold.columns)
    assert any(col.startswith("sensor_3_cps_5") for col in gold.columns)


def test_save_and_load_gold_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    out = save_gold(df, tmp_path / "gold.parquet")
    pd.testing.assert_frame_equal(load_gold(out), df)


def test_build_features_idempotent_on_constants_extra_drop():
    df = _toy_cmapss()
    cfg = FeatureConfig(rolling_windows=(5,), fft_window=10, cusum_window=5, extra_drop=("sensor_3",))
    _, kept = build_features(df, config=cfg)
    assert "sensor_3" not in kept


@pytest.mark.parametrize("window", [5, 15])
def test_spectral_feature_added_per_window(window):
    df = _toy_cmapss()
    out = add_spectral_features(df, sensor_cols=["sensor_1"], window=window)
    assert f"sensor_1_fft_low_{window}" in out.columns
