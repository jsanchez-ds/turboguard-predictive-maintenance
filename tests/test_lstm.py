"""Tests for the LSTM RUL module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from turboguard.models.rul.lstm import (
    LSTMRUL,
    SensorScaler,
    make_test_windows,
    make_windows,
    train_lstm,
)


def _toy_engines(n_engines: int = 6, cycles: int = 50, n_sensors: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_engines + 1):
        for c in range(1, cycles + 1):
            row = {"unit_id": uid, "cycle": c, "RUL": cycles - c}
            for s in range(n_sensors):
                row[f"sensor_{s}"] = float(rng.normal(loc=(cycles - c) * 0.1 + s, scale=0.3))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["RUL_clipped"] = df["RUL"].clip(upper=20)
    return df


def test_sensor_scaler_zero_centers_train_data():
    df = _toy_engines()
    cols = [c for c in df.columns if c.startswith("sensor_")]
    scaler = SensorScaler.fit(df, cols)
    z = scaler.transform(df)
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-6)


def test_sensor_scaler_handles_constant_columns():
    df = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0, 2.0]})
    scaler = SensorScaler.fit(df, ["a", "b"])
    z = scaler.transform(df)
    # Constant column collapses to zero (mean centered, std forced to 1.0).
    assert np.allclose(z[:, 0], 0.0)


def test_make_windows_shape_and_alignment():
    df = _toy_engines(n_engines=2, cycles=40)
    cols = [c for c in df.columns if c.startswith("sensor_")]
    scaler = SensorScaler.fit(df, cols)
    X, y, groups = make_windows(df, cols, window=10, scaler=scaler)
    # 40 - 10 + 1 = 31 windows per engine, two engines.
    assert X.shape == (62, 10, len(cols))
    assert y.shape == (62,)
    assert set(groups.tolist()) == {1, 2}


def test_make_windows_pads_short_engines():
    df = _toy_engines(n_engines=1, cycles=5)
    cols = [c for c in df.columns if c.startswith("sensor_")]
    scaler = SensorScaler.fit(df, cols)
    X, _, groups = make_windows(df, cols, window=10, scaler=scaler)
    # The single short engine still produces exactly one (front-padded) window.
    assert X.shape == (1, 10, len(cols))
    assert groups.tolist() == [1]


def test_make_test_windows_one_per_engine():
    df = _toy_engines(n_engines=3, cycles=40)
    cols = [c for c in df.columns if c.startswith("sensor_")]
    scaler = SensorScaler.fit(df, cols)
    X, groups = make_test_windows(df, cols, window=15, scaler=scaler)
    assert X.shape == (3, 15, len(cols))
    assert groups.tolist() == [1, 2, 3]


def test_lstm_forward_output_shape():
    model = LSTMRUL(n_features=4, hidden_size=8, num_layers=1, dropout=0.0)
    x = torch.randn(7, 12, 4)
    out = model(x)
    assert out.shape == (7,)


def test_train_lstm_smoke(tmp_path):
    """Tiny end-to-end run on synthetic data — verifies the loop & MLflow logging."""
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_engines(n_engines=8, cycles=40)
    train_df = df[df.unit_id <= 6]
    val_df = df[df.unit_id > 6]
    cols = [c for c in df.columns if c.startswith("sensor_")]
    result = train_lstm(
        train_df,
        val_df,
        cols,
        window=10,
        hidden_size=8,
        num_layers=1,
        max_epochs=3,
        patience=2,
        batch_size=32,
        device="cpu",
        run_name="smoke",
    )
    assert isinstance(result.val_metrics["val_rmse"], float)
    assert np.isfinite(result.val_metrics["val_nasa_score"])
    # Best epoch must be set.
    assert result.best_epoch >= 1
