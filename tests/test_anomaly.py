"""Tests for Isolation Forest and LSTM autoencoder anomaly detectors."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turboguard.models.anomaly.autoencoder import (
    LSTMAutoencoder,
    score_autoencoder,
    train_autoencoder,
)
from turboguard.models.anomaly.isolation_forest import (
    early_warning_lead,
    score_engines,
    select_healthy_cycles,
    train_isolation_forest,
)


def _toy_gold(n_engines: int = 8, base_cycles: int = 60) -> pd.DataFrame:
    """Toy gold: linear drift in feature_0, noise elsewhere; engines fail at random cycles."""
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_engines + 1):
        cycles = base_cycles + int(rng.integers(-10, 10))
        for c in range(1, cycles + 1):
            row = {"unit_id": uid, "cycle": c, "RUL": cycles - c}
            row["feature_0"] = 0.05 * c + rng.normal(0, 0.2)  # drifts up over life
            row["feature_1"] = rng.normal(0, 1.0)
            row["feature_2"] = rng.normal(0, 1.0)
            rows.append(row)
    return pd.DataFrame(rows)


def _toy_raw(n_engines: int = 6, base_cycles: int = 60, n_sensors: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_engines + 1):
        cycles = base_cycles + int(rng.integers(-10, 10))
        for c in range(1, cycles + 1):
            row = {"unit_id": uid, "cycle": c, "RUL": cycles - c}
            for s in range(n_sensors):
                drift = 0.03 * c if s == 0 else 0.0
                row[f"sensor_{s}"] = float(drift + rng.normal(0, 0.3))
            rows.append(row)
    return pd.DataFrame(rows)


# ----- Isolation Forest -------------------------------------------------------


def test_select_healthy_cycles_returns_first_fraction_per_engine():
    df = _toy_gold()
    healthy = select_healthy_cycles(df, healthy_fraction=0.25)
    # Each engine: cutoff = round(0.25 * max_cycle).
    assert healthy.unit_id.nunique() == df.unit_id.nunique()
    for uid, sub in healthy.groupby("unit_id"):
        max_cycle = df[df.unit_id == uid].cycle.max()
        assert sub.cycle.max() <= max(1, int(round(max_cycle * 0.25)))


def test_isolation_forest_flags_late_life_more_than_early_life(tmp_path):
    """Late-life cycles should have a higher anomaly rate than early-life cycles."""
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_gold(n_engines=10, base_cycles=80)
    feature_cols = ["feature_0", "feature_1", "feature_2"]
    result = train_isolation_forest(
        df, feature_cols=feature_cols, healthy_fraction=0.25, n_estimators=80
    )
    scored = score_engines(result, df)
    # Compare anomaly rates in the first vs last 25% of each engine's life.
    df_with_score = df.merge(scored[["unit_id", "cycle", "is_anomaly"]], on=["unit_id", "cycle"])
    early_rate = []
    late_rate = []
    for _, sub in df_with_score.groupby("unit_id"):
        max_cycle = sub.cycle.max()
        early = sub[sub.cycle <= max_cycle * 0.25]
        late = sub[sub.cycle >= max_cycle * 0.75]
        early_rate.append(early.is_anomaly.mean())
        late_rate.append(late.is_anomaly.mean())
    assert np.mean(late_rate) > np.mean(early_rate)


def test_early_warning_lead_returns_one_row_per_engine(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_gold(n_engines=6, base_cycles=80)
    result = train_isolation_forest(df, n_estimators=80)
    scored = score_engines(result, df)
    leads = early_warning_lead(scored, df, consecutive=3)
    assert len(leads) == df.unit_id.nunique()
    assert {"unit_id", "first_alarm_cycle", "failure_cycle", "lead_time"} <= set(leads.columns)


# ----- LSTM autoencoder -------------------------------------------------------


def test_autoencoder_forward_shape():
    model = LSTMAutoencoder(n_features=3, hidden_size=8, latent_size=4)
    x = torch.randn(5, 12, 3)
    out = model(x)
    assert out.shape == x.shape


def test_train_autoencoder_smoke(tmp_path):
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_raw(n_engines=8, base_cycles=60)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    result = train_autoencoder(
        df,
        sensor_cols,
        window=10,
        hidden_size=8,
        latent_size=4,
        max_epochs=2,
        device="cpu",
    )
    scored = score_autoencoder(result, df, device="cpu")
    assert {"unit_id", "cycle", "recon_mse", "is_anomaly"} <= set(scored.columns)
    assert np.isfinite(scored["recon_mse"]).all()


# torch import used in test above
import torch  # noqa: E402
