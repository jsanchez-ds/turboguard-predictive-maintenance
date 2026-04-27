"""Tests for engine-level aggregation and Weibull/Cox survival fitters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turboguard.models.survival.engine_features import (
    _slope_per_engine,
    aggregate_engine_features,
    make_test_panel,
    make_train_panel,
)
from turboguard.models.survival.weibull_cox import fit_cox_ph, fit_weibull_aft, predict_rul


def _toy_cycles(n_engines: int = 60, mean_lifetime: int = 80, n_sensors: int = 2) -> pd.DataFrame:
    """Synthetic engines with a per-engine 'frailty' that drives lifetime + sensor means.

    A frail engine fails sooner *and* shows a higher baseline reading on sensor 0,
    so survival models have a real covariate–lifetime relationship to fit.
    """
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_engines + 1):
        frailty = float(rng.normal(0.0, 1.0))
        lifetime = max(15, int(mean_lifetime - 20 * frailty + rng.normal(0, 5)))
        for c in range(1, lifetime + 1):
            row = {"unit_id": uid, "cycle": c, "RUL": lifetime - c}
            for s in range(n_sensors):
                if s == 0:
                    row[f"sensor_{s}"] = float(frailty + 0.02 * c + rng.normal(0, 0.2))
                else:
                    row[f"sensor_{s}"] = float(rng.normal(0, 0.5))
            rows.append(row)
    return pd.DataFrame(rows)


# ----- engine_features --------------------------------------------------------


def test_slope_per_engine_on_arange():
    assert _slope_per_engine(np.arange(10, dtype=float)) == 1.0


def test_aggregate_engine_features_one_row_per_engine():
    df = _toy_cycles(n_engines=5)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    out = aggregate_engine_features(df, sensor_cols)
    assert len(out) == 5
    assert "duration" in out.columns
    for col in sensor_cols:
        assert {f"{col}_mean", f"{col}_std", f"{col}_slope", f"{col}_last"} <= set(out.columns)


def test_train_panel_event_is_one():
    df = _toy_cycles()
    panel = make_train_panel(df, [c for c in df.columns if c.startswith("sensor_")])
    assert (panel["event"] == 1).all()


def test_test_panel_carries_true_rul():
    df = _toy_cycles(n_engines=4)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    truth = pd.Series([5, 7, 10, 13], index=[1, 2, 3, 4])
    panel = make_test_panel(df, sensor_cols, truth)
    assert (panel["event"] == 0).all()
    assert (panel["true_total_lifetime"] == panel["duration"] + panel["true_RUL"]).all()


# ----- weibull / cox ----------------------------------------------------------


@pytest.mark.parametrize("trainer", [fit_weibull_aft, fit_cox_ph])
def test_survival_fitter_runs_and_returns_cindex(tmp_path, trainer):
    """End-to-end smoke: aggregate, fit, return a finite C-index."""
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_cycles(n_engines=20, mean_lifetime=80)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    train_panel = make_train_panel(df, sensor_cols)
    result = trainer(train_panel, val_fraction=0.25, rng_seed=0)
    assert "val_cindex" in result.val_metrics
    assert 0.0 <= result.val_metrics["val_cindex"] <= 1.0


def test_predict_rul_returns_one_value_per_engine():
    """The conditional projection produces one finite RUL per test engine."""
    df = _toy_cycles(n_engines=15, mean_lifetime=80)
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    train_panel = make_train_panel(df, sensor_cols)
    # Fake test panel = train data truncated halfway.
    truncated = (
        df.groupby("unit_id")
        .apply(lambda s: s[s.cycle <= s.cycle.max() // 2])
        .reset_index(drop=True)
    )
    truth = pd.Series(
        df.groupby("unit_id").cycle.max() - truncated.groupby("unit_id").cycle.max()
    )
    test_panel = make_test_panel(truncated, sensor_cols, truth)

    res = fit_weibull_aft(train_panel, test_panel=test_panel, val_fraction=0.25)
    assert res.test_metrics is not None
    assert np.isfinite(res.test_metrics["test_rmse"])
    rul_pred = predict_rul(
        res.fitter,
        test_panel,
        feature_cols=[c for c in test_panel.columns if c not in {"duration", "event", "true_RUL", "true_total_lifetime"}],
    )
    assert rul_pred.shape == (len(test_panel),)
    assert np.all(np.isfinite(rul_pred))
    assert np.all(rul_pred >= 0.0)
