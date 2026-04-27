"""Tests for engine-stratified splits and baseline trainers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turboguard.models.rul.baselines import _eval, train_lightgbm, train_xgboost
from turboguard.models.rul.splits import engine_stratified_split


def _toy_gold(n_engines: int = 12, cycles: int = 40, n_features: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_engines + 1):
        for c in range(1, cycles + 1):
            features = rng.normal(size=n_features) + (cycles - c) * 0.05  # signal w/ RUL.
            row = {"unit_id": uid, "cycle": c, "RUL": cycles - c}
            for i, v in enumerate(features):
                row[f"feature_{i}"] = float(v)
            rows.append(row)
    df = pd.DataFrame(rows)
    df["RUL_clipped"] = df["RUL"].clip(upper=20)
    return df


def test_engine_stratified_split_no_engine_overlap():
    df = _toy_gold(n_engines=10)
    split = engine_stratified_split(df, val_fraction=0.3)
    train_engines = set(np.unique(split.groups_train).tolist())
    val_engines = set(np.unique(split.groups_val).tolist())
    assert train_engines.isdisjoint(val_engines)
    assert len(val_engines) >= 1
    assert split.X_train.shape[1] == split.X_val.shape[1] == len(split.feature_cols)


def test_engine_split_raises_on_missing_target():
    df = _toy_gold().drop(columns=["RUL_clipped"])
    with pytest.raises(KeyError, match="target column"):
        engine_stratified_split(df, target_col="RUL_clipped")


def test_eval_returns_expected_keys():
    y = np.array([10.0, 20.0, 30.0])
    metrics = _eval(y, y, prefix="x")
    assert metrics["x_rmse"] == pytest.approx(0.0)
    assert metrics["x_mae"] == pytest.approx(0.0)
    assert metrics["x_nasa_score"] == pytest.approx(0.0)


def test_eval_clips_negative_predictions():
    y = np.array([5.0, 10.0])
    pred = np.array([-3.0, 10.0])  # negative prediction is silently clipped to 0.
    metrics = _eval(y, pred, prefix="x")
    # If clipping works, the "5 vs 0" gap drives RMSE — squarely 5.
    assert metrics["x_rmse"] == pytest.approx(np.sqrt((5.0**2) / 2))


@pytest.mark.parametrize("trainer", [train_xgboost, train_lightgbm])
def test_baseline_trains_and_returns_metrics(tmp_path, trainer):
    """Smoke-test each baseline against a tiny gold table."""
    import mlflow

    mlflow.set_tracking_uri(f"file:///{tmp_path.as_posix()}/mlruns")
    df = _toy_gold(n_engines=12, cycles=40)
    split = engine_stratified_split(df, val_fraction=0.25, rng_seed=0)
    # Tiny model — we just need the loop to execute without exploding.
    fast = {"n_estimators": 30, "learning_rate": 0.2}
    if trainer is train_xgboost:
        fast["early_stopping_rounds"] = 5
    result = trainer(split, params=fast)
    for key in ("val_rmse", "val_mae", "val_nasa_score"):
        assert key in result.val_metrics
        assert np.isfinite(result.val_metrics[key])
