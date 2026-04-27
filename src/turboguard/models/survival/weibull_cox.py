"""Weibull AFT and Cox PH survival models for engine RUL.

Both fitters consume the engine-level panel produced by
:mod:`turboguard.models.survival.engine_features`. We expose three fitting
helpers:

* :func:`fit_weibull_aft` — parametric Weibull accelerated-failure-time model.
  Yields a closed-form survival curve and is robust on small datasets.
* :func:`fit_cox_ph` — semi-parametric Cox proportional-hazards model.

The accompanying :func:`predict_rul` converts a fitted survival model into a
predicted RUL for a panel of test engines: it computes the **conditional
expected total lifetime given the engine has already survived ``duration``
cycles**, then subtracts ``duration``. This is the standard survival-to-RUL
projection (Wang et al. 2008).

All trainers log to MLflow when a run is open at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlflow
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, WeibullAFTFitter
from lifelines.utils import concordance_index
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ..rul.nasa_score import nasa_score


class _SurvivalFitter(Protocol):
    """Common interface needed for RUL projection."""

    def predict_survival_function(self, X: pd.DataFrame, times: list[float]) -> pd.DataFrame: ...


@dataclass
class SurvivalResult:
    fitter: object
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None


# ----- helpers ----------------------------------------------------------------


def _drop_low_variance(X: pd.DataFrame, threshold: float = 1e-6) -> pd.DataFrame:
    return X.loc[:, X.std(axis=0) > threshold]


def _engine_split(
    panel: pd.DataFrame,
    val_fraction: float = 0.2,
    rng_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(rng_seed)
    engines = np.array(panel.index.tolist())
    rng.shuffle(engines)
    n_val = max(1, int(round(len(engines) * val_fraction)))
    val_engines = set(engines[:n_val].tolist())
    return panel[~panel.index.isin(val_engines)], panel[panel.index.isin(val_engines)]


def _val_metrics(
    fitter: _SurvivalFitter,
    val: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, float]:
    """C-index on the validation set (higher is better; 1.0 = perfect ordering)."""
    times = list(np.linspace(1, val["duration"].max() * 2, 200))
    sf = fitter.predict_survival_function(val[feature_cols], times=times)
    # Predicted lifetime ≈ first time at which S(t) drops below 0.5 (median).
    median_lifetime = []
    for col in sf.columns:
        s = sf[col].to_numpy()
        below = np.where(s <= 0.5)[0]
        median_lifetime.append(times[below[0]] if below.size > 0 else times[-1])
    median_lifetime = np.asarray(median_lifetime, dtype=float)
    cidx = concordance_index(val["duration"].to_numpy(), median_lifetime, val["event"].to_numpy())
    return {"val_cindex": float(cidx)}


def _test_metrics_from_rul(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0.0, None)
    return {
        "test_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
        "test_nasa_score": float(nasa_score(y_true, y_pred)),
    }


def predict_rul(
    fitter: _SurvivalFitter,
    test_panel: pd.DataFrame,
    feature_cols: list[str],
    train_duration_max: float | None = None,
    horizon_multiplier: float = 1.5,
) -> np.ndarray:
    """Project survival curves into RUL predictions.

    Uses the **conditional median residual lifetime**: smallest ``t > d``
    such that ``S(t) / S(d) <= 0.5``. Median is far more stable than the
    expected-value integral because it's robust to heavy survival-curve tails
    that explode in Cox PH when extrapolated past the largest training
    duration.

    Parameters
    ----------
    train_duration_max : float, optional
        Maximum engine lifetime observed in training. If provided, the time
        grid is clipped to ``train_duration_max * horizon_multiplier`` to
        avoid hallucinating beyond the support of the training data.
    """
    if train_duration_max is None:
        train_duration_max = float(test_panel["duration"].max())
    horizon = float(train_duration_max * horizon_multiplier)
    horizon = max(horizon, float(test_panel["duration"].max()) + 1.0)
    times = list(np.linspace(1.0, horizon, 400))
    times_arr = np.asarray(times)
    sf = fitter.predict_survival_function(test_panel[feature_cols], times=times)

    rul_pred = np.zeros(len(test_panel), dtype=float)
    for i, col in enumerate(sf.columns):
        s = sf[col].to_numpy()
        d = float(test_panel["duration"].iloc[i])

        # If we never see times > d (engine outlived our grid), predict 0 RUL.
        mask = times_arr > d
        if not mask.any():
            rul_pred[i] = 0.0
            continue

        # S(d): interpolate to the nearest grid point at or before d.
        s_d_idx = max(0, np.searchsorted(times_arr, d, side="right") - 1)
        s_at_d = max(s[s_d_idx], 1e-6)

        # Find median residual life: first t > d where S(t)/S(d) <= 0.5.
        cond_surv = s[mask] / s_at_d
        below_half = np.where(cond_surv <= 0.5)[0]
        if below_half.size > 0:
            median_total_lifetime = float(times_arr[mask][below_half[0]])
        else:
            # Survival never crosses 0.5 within horizon — clip to horizon.
            median_total_lifetime = float(times_arr[-1])
        rul_pred[i] = max(0.0, median_total_lifetime - d)
    return rul_pred


# ----- Weibull AFT ------------------------------------------------------------


def fit_weibull_aft(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame | None = None,
    val_fraction: float = 0.2,
    penalizer: float = 0.05,
    rng_seed: int = 0,
    feature_cols: list[str] | None = None,
    run_name: str = "weibull-aft",
) -> SurvivalResult:
    """Fit a Weibull AFT model and (optionally) score it on a test panel."""
    train, val = _engine_split(train_panel, val_fraction=val_fraction, rng_seed=rng_seed)
    if feature_cols is None:
        feature_cols = [c for c in train.columns if c not in {"duration", "event"}]
    # Drop near-constant columns; AFT optimisers struggle with them.
    keep = _drop_low_variance(train[feature_cols]).columns.tolist()
    feature_cols = list(keep)

    fitter = WeibullAFTFitter(penalizer=penalizer)
    fitter.fit(
        train[[*feature_cols, "duration", "event"]],
        duration_col="duration",
        event_col="event",
    )

    val_metrics = _val_metrics(fitter, val, feature_cols)
    test_metrics = None
    if test_panel is not None and "true_RUL" in test_panel.columns:
        rul_pred = predict_rul(
            fitter, test_panel, feature_cols, train_duration_max=float(train["duration"].max())
        )
        test_metrics = _test_metrics_from_rul(test_panel["true_RUL"].to_numpy(), rul_pred)

    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
        opened = True
    else:
        opened = False
    try:
        mlflow.log_params(
            {
                "model_family": "weibull_aft",
                "n_features": len(feature_cols),
                "n_train_engines": len(train),
                "n_val_engines": len(val),
                "penalizer": penalizer,
            }
        )
        mlflow.log_metrics(val_metrics)
        if test_metrics:
            mlflow.log_metrics(test_metrics)
    finally:
        if opened:
            mlflow.end_run()

    return SurvivalResult(fitter=fitter, val_metrics=val_metrics, test_metrics=test_metrics)


# ----- Cox PH -----------------------------------------------------------------


def fit_cox_ph(
    train_panel: pd.DataFrame,
    test_panel: pd.DataFrame | None = None,
    val_fraction: float = 0.2,
    penalizer: float = 0.05,
    rng_seed: int = 0,
    feature_cols: list[str] | None = None,
    run_name: str = "cox-ph",
) -> SurvivalResult:
    """Fit a Cox PH model and (optionally) score it on a test panel."""
    train, val = _engine_split(train_panel, val_fraction=val_fraction, rng_seed=rng_seed)
    if feature_cols is None:
        feature_cols = [c for c in train.columns if c not in {"duration", "event"}]
    keep = _drop_low_variance(train[feature_cols]).columns.tolist()
    feature_cols = list(keep)

    fitter = CoxPHFitter(penalizer=penalizer)
    fitter.fit(
        train[[*feature_cols, "duration", "event"]],
        duration_col="duration",
        event_col="event",
    )

    val_metrics = _val_metrics(fitter, val, feature_cols)
    test_metrics = None
    if test_panel is not None and "true_RUL" in test_panel.columns:
        rul_pred = predict_rul(
            fitter, test_panel, feature_cols, train_duration_max=float(train["duration"].max())
        )
        test_metrics = _test_metrics_from_rul(test_panel["true_RUL"].to_numpy(), rul_pred)

    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
        opened = True
    else:
        opened = False
    try:
        mlflow.log_params(
            {
                "model_family": "cox_ph",
                "n_features": len(feature_cols),
                "n_train_engines": len(train),
                "n_val_engines": len(val),
                "penalizer": penalizer,
            }
        )
        mlflow.log_metrics(val_metrics)
        if test_metrics:
            mlflow.log_metrics(test_metrics)
    finally:
        if opened:
            mlflow.end_run()

    return SurvivalResult(fitter=fitter, val_metrics=val_metrics, test_metrics=test_metrics)
