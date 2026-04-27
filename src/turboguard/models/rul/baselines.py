"""XGBoost and LightGBM RUL baselines with MLflow tracking.

Conventions:
  * Train target is piecewise-linearly clipped at 125 (Heimes 2008).
  * Evaluation against test ground truth uses the **un-clipped** RUL.
  * Validation uses an engine-stratified holdout (no engine appears in both).

Each ``train_*`` function:
  1. Fits the booster on the train split with early stopping on val.
  2. Logs hyperparameters, metrics (RMSE, MAE, NASA score) and the model with
     a signature into MLflow under the active run/experiment.
  3. Returns the fitted model and a metrics dict.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .nasa_score import nasa_score
from .splits import EngineSplit


@dataclass
class TrainResult:
    model: object
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None


def _eval(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute RMSE / MAE / NASA score; clip predictions to be non-negative."""
    y_pred_clipped = np.clip(y_pred, 0.0, None)
    return {
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_clipped))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred_clipped)),
        f"{prefix}_nasa_score": float(nasa_score(y_true, y_pred_clipped)),
    }


# ----- XGBoost ----------------------------------------------------------------


XGB_DEFAULTS: dict = {
    "objective": "reg:squarederror",
    "n_estimators": 800,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 30,
    "n_jobs": -1,
    "verbosity": 0,
    "random_state": 0,
}


def train_xgboost(
    split: EngineSplit,
    params: dict | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
    run_name: str = "xgboost",
) -> TrainResult:
    """Train an XGBoost RUL regressor and log everything to MLflow."""
    cfg = {**XGB_DEFAULTS, **(params or {})}
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        mlflow.log_param("model_family", "xgboost")
        mlflow.log_param("n_features", len(split.feature_cols))
        mlflow.log_param("n_train_engines", len(np.unique(split.groups_train)))
        mlflow.log_param("n_val_engines", len(np.unique(split.groups_val)))

        model = xgb.XGBRegressor(**cfg)
        model.fit(
            split.X_train,
            split.y_train,
            eval_set=[(split.X_val, split.y_val)],
            verbose=False,
        )

        y_val_pred = model.predict(split.X_val)
        val_metrics = _eval(split.y_val, y_val_pred, "val")
        mlflow.log_metrics(val_metrics)

        test_metrics = None
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test[split.feature_cols])
            test_metrics = _eval(y_test, y_test_pred, "test")
            mlflow.log_metrics(test_metrics)

        signature = infer_signature(split.X_train, model.predict(split.X_train[:5]))
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=split.X_train.head(2),
        )
        return TrainResult(model=model, val_metrics=val_metrics, test_metrics=test_metrics)


# ----- LightGBM ---------------------------------------------------------------


LGBM_DEFAULTS: dict = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "verbosity": -1,
    "random_state": 0,
}


def train_lightgbm(
    split: EngineSplit,
    params: dict | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: np.ndarray | None = None,
    run_name: str = "lightgbm",
) -> TrainResult:
    """Train a LightGBM RUL regressor and log everything to MLflow."""
    cfg = {**LGBM_DEFAULTS, **(params or {})}
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(cfg)
        mlflow.log_param("model_family", "lightgbm")
        mlflow.log_param("n_features", len(split.feature_cols))
        mlflow.log_param("n_train_engines", len(np.unique(split.groups_train)))
        mlflow.log_param("n_val_engines", len(np.unique(split.groups_val)))

        model = lgb.LGBMRegressor(**cfg)
        model.fit(
            split.X_train,
            split.y_train,
            eval_set=[(split.X_val, split.y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        y_val_pred = model.predict(split.X_val)
        val_metrics = _eval(split.y_val, y_val_pred, "val")
        mlflow.log_metrics(val_metrics)

        test_metrics = None
        if X_test is not None and y_test is not None:
            y_test_pred = model.predict(X_test[split.feature_cols])
            test_metrics = _eval(y_test, y_test_pred, "test")
            mlflow.log_metrics(test_metrics)

        signature = infer_signature(split.X_train, model.predict(split.X_train[:5]))
        mlflow.lightgbm.log_model(
            model,
            name="model",
            signature=signature,
            input_example=split.X_train.head(2),
        )
        return TrainResult(model=model, val_metrics=val_metrics, test_metrics=test_metrics)
