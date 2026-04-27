"""Sliding-window LSTM RUL regressor in PyTorch.

The literature consensus on C-MAPSS is that LSTMs use *raw* (z-scored) sensor
values rather than the engineered tabular features the gradient boosters
consume — that way the comparison between approaches is fair and the LSTM
gets to learn its own temporal aggregations.

Pipeline:
  1. Z-score each sensor based on **train** statistics only.
  2. Slide a window of length `W` over each engine; the target at every
     window is the RUL at its last cycle (clipped at 125, like the boosting
     baselines).
  3. Train with MSE loss + early stopping on the validation NASA score.
  4. Log everything to MLflow with a model signature so the run is
     registry-ready.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from torch.utils.data import DataLoader, Dataset

from .nasa_score import nasa_score


# ----- normalization ----------------------------------------------------------


@dataclass
class SensorScaler:
    """Per-column z-score using train statistics."""

    mean: np.ndarray
    std: np.ndarray
    columns: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "SensorScaler":
        sub = df[columns].to_numpy(dtype=float)
        mean = sub.mean(axis=0)
        std = sub.std(axis=0)
        std = np.where(std < 1e-9, 1.0, std)
        return cls(mean=mean, std=std, columns=list(columns))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        sub = df[self.columns].to_numpy(dtype=float)
        return (sub - self.mean) / self.std


# ----- windowing --------------------------------------------------------------


def make_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
    target_col: str = "RUL_clipped",
    group_col: str = "unit_id",
    scaler: SensorScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows per engine.

    Engines shorter than `window` cycles are *front-padded* by repeating the
    first cycle's reading — this is the standard trick (Zheng et al. 2017) and
    keeps every engine contributing at least one window.

    Returns
    -------
    (X, y, groups)
        X: (N, window, n_features) float32
        y: (N,) float32 RUL at the last cycle of each window
        groups: (N,) int unit_id of each window
    """
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    g_list: list[int] = []

    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)
    if scaler is not None:
        scaled = scaler.transform(df)
    else:
        scaled = df[sensor_cols].to_numpy(dtype=float)

    for uid, idx in df.groupby(group_col, sort=False).indices.items():
        idx = np.asarray(idx)
        engine_X = scaled[idx]
        engine_y = df.loc[idx, target_col].to_numpy() if target_col in df.columns else None
        n_cycles = engine_X.shape[0]
        for end in range(window, n_cycles + 1):
            X_list.append(engine_X[end - window : end])
            if engine_y is not None:
                y_list.append(float(engine_y[end - 1]))
            g_list.append(int(uid))
        if n_cycles < window:
            # Front-pad with the first row.
            pad = np.tile(engine_X[0], (window - n_cycles, 1))
            seq = np.vstack([pad, engine_X])
            X_list.append(seq)
            if engine_y is not None:
                y_list.append(float(engine_y[-1]))
            g_list.append(int(uid))

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32) if y_list else np.zeros((X.shape[0],), dtype=np.float32)
    groups = np.asarray(g_list, dtype=np.int64)
    return X, y, groups


def make_test_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window: int,
    scaler: SensorScaler,
    group_col: str = "unit_id",
) -> tuple[np.ndarray, np.ndarray]:
    """Build *one window per engine* ending at its last observed cycle.

    Used to score against the official ``RUL_FDxxx.txt`` ground truth.
    """
    df = df.sort_values([group_col, "cycle"]).reset_index(drop=True)
    scaled = scaler.transform(df)
    X_list: list[np.ndarray] = []
    g_list: list[int] = []
    for uid, idx in df.groupby(group_col, sort=False).indices.items():
        idx = np.asarray(idx)
        engine_X = scaled[idx]
        if engine_X.shape[0] >= window:
            X_list.append(engine_X[-window:])
        else:
            pad = np.tile(engine_X[0], (window - engine_X.shape[0], 1))
            X_list.append(np.vstack([pad, engine_X]))
        g_list.append(int(uid))
    return np.asarray(X_list, dtype=np.float32), np.asarray(g_list, dtype=np.int64)


# ----- model ------------------------------------------------------------------


class LSTMRUL(nn.Module):
    """Stacked-LSTM regressor with a small MLP head on the last hidden state."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class _ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ----- training ---------------------------------------------------------------


@dataclass
class LSTMResult:
    model: LSTMRUL
    scaler: SensorScaler
    val_metrics: dict[str, float]
    test_metrics: dict[str, float] | None = None
    best_epoch: int = -1


def _eval_pred(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    y_pred = np.clip(y_pred, 0.0, None)
    rmse = float(math.sqrt(((y_pred - y_true) ** 2).mean()))
    mae = float(np.abs(y_pred - y_true).mean())
    return {
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_nasa_score": float(nasa_score(y_true, y_pred)),
    }


def train_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    sensor_cols: list[str],
    *,
    window: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.25,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 60,
    patience: int = 8,
    target_col: str = "RUL_clipped",
    test_df: pd.DataFrame | None = None,
    test_y_true: np.ndarray | None = None,
    device: str | None = None,
    run_name: str = "lstm",
    seed: int = 0,
) -> LSTMResult:
    """End-to-end LSTM training with MLflow tracking.

    Validation uses the (un-clipped) ground truth window-target where available;
    when ``test_df`` and ``test_y_true`` are provided, the model is evaluated on
    one window per test engine and metrics are logged under the ``test_*`` prefix.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler = SensorScaler.fit(train_df, sensor_cols)
    X_tr, y_tr, _ = make_windows(train_df, sensor_cols, window, target_col, scaler=scaler)
    X_va, y_va, _ = make_windows(val_df, sensor_cols, window, target_col, scaler=scaler)

    train_loader = DataLoader(_ArrayDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(_ArrayDataset(X_va, y_va), batch_size=batch_size, shuffle=False)

    model = LSTMRUL(
        n_features=len(sensor_cols),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_score = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model_family": "lstm",
                "n_features": len(sensor_cols),
                "window": window,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "patience": patience,
                "device": device,
                "seed": seed,
            }
        )

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optim.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device)
                    val_preds.append(model(xb).cpu().numpy())
            y_val_pred = np.concatenate(val_preds)
            val_metrics = _eval_pred(y_va, y_val_pred, "val")
            mlflow.log_metrics(
                {"train_mse": train_loss, **val_metrics},
                step=epoch,
            )

            if val_metrics["val_nasa_score"] < best_val_score:
                best_val_score = val_metrics["val_nasa_score"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final val + optional test.
        model.eval()
        with torch.no_grad():
            val_preds = []
            for xb, _ in val_loader:
                val_preds.append(model(xb.to(device)).cpu().numpy())
            y_val_pred = np.concatenate(val_preds)
        val_metrics_final = _eval_pred(y_va, y_val_pred, "val")
        mlflow.log_metrics({f"final_{k}": v for k, v in val_metrics_final.items()})
        mlflow.log_metric("best_epoch", best_epoch)

        test_metrics = None
        if test_df is not None and test_y_true is not None:
            X_test, _ = make_test_windows(test_df, sensor_cols, window, scaler)
            with torch.no_grad():
                y_test_pred = model(torch.from_numpy(X_test).to(device)).cpu().numpy()
            test_metrics = _eval_pred(test_y_true, y_test_pred, "test")
            mlflow.log_metrics(test_metrics)

        # Log model with signature + a small input example.
        example_in = X_tr[:2]
        with torch.no_grad():
            example_out = model(torch.from_numpy(example_in).to(device)).cpu().numpy()
        signature = infer_signature(example_in, example_out)
        mlflow.pytorch.log_model(model, name="model", signature=signature)

    return LSTMResult(
        model=model,
        scaler=scaler,
        val_metrics=val_metrics_final,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
    )
