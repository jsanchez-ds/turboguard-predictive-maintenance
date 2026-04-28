"""LSTM autoencoder for unsupervised anomaly detection on sensor sequences.

The autoencoder is trained to *reconstruct* sliding windows of healthy
(early-life) sensor data. At inference time, reconstruction error rises sharply
when the engine drifts away from its healthy regime — providing an
unsupervised anomaly score.

Compared to Isolation Forest:
  * IF works on tabular features and captures point-wise anomalies.
  * The autoencoder operates on raw sequences and can capture *temporal*
    anomalies — e.g. a gradual drift that no individual cycle's value flags.

We keep the model intentionally small so it trains in seconds on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from turboguard.models.rul.lstm import SensorScaler, make_windows


class LSTMAutoencoder(nn.Module):
    """Encoder-decoder LSTM that reconstructs an input sequence.

    The encoder summarises the window into a fixed-size latent vector; the
    decoder repeats that vector across timesteps and emits a per-step
    reconstruction.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 32,
        latent_size: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode: take the last hidden state of the encoder.
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])  # (batch, latent)
        # Decode: feed the same latent vector across every timestep.
        seq_len = x.shape[1]
        decoder_input = self.from_latent(z).unsqueeze(1).expand(-1, seq_len, -1)
        out, _ = self.decoder(decoder_input)
        return self.out(out)


@dataclass
class AutoencoderResult:
    model: LSTMAutoencoder
    scaler: SensorScaler
    sensor_cols: list[str]
    window: int
    threshold: float
    train_loss: float


def _select_healthy_train(df: pd.DataFrame, healthy_fraction: float = 0.25) -> pd.DataFrame:
    rows = []
    for _, sub in df.groupby("unit_id", sort=False):
        max_cycle = sub["cycle"].max()
        cutoff = max(1, int(round(max_cycle * healthy_fraction)))
        rows.append(sub[sub["cycle"] <= cutoff])
    return pd.concat(rows, axis=0).reset_index(drop=True)


def train_autoencoder(
    df: pd.DataFrame,
    sensor_cols: list[str],
    *,
    window: int = 30,
    hidden_size: int = 32,
    latent_size: int = 8,
    healthy_fraction: float = 0.25,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_epochs: int = 25,
    threshold_quantile: float = 0.95,
    device: str | None = None,
    seed: int = 0,
    run_name: str = "lstm-autoencoder",
) -> AutoencoderResult:
    """Train an LSTM autoencoder on early-life cycles only."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    healthy = _select_healthy_train(df, healthy_fraction=healthy_fraction)
    scaler = SensorScaler.fit(healthy, sensor_cols)
    # Use a placeholder target column so make_windows runs uniformly — it isn't
    # used by the autoencoder (we reconstruct the input itself).
    placeholder = healthy.copy()
    placeholder["__placeholder__"] = 0.0
    X, _, _ = make_windows(
        placeholder, sensor_cols, window, target_col="__placeholder__", scaler=scaler
    )
    X_tensor = torch.from_numpy(X).float()
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(
        n_features=len(sensor_cols),
        hidden_size=hidden_size,
        latent_size=latent_size,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
        opened = True
    else:
        opened = False
    try:
        mlflow.log_params(
            {
                "model_family": "lstm_autoencoder",
                "n_features": len(sensor_cols),
                "window": window,
                "hidden_size": hidden_size,
                "latent_size": latent_size,
                "healthy_fraction": healthy_fraction,
                "lr": lr,
                "batch_size": batch_size,
                "max_epochs": max_epochs,
                "device": device,
                "seed": seed,
            }
        )
        last_loss = float("inf")
        for epoch in range(1, max_epochs + 1):
            model.train()
            running = 0.0
            for (xb,) in loader:
                xb = xb.to(device, non_blocking=True)
                optim.zero_grad()
                recon = model(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                optim.step()
                running += loss.item() * xb.size(0)
            running /= len(loader.dataset)
            mlflow.log_metric("train_mse", running, step=epoch)
            last_loss = running

        # Reconstruction error on healthy set → choose a threshold.
        model.eval()
        with torch.no_grad():
            recon = model(X_tensor.to(device)).cpu().numpy()
        mse_per_window = ((recon - X) ** 2).mean(axis=(1, 2))
        threshold = float(np.quantile(mse_per_window, threshold_quantile))
        mlflow.log_metric("anomaly_threshold", threshold)

        # Log the model with a signature.
        from mlflow.models import infer_signature
        with torch.no_grad():
            example_in = X[:2]
            example_out = model(torch.from_numpy(example_in).to(device)).cpu().numpy()
        signature = infer_signature(example_in, example_out)
        mlflow.pytorch.log_model(model, name="model", signature=signature)
    finally:
        if opened:
            mlflow.end_run()

    return AutoencoderResult(
        model=model,
        scaler=scaler,
        sensor_cols=sensor_cols,
        window=window,
        threshold=threshold,
        train_loss=last_loss,
    )


def score_autoencoder(
    result: AutoencoderResult,
    df: pd.DataFrame,
    device: str | None = None,
) -> pd.DataFrame:
    """Compute per-window reconstruction error and anomaly flags.

    Returns a DataFrame with columns ``unit_id``, ``cycle``, ``recon_mse``,
    ``is_anomaly``. The ``cycle`` reported is the *last* cycle of each window —
    we attribute the score to the cycle that completes the sliding window.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    placeholder = df.copy()
    placeholder["__placeholder__"] = 0.0
    X, _, groups = make_windows(
        placeholder,
        result.sensor_cols,
        result.window,
        target_col="__placeholder__",
        scaler=result.scaler,
    )

    result.model.eval()
    with torch.no_grad():
        recon = result.model(torch.from_numpy(X).float().to(device)).cpu().numpy()
    mse = ((recon - X) ** 2).mean(axis=(1, 2))

    # Recover the last-cycle index of each window to attach a `cycle` column.
    df_sorted = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
    last_cycle: list[int] = []
    for uid in groups:
        sub = df_sorted[df_sorted.unit_id == uid]
        last_cycle.append(int(sub["cycle"].iloc[-1]))  # placeholder, overwritten below
    # The make_windows function emits windows in order per engine; we can
    # rebuild the cycle index more precisely.
    cycles_out: list[int] = []
    for uid, count in pd.Series(groups).value_counts(sort=False).items():
        sub = df_sorted[df_sorted.unit_id == uid]
        cycles = sub["cycle"].to_numpy()
        if len(cycles) >= result.window:
            for end in range(result.window, len(cycles) + 1):
                cycles_out.append(int(cycles[end - 1]))
        else:
            cycles_out.append(int(cycles[-1]))

    out = pd.DataFrame(
        {
            "unit_id": groups.astype(int),
            "cycle": cycles_out,
            "recon_mse": mse,
            "is_anomaly": mse > result.threshold,
        }
    )
    return out
