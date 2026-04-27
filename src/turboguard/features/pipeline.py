"""End-to-end feature engineering pipeline.

Combines:
  1. Constant-sensor pruning.
  2. Per-sensor rolling statistics (mean / std / min / max / slope) over
     multiple windows.
  3. FFT band-energy features over a single coarse window.
  4. CUSUM change-point features.

Returns a "gold" DataFrame ready to feed into a tabular model. The companion
:func:`save_gold` / :func:`load_gold` helpers persist the result as Parquet so
downstream notebooks (02_baselines, etc.) can pick it up without recomputing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .changepoint import add_changepoint_features
from .rolling import add_rolling_features
from .spectral import add_spectral_features


@dataclass
class FeatureConfig:
    """Knobs for the feature pipeline. Defaults match the C-MAPSS literature."""

    rolling_windows: tuple[int, ...] = (5, 15, 30)
    fft_window: int = 30
    cusum_window: int = 15
    cusum_threshold: float = 2.5
    constant_std_threshold: float = 1e-6
    drop_constant: bool = True
    extra_drop: tuple[str, ...] = field(default_factory=tuple)


def find_constant_sensors(
    df: pd.DataFrame,
    sensor_cols: Iterable[str],
    threshold: float = 1e-6,
    group_col: str = "unit_id",
) -> list[str]:
    """Return sensor columns whose per-engine standard deviation averages near zero."""
    stds = df.groupby(group_col)[list(sensor_cols)].std().mean()
    return stds[stds < threshold].index.tolist()


def build_features(
    df: pd.DataFrame,
    sensor_cols: Iterable[str] | None = None,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the full pipeline.

    Parameters
    ----------
    df : DataFrame
        Long-format C-MAPSS table with at least ``unit_id``, ``cycle`` and
        sensor columns. Must include an ``RUL`` column for downstream models;
        we do not enforce it here so the same pipeline can run on test data.
    sensor_cols : iterable of str, optional
        Sensor columns to use. Defaults to every ``sensor_*`` column in df.
    config : FeatureConfig, optional
        Pipeline knobs. Defaults to literature settings.

    Returns
    -------
    (gold_df, kept_sensors)
        The engineered DataFrame and the list of sensor columns that survived
        constant-pruning.
    """
    cfg = config or FeatureConfig()
    if sensor_cols is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    sensor_cols = list(sensor_cols)

    if cfg.drop_constant:
        const = find_constant_sensors(df, sensor_cols, threshold=cfg.constant_std_threshold)
        sensor_cols = [c for c in sensor_cols if c not in const and c not in cfg.extra_drop]

    out = add_rolling_features(df, sensor_cols, windows=cfg.rolling_windows)
    out = add_spectral_features(out, sensor_cols, window=cfg.fft_window)
    out = add_changepoint_features(
        out, sensor_cols, window=cfg.cusum_window, threshold=cfg.cusum_threshold
    )
    return out, sensor_cols


def save_gold(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist the gold features as Parquet (snappy compression)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression="snappy", index=False)
    return p


def load_gold(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
