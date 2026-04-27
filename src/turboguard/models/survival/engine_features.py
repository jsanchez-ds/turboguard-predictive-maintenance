"""Aggregate cycle-level sensor data into one row per engine for survival models.

Survival analysis works at the **subject** level: each engine is one observation
with covariates ``X``, a duration ``T``, and an event indicator ``E``. For
C-MAPSS:

* **Train data**: every engine ran to failure → ``T`` = max cycle observed,
  ``E = 1`` (uncensored).
* **Test data**: engines are observed for a partial run → ``T`` = last observed
  cycle, ``E = 0`` (right-censored). Ground-truth remaining life comes from
  ``RUL_FDxxx.txt`` and we use it only for *evaluation*, not for fitting.

The covariates aggregated per engine:

* Mean, std and linear slope of each non-constant sensor over the engine's life.
* Last-cycle reading of each sensor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _slope_per_engine(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n)
    x_mean = x.mean()
    y_mean = values.mean()
    num = ((x - x_mean) * (values - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    return float(num / den) if den != 0 else 0.0


def aggregate_engine_features(
    df: pd.DataFrame,
    sensor_cols: list[str],
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Build a one-row-per-engine DataFrame with aggregate sensor stats.

    Returns a DataFrame indexed by ``unit_id`` with one row per engine and
    columns ``{sensor}_mean``, ``{sensor}_std``, ``{sensor}_slope``,
    ``{sensor}_last``, plus a ``duration`` column equal to the engine's max
    observed cycle.
    """
    rows: list[dict[str, float]] = []
    for uid, sub in df.sort_values([group_col, "cycle"]).groupby(group_col, sort=True):
        rec: dict[str, float] = {group_col: int(uid), "duration": int(sub["cycle"].max())}
        for col in sensor_cols:
            arr = sub[col].to_numpy(dtype=float)
            rec[f"{col}_mean"] = float(arr.mean())
            rec[f"{col}_std"] = float(arr.std())
            rec[f"{col}_slope"] = _slope_per_engine(arr)
            rec[f"{col}_last"] = float(arr[-1])
        rows.append(rec)
    out = pd.DataFrame(rows).set_index(group_col)
    return out


def make_train_panel(
    df_train_cycles: pd.DataFrame,
    sensor_cols: list[str],
) -> pd.DataFrame:
    """Engine-level panel for training: every engine is uncensored (event=1)."""
    panel = aggregate_engine_features(df_train_cycles, sensor_cols)
    panel["event"] = 1
    return panel


def make_test_panel(
    df_test_cycles: pd.DataFrame,
    sensor_cols: list[str],
    rul_truth: pd.Series,
) -> pd.DataFrame:
    """Engine-level panel for test data.

    ``duration`` is the engine's last observed cycle (so the engine has
    *survived* that long without failing). The true total lifetime is
    ``duration + true_RUL`` from the ground-truth file. We attach it as a
    column so downstream code can compute prediction errors.
    """
    panel = aggregate_engine_features(df_test_cycles, sensor_cols)
    panel["event"] = 0  # right-censored: each engine still alive at last observed cycle.
    panel["true_RUL"] = rul_truth.reindex(panel.index).to_numpy()
    panel["true_total_lifetime"] = panel["duration"] + panel["true_RUL"]
    return panel
