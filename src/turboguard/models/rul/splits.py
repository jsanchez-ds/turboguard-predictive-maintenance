"""Engine-stratified splits and test-set preparation for C-MAPSS.

The cardinal sin in C-MAPSS modeling is leaking cycles of the same engine across
train/val. This module enforces engine-level separation and exposes a helper to
materialize the official test set (one row per engine at its last observed cycle,
labeled with the ground-truth RUL from ``RUL_FDxxx.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from turboguard.data.cmapss import CMAPSSData
from turboguard.features.pipeline import FeatureConfig, build_features


@dataclass
class EngineSplit:
    """Contains aligned X / y / groups for train and validation."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    groups_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    groups_val: np.ndarray
    feature_cols: list[str]


def engine_stratified_split(
    gold: pd.DataFrame,
    target_col: str = "RUL_clipped",
    val_fraction: float = 0.2,
    group_col: str = "unit_id",
    drop_cols: tuple[str, ...] = ("unit_id", "cycle", "RUL", "RUL_clipped"),
    rng_seed: int = 0,
) -> EngineSplit:
    """Hold out a fraction of *engines* (not rows) for validation."""
    if target_col not in gold.columns:
        raise KeyError(
            f"target column {target_col!r} not in gold features. "
            f"Did you call add_rul_clipped() before splitting?"
        )

    rng = np.random.default_rng(rng_seed)
    engines = np.array(sorted(gold[group_col].unique()))
    rng.shuffle(engines)
    n_val = max(1, int(round(len(engines) * val_fraction)))
    val_engines = set(engines[:n_val].tolist())

    feature_cols = [
        c for c in gold.columns if c not in drop_cols and gold[c].dtype.kind in "fi"
    ]
    train_mask = ~gold[group_col].isin(val_engines)
    val_mask = gold[group_col].isin(val_engines)

    return EngineSplit(
        X_train=gold.loc[train_mask, feature_cols].reset_index(drop=True),
        y_train=gold.loc[train_mask, target_col].to_numpy(),
        groups_train=gold.loc[train_mask, group_col].to_numpy(),
        X_val=gold.loc[val_mask, feature_cols].reset_index(drop=True),
        y_val=gold.loc[val_mask, target_col].to_numpy(),
        groups_val=gold.loc[val_mask, group_col].to_numpy(),
        feature_cols=feature_cols,
    )


def prepare_test_set(
    cmapss: CMAPSSData,
    config: FeatureConfig | None = None,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Build features for ``test_FDxxx.txt`` and select the last cycle per engine.

    Returns the per-engine feature matrix (one row per test engine), the ground-
    truth RUL vector aligned with that matrix, and the list of feature columns
    in the order they were produced. The RUL is *not* clipped — score against
    the raw NASA ground truth.
    """
    if config is None:
        config = FeatureConfig()
    test_features, _ = build_features(cmapss.test, config=config)

    # Pick the last observed cycle per engine.
    last_cycle = test_features.groupby("unit_id")["cycle"].transform("max")
    last_rows = test_features[test_features["cycle"] == last_cycle].sort_values("unit_id")

    # Align with ground-truth RUL.
    truth = cmapss.rul.set_index("unit_id")["RUL"]
    y_test = truth.loc[last_rows["unit_id"].values].to_numpy()

    feature_cols = [
        c
        for c in last_rows.columns
        if c not in {"unit_id", "cycle"} and last_rows[c].dtype.kind in "fi"
    ]
    return last_rows[feature_cols].reset_index(drop=True), y_test, feature_cols
