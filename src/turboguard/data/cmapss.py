"""NASA C-MAPSS turbofan engine degradation dataset loader.

The dataset has 4 sub-datasets (FD001–FD004). Each consists of:
  - train_FDxxx.txt: run-to-failure multivariate time series for N engines.
  - test_FDxxx.txt:  truncated multivariate time series for N engines (RUL > 0 at last cycle).
  - RUL_FDxxx.txt:   ground-truth Remaining Useful Life for the last cycle of each test engine.

Each row is one cycle of one engine, with columns:
  unit_id, cycle, op_setting_1..3, sensor_1..21
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

DATASETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass
class CMAPSSData:
    """Container for a single C-MAPSS sub-dataset (e.g. FD001)."""

    name: str
    train: pd.DataFrame  # run-to-failure trajectories
    test: pd.DataFrame  # truncated trajectories
    rul: pd.DataFrame  # ground-truth RUL at last test cycle (one per unit)

    def with_rul(self) -> pd.DataFrame:
        """Return training data with a per-row RUL column (RUL = max_cycle - cycle)."""
        max_cycles = self.train.groupby("unit_id")["cycle"].max().rename("max_cycle")
        df = self.train.merge(max_cycles, on="unit_id", how="left")
        df["RUL"] = df["max_cycle"] - df["cycle"]
        return df.drop(columns="max_cycle")


def _read_space_table(path: Path) -> pd.DataFrame:
    """C-MAPSS files are whitespace-delimited with no header."""
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Some files have trailing whitespace columns — drop all-NaN columns.
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMNS[: df.shape[1]]
    return df


def _resolve_root(root: str | Path) -> Path:
    """Resolve `root`, walking up from the current directory if needed.

    Lets notebooks under ``notebooks/`` and scripts under ``scripts/`` use the
    same default ``data/raw/cmapss`` path without juggling relative roots.
    """
    p = Path(root)
    if p.is_absolute() and p.exists():
        return p
    cwd = Path.cwd().resolve()
    candidates = [cwd, *cwd.parents]
    for parent in candidates:
        candidate = parent / p
        if candidate.exists():
            return candidate
    # Nothing matched — return the original (will raise FileNotFoundError on read).
    return p


def load_cmapss(name: str = "FD001", root: str | Path = "data/raw/cmapss") -> CMAPSSData:
    """Load one of the C-MAPSS sub-datasets.

    Parameters
    ----------
    name : str
        One of FD001, FD002, FD003, FD004.
    root : str or Path
        Directory containing the raw .txt files. Relative paths are resolved
        against the current directory and its parents (so notebooks under
        ``notebooks/`` can use the default).
    """
    if name not in DATASETS:
        raise ValueError(f"name must be one of {DATASETS}, got {name!r}")
    root = _resolve_root(root)
    train = _read_space_table(root / f"train_{name}.txt")
    test = _read_space_table(root / f"test_{name}.txt")
    rul = pd.read_csv(root / f"RUL_{name}.txt", sep=r"\s+", header=None, engine="python")
    rul.columns = ["RUL"]
    rul.insert(0, "unit_id", np.arange(1, len(rul) + 1))
    return CMAPSSData(name=name, train=train, test=test, rul=rul)


def add_rul_clipped(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """Add an RUL column clipped at `cap` (NASA convention: piecewise-linear RUL).

    Models trained against an unclipped RUL tend to over-react to cycles where
    the engine is far from failure. The standard fix is to clip RUL at 125
    (Heimes 2008; Babu et al. 2016).
    """
    if "RUL" not in df.columns:
        raise KeyError("Expected an 'RUL' column. Use CMAPSSData.with_rul() first.")
    out = df.copy()
    out["RUL_clipped"] = out["RUL"].clip(upper=cap)
    return out
