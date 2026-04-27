"""Tests for the C-MAPSS loader path resolution.

Doesn't require the actual NASA dataset — exercises path-finding behavior
against synthetic fixtures so CI passes without downloading 12 MB.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turboguard.data.cmapss import _resolve_root, load_cmapss


def _write_synthetic_cmapss(root: Path, name: str = "FD001", n_engines: int = 3) -> None:
    """Write a tiny synthetic C-MAPSS-shaped dataset to `root`."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, n_engines + 1):
        cycles = int(rng.integers(20, 30))
        for c in range(1, cycles + 1):
            row = [uid, c] + [0.5] * 3 + list(rng.normal(size=21))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(root / f"train_{name}.txt", sep=" ", header=False, index=False)
    df.to_csv(root / f"test_{name}.txt", sep=" ", header=False, index=False)
    pd.DataFrame({"RUL": [10] * n_engines}).to_csv(
        root / f"RUL_{name}.txt", sep=" ", header=False, index=False
    )


def test_resolve_root_finds_parent_directory(tmp_path: Path, monkeypatch):
    # Layout:  tmp/proj/data/raw/cmapss   (data lives here)
    #          tmp/proj/notebooks         (cwd is here)
    proj = tmp_path / "proj"
    (proj / "data" / "raw" / "cmapss").mkdir(parents=True)
    (proj / "notebooks").mkdir()
    monkeypatch.chdir(proj / "notebooks")
    resolved = _resolve_root("data/raw/cmapss")
    assert resolved == proj / "data" / "raw" / "cmapss"


def test_load_cmapss_works_from_notebooks_subdir(tmp_path: Path, monkeypatch):
    proj = tmp_path / "proj"
    cmapss = proj / "data" / "raw" / "cmapss"
    _write_synthetic_cmapss(cmapss, "FD001", n_engines=3)
    (proj / "notebooks").mkdir()
    monkeypatch.chdir(proj / "notebooks")

    d = load_cmapss("FD001")  # default relative path, must walk up.
    assert d.train["unit_id"].nunique() == 3
    assert "RUL" in d.with_rul().columns


def test_invalid_name_raises():
    with pytest.raises(ValueError, match="name must be one of"):
        load_cmapss("FD999")
