"""Tests for the NASA C-MAPSS scoring function."""

import numpy as np
import pytest

from turboguard.models.rul.nasa_score import nasa_score


def test_perfect_prediction_yields_zero():
    y = np.array([10, 20, 30, 40], dtype=float)
    assert nasa_score(y, y) == pytest.approx(0.0)


def test_late_prediction_penalised_more_than_early():
    # Predicting 10 cycles late should hurt more than predicting 10 cycles early.
    y_true = np.array([50.0])
    early = nasa_score(y_true, np.array([40.0]))  # d = -10
    late = nasa_score(y_true, np.array([60.0]))  # d = +10
    assert late > early
    # Sanity check on the closed-form: exp(10/10)-1 vs exp(10/13)-1.
    assert late == pytest.approx(np.exp(1.0) - 1.0, rel=1e-6)
    assert early == pytest.approx(np.exp(10 / 13.0) - 1.0, rel=1e-6)


def test_score_is_summed_across_samples():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0, 30.0])  # only second sample is wrong (10 late)
    expected = np.exp(10 / 10.0) - 1.0
    assert nasa_score(y_true, y_pred) == pytest.approx(expected, rel=1e-6)
