"""NASA C-MAPSS scoring function.

The NASA scoring function asymmetrically penalizes late predictions (predicting
the engine is healthier than it actually is) more than early predictions —
which is what predictive maintenance actually wants.

Definition (Saxena et al. 2008):
    s_i = exp(-d_i / 13) - 1, if d_i < 0  (early)
    s_i = exp( d_i / 10) - 1, if d_i >= 0 (late)
    score = sum_i s_i

where d_i = predicted_RUL_i - true_RUL_i.

Lower is better. The benchmark target on FD001 is single-digit hundreds.
"""

from __future__ import annotations

import numpy as np


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the NASA C-MAPSS score (lower is better)."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = d < 0
    score = np.where(early, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(score.sum())
