"""Isolation Forest anomaly detection on cycle-level engineered features.

We treat the **early life** of each engine as 'healthy' (training data) and
score every cycle by its anomaly score. As an engine approaches failure the
score should rise — a leading indicator of degradation. SHAP values explain
*which features* drive each anomaly call, which is what makes the detector
auditable in a regulated context.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from turboguard.models.rul.nasa_score import nasa_score


@dataclass
class IsolationForestResult:
    model: IsolationForest
    feature_cols: list[str]
    threshold: float
    early_warning_lead: dict[str, float] | None = None


def select_healthy_cycles(
    gold: pd.DataFrame,
    healthy_fraction: float = 0.25,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Return the first `healthy_fraction` of each engine's life.

    These rows are assumed to be free of degradation and form the IF training
    set. The default 25% mirrors common practice (e.g. Khelif et al. 2017).
    """
    out_rows: list[pd.DataFrame] = []
    for _, sub in gold.groupby(group_col, sort=False):
        max_cycle = sub["cycle"].max()
        cutoff = max(1, int(round(max_cycle * healthy_fraction)))
        out_rows.append(sub[sub["cycle"] <= cutoff])
    return pd.concat(out_rows, axis=0).reset_index(drop=True)


def train_isolation_forest(
    gold: pd.DataFrame,
    feature_cols: list[str] | None = None,
    healthy_fraction: float = 0.25,
    contamination: float = 0.05,
    n_estimators: int = 200,
    max_samples: int | str = "auto",
    threshold_quantile: float = 0.01,
    rng_seed: int = 0,
    run_name: str = "isolation-forest",
) -> IsolationForestResult:
    """Fit IsolationForest on early-life cycles and pick a low-quantile threshold.

    ``threshold_quantile`` is the quantile of healthy ``score_samples`` used as
    the alarm threshold. 0.01 (default) is conservative — flag only cycles that
    are more anomalous than the bottom 1% of healthy training cycles. Loosen to
    0.05 or 0.10 for a noisier, higher-recall alert. ``score_samples`` returns
    *higher* values for less anomalous points, so we threshold from below.
    """
    if feature_cols is None:
        feature_cols = [
            c
            for c in gold.columns
            if c not in {"unit_id", "cycle", "RUL", "RUL_clipped"}
            and gold[c].dtype.kind in "fi"
        ]

    healthy = select_healthy_cycles(gold, healthy_fraction=healthy_fraction)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=rng_seed,
        n_jobs=-1,
    )
    model.fit(healthy[feature_cols])

    # Score the healthy set; threshold = low-quantile (lower scores = more anomalous).
    healthy_scores = model.score_samples(healthy[feature_cols])
    threshold = float(np.quantile(healthy_scores, threshold_quantile))

    if mlflow.active_run() is None:
        mlflow.start_run(run_name=run_name)
        opened = True
    else:
        opened = False
    try:
        mlflow.log_params(
            {
                "model_family": "isolation_forest",
                "n_estimators": n_estimators,
                "contamination": contamination,
                "healthy_fraction": healthy_fraction,
                "n_features": len(feature_cols),
                "n_train_rows": len(healthy),
            }
        )
        mlflow.log_metric("anomaly_threshold", threshold)
    finally:
        if opened:
            mlflow.end_run()

    return IsolationForestResult(model=model, feature_cols=feature_cols, threshold=threshold)


def score_engines(
    result: IsolationForestResult,
    gold: pd.DataFrame,
    group_col: str = "unit_id",
) -> pd.DataFrame:
    """Return cycle-level anomaly scores plus a boolean flag.

    Lower score → more anomalous. ``is_anomaly = score < threshold``.
    """
    scored = gold[[group_col, "cycle"]].copy()
    scored["anomaly_score"] = result.model.score_samples(gold[result.feature_cols])
    scored["is_anomaly"] = scored["anomaly_score"] < result.threshold
    return scored


def early_warning_lead(
    scored: pd.DataFrame,
    gold: pd.DataFrame,
    group_col: str = "unit_id",
    consecutive: int = 3,
) -> pd.DataFrame:
    """For each engine, find the first cycle of ``consecutive`` consecutive anomalies.

    Returns one row per engine with:
      - first_alarm_cycle: first cycle of a `consecutive`-run anomaly streak
      - failure_cycle: engine's last observed cycle (proxy for failure)
      - lead_time: failure_cycle - first_alarm_cycle (positive = early warning,
        negative = the alarm fired after the engine already failed)
    """
    rows = []
    df = scored.merge(gold[[group_col, "cycle"]], on=[group_col, "cycle"])
    for uid, sub in df.sort_values([group_col, "cycle"]).groupby(group_col, sort=False):
        flags = sub["is_anomaly"].to_numpy()
        cycles = sub["cycle"].to_numpy()
        first_alarm = None
        # Sliding window of `consecutive` flags.
        if len(flags) >= consecutive:
            for i in range(len(flags) - consecutive + 1):
                if flags[i : i + consecutive].all():
                    first_alarm = int(cycles[i])
                    break
        failure = int(cycles[-1])
        rows.append(
            {
                "unit_id": int(uid),
                "first_alarm_cycle": first_alarm,
                "failure_cycle": failure,
                "lead_time": (failure - first_alarm) if first_alarm is not None else None,
            }
        )
    return pd.DataFrame(rows)


def shap_top_drivers(
    result: IsolationForestResult,
    gold_subset: pd.DataFrame,
    sample_size: int = 200,
    top_k: int = 15,
) -> pd.DataFrame:
    """Approximate SHAP values for IF anomaly scores via a small sample.

    SHAP for IsolationForest uses TreeExplainer; we subsample because the
    explanation is expensive on tens of thousands of rows. Returns a DataFrame
    with the top-k features by mean(|SHAP|) — these are the global drivers
    of the anomaly score.
    """
    import shap

    if len(gold_subset) > sample_size:
        gold_subset = gold_subset.sample(n=sample_size, random_state=0)
    explainer = shap.TreeExplainer(result.model)
    shap_values = explainer.shap_values(gold_subset[result.feature_cols])
    abs_mean = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": result.feature_cols, "mean_abs_shap": abs_mean})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
