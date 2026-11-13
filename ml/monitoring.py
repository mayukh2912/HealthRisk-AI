"""Phase 9 - monitoring artefacts.

* ``reference_profile``: per-feature reference distributions from the TRAINING set
  (decile bin edges for numeric features, category shares for categoricals) and the
  training-set prediction distribution. Shipped with the model so the API can
  compute PSI for any uploaded batch.
* ``baseline_drift``: PSI of (a) the held-out test set and (b) the latest 20% of
  encounters by ``encounter_id`` (a proxy for time order - the public dataset has
  no admission dates) against the training reference. These are computed, not
  simulated.

No real-world outcome labels are collected after deployment, so performance
monitoring on live data is NOT claimed; only input/prediction drift is monitored.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import config  # noqa: F401
from healthrisk_runtime.drift import drift_report, histogram
from healthrisk_runtime.features import MODEL_CATEGORICAL_FEATURES, MODEL_NUMERIC_FEATURES

PRED_EDGES = [round(x, 2) for x in np.linspace(0, 1, 21)]
MONITORED_NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_medications", "number_inpatient",
                     "number_emergency", "number_outpatient", "number_diagnoses", "age_years",
                     "total_prior_visits", "meds_per_day", "n_diabetes_meds"]
MONITORED_CATEGORICAL = ["gender", "admission_type", "discharge_disposition", "admission_source",
                         "diag_1_group", "A1Cresult", "insulin", "los_category"]


def reference_profile(X_train: pd.DataFrame, p_train: np.ndarray) -> Dict[str, Any]:
    feats: Dict[str, Any] = {}
    for f in MONITORED_NUMERIC:
        v = X_train[f].dropna().values
        edges = sorted(set(np.round(np.quantile(v, np.linspace(0.1, 0.9, 9)), 4).tolist()))
        counts = np.bincount(np.searchsorted(edges, v, side="right"), minlength=len(edges) + 1)
        labels = [f"< {edges[0]:g}"] + [f"{a:g} – {b:g}" for a, b in zip(edges[:-1], edges[1:])] + [f"≥ {edges[-1]:g}"]
        feats[f] = {"kind": "numeric", "edges": edges, "bin_labels": labels,
                    "proportions": (counts / counts.sum()).round(6).tolist()}
    for f in MONITORED_CATEGORICAL:
        vc = X_train[f].fillna("Missing").astype(str).value_counts(normalize=True)
        keep = vc[vc >= 0.005]
        feats[f] = {"kind": "categorical", "categories": keep.index.tolist(),
                    "proportions": keep.round(6).tolist() + [round(float(1 - keep.sum()), 6)]}
    return {
        "source": "training set (patient-group split)", "n": int(len(X_train)), "features": feats,
        "prediction_hist": {"edges": PRED_EDGES, "counts": histogram(p_train.tolist(), PRED_EDGES)},
        "psi_bands": {"stable": "< 0.10", "moderate": "0.10 – 0.25", "significant": "≥ 0.25"},
    }


def _frame_rows(X: pd.DataFrame) -> List[Dict[str, Any]]:
    X = X.astype(object).where(pd.notna(X), None)
    return X.to_dict("records")


def baseline_drift(profile: Dict[str, Any], X_test: pd.DataFrame, X_late: pd.DataFrame,
                   p_test: np.ndarray, p_late: np.ndarray) -> Dict[str, Any]:
    out = {}
    for name, X, p in (("held_out_test", X_test, p_test), ("latest_20pct_encounters", X_late, p_late)):
        rep = drift_report(profile, _frame_rows(X))
        out[name] = {"n": int(len(X)), "features": rep,
                     "prediction_hist": {"edges": PRED_EDGES, "counts": histogram(p.tolist(), PRED_EDGES)},
                     "mean_prediction": round(float(p.mean()), 4)}
    out["notes"] = ("'latest_20pct_encounters' uses encounter_id order as a proxy for time; the public dataset "
                    "has no dates, so this is an approximation of temporal drift. Those encounters overlap the "
                    "training data, so it demonstrates the mechanism rather than a true prospective check.")
    return out
