"""Phase 5 - evaluation utilities for an imbalanced binary problem.

Positive class = readmitted within 30 days. Accuracy is reported but never used
for model selection.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, fbeta_score,
    log_loss, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)


def metrics_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "alert_rate": float(pred.mean()),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def threshold_free_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
        "ece": expected_calibration_error(y, p),
        "prevalence": float(y.mean()),
    }


def best_fbeta_threshold(y: np.ndarray, p: np.ndarray, beta: float = 2.0, max_alert_rate: float = 1.0) -> float:
    """Threshold maximising F-beta, restricted to thresholds that flag at most
    ``max_alert_rate`` of encounters (follow-up capacity is finite)."""
    grid = np.round(np.arange(0.02, 0.80, 0.005), 4)
    best_t, best_s = float(grid[-1]), -1.0
    for t in grid:
        pred = (p >= t).astype(int)
        if pred.mean() > max_alert_rate:
            continue
        s = fbeta_score(y, pred, beta=beta, zero_division=0)
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t


def _downsample(xs: np.ndarray, ys: np.ndarray, n: int = 200):
    if len(xs) <= n:
        return xs, ys
    idx = np.unique(np.linspace(0, len(xs) - 1, n).astype(int))
    return xs[idx], ys[idx]


def curves(y: np.ndarray, p: np.ndarray, p_uncal: np.ndarray | None = None) -> Dict[str, Any]:
    fpr, tpr, _ = roc_curve(y, p)
    fpr, tpr = _downsample(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p)
    rec, prec = _downsample(rec[::-1], prec[::-1])
    frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    out = {
        "roc": [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in zip(fpr, tpr)],
        "pr": [{"recall": round(float(a), 4), "precision": round(float(b), 4)} for a, b in zip(rec, prec)],
        "calibration": [{"mean_predicted": round(float(a), 4), "observed_rate": round(float(b), 4)}
                        for a, b in zip(mean_pred, frac)],
    }
    if p_uncal is not None:
        f2, m2 = calibration_curve(y, p_uncal, n_bins=10, strategy="quantile")
        out["calibration_uncalibrated"] = [{"mean_predicted": round(float(a), 4), "observed_rate": round(float(b), 4)}
                                           for a, b in zip(m2, f2)]
    return out


def threshold_table(y: np.ndarray, p: np.ndarray, grid: Sequence[float] | None = None) -> List[Dict[str, Any]]:
    grid = grid if grid is not None else np.round(np.arange(0.01, 0.81, 0.01), 2)
    rows = []
    for t in grid:
        m = metrics_at_threshold(y, p, float(t))
        cm = m["confusion_matrix"]
        rows.append({"threshold": float(t), "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
                     "specificity": round(m["specificity"], 4), "f1": round(m["f1"], 4),
                     "alert_rate": round(m["alert_rate"], 4), **cm})
    return rows


def risk_band_summary(y: np.ndarray, p: np.ndarray, bands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out, lo = [], 0.0
    for b in bands:
        m = (p >= lo) & (p < b["upper"])
        out.append({"band": b["name"], "lower": lo, "upper": min(b["upper"], 1.0), "count": int(m.sum()),
                    "share": round(float(m.mean()), 4),
                    "observed_rate": round(float(y[m].mean()), 4) if m.any() else None})
        lo = b["upper"]
    return out


def subgroup_metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray, threshold: float, min_n: int = 200
                     ) -> List[Dict[str, Any]]:
    rows = []
    for g in sorted(set(groups), key=str):
        m = groups == g
        if m.sum() < min_n or y[m].min() == y[m].max():
            continue
        mt = metrics_at_threshold(y[m], p[m], threshold)
        rows.append({"group": str(g), "n": int(m.sum()), "prevalence": round(float(y[m].mean()), 4),
                     "roc_auc": round(float(roc_auc_score(y[m], p[m])), 4),
                     "pr_auc": round(float(average_precision_score(y[m], p[m])), 4),
                     "recall": round(mt["recall"], 4), "precision": round(mt["precision"], 4),
                     "mean_predicted": round(float(p[m].mean()), 4)})
    return rows
