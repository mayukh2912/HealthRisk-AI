"""Population Stability Index (PSI) drift computation - pure Python.

The reference profile (bin edges + expected proportions) is computed from the
training set by ``ml/monitoring.py`` and shipped with the model artifact.
"""
from __future__ import annotations

import bisect
import math
from typing import Any, Dict, List, Mapping, Sequence

EPS = 1e-4
PSI_BANDS = [(0.1, "Stable"), (0.25, "Moderate shift")]


def psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, EPS), max(a, EPS)
        total += (a - e) * math.log(a / e)
    return total


def psi_label(value: float) -> str:
    for upper, label in PSI_BANDS:
        if value < upper:
            return label
    return "Significant shift"


def numeric_distribution(values: Sequence[float], edges: Sequence[float]) -> List[float]:
    """Proportions over bins defined by interior ``edges`` (len(edges)+1 bins)."""
    counts = [0] * (len(edges) + 1)
    n = 0
    for v in values:
        if v is None:
            continue
        counts[bisect.bisect_right(edges, v)] += 1
        n += 1
    return [c / n if n else 0.0 for c in counts]


def categorical_distribution(values: Sequence[Any], categories: Sequence[str]) -> List[float]:
    cats = list(categories)
    idx = {c: i for i, c in enumerate(cats)}
    counts = [0] * (len(cats) + 1)  # last slot = "other / unseen"
    for v in values:
        counts[idx.get("Missing" if v is None else str(v), len(cats))] += 1
    n = len(values)
    return [c / n if n else 0.0 for c in counts]


def drift_report(reference: Mapping[str, Any], feature_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for feat, ref in reference["features"].items():
        vals = [r.get(feat) for r in feature_rows]
        if ref["kind"] == "numeric":
            actual = numeric_distribution([None if v is None else float(v) for v in vals], ref["edges"])
            labels = ref["bin_labels"]
        else:
            actual = categorical_distribution(vals, ref["categories"])
            labels = list(ref["categories"]) + ["Other/unseen"]
        value = psi(ref["proportions"], actual)
        out.append({
            "feature": feat, "kind": ref["kind"], "psi": round(value, 4), "status": psi_label(value),
            "bins": [{"bin": b, "reference": round(e, 4), "current": round(a, 4)}
                     for b, e, a in zip(labels, ref["proportions"], actual)],
        })
    out.sort(key=lambda d: d["psi"], reverse=True)
    return out


def histogram(values: Sequence[float], edges: Sequence[float]) -> List[int]:
    counts = [0] * (len(edges) - 1)
    for v in values:
        i = bisect.bisect_right(edges, v) - 1
        counts[min(max(i, 0), len(counts) - 1)] += 1
    return counts
