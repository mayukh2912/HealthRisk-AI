"""Builds the aggregate, non-identifiable analytics document served by GET /api/analytics.

Only group-level aggregates (counts >= 15 or rates over many encounters) are
included - never row-level records.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from . import config


def build_analytics(metadata: Dict[str, Any], evaluation: Dict[str, Any], xai: Dict[str, Any],
                    drift: Dict[str, Any]) -> Dict[str, Any]:
    eda = json.loads((config.REPORTS_DIR / "eda_summary.json").read_text())
    dq = json.loads((config.REPORTS_DIR / "data_quality_report.json").read_text())
    eda.pop("figures", None)
    return {
        "generated_from": {"model_version": metadata["model_version"], "dataset_sha256": metadata["dataset"]["sha256"]},
        "dataset": {"historical": eda["overview"], "quality": dq["checks"], "cohort": dq["cohort"],
                    "note": "Historical statistics describe the 1999-2008 dataset (modelling cohort)."},
        "eda": eda,
        "model": {
            "name": metadata["model_name"], "version": metadata["model_version"], "trained_at": metadata["trained_at"],
            "decision_threshold": metadata["decision_threshold"], "risk_bands": metadata["risk_bands"],
            "selection_rule": evaluation["selection_rule"], "threshold_strategy": evaluation["threshold_strategy"],
            "test_set": evaluation["test_set"], "best_params": metadata["best_params"],
            "note": "Model metrics are computed on a held-out, patient-disjoint test set.",
        },
        "evaluation": {k: evaluation[k] for k in ("test_metrics", "test_at_0_5", "curves", "threshold_table",
                                                   "risk_bands", "subgroups", "comparison", "caveat")},
        "explainability": xai,
        "monitoring": {"baseline_drift": drift},
    }
