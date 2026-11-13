import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from conftest import ROOT, needs_artifacts, needs_data
from ml.data_ingestion import build_cohort
from ml.evaluate import best_fbeta_threshold, metrics_at_threshold, threshold_free_metrics
from ml.explainability import shap_values
from ml.export import export_pipeline
from ml.feature_engineering import TARGET, engineer_frame
from ml.preprocessing import build_preprocessor
from ml.train import model_space, raw_scores, select_model
from healthrisk_runtime.engine import PortableModel, _to_f32, load_model
from healthrisk_runtime.features import EXCLUDED_COLUMNS, MODEL_FEATURES

META = {"model_version": "test", "model_name": "t", "calibration": {"a": 1.0, "b": 0.0},
        "decision_threshold": 0.5, "risk_bands": [{"name": "Low", "upper": 0.1}, {"name": "High", "upper": 1.01}]}
PARAMS = {"logistic_regression": {}, "decision_tree": {"max_depth": 5, "min_samples_leaf": 20},
          "random_forest": {"n_estimators": 15, "max_depth": 5, "class_weight": "balanced_subsample"},
          "gradient_boosting": {"n_estimators": 25, "max_depth": 3},
          "xgboost": {"n_estimators": 30, "max_depth": 4, "scale_pos_weight": 3.0}}


def test_metrics_on_known_values():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.6, 0.4, 0.9])
    m = metrics_at_threshold(y, p, 0.5)
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == m["recall"] == m["specificity"] == 0.5
    t = threshold_free_metrics(y, p)
    assert t["roc_auc"] == 0.75 and 0 <= t["brier"] <= 1


def test_threshold_respects_alert_cap():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.1; p = np.clip(y * 0.1 + rng.random(2000) * 0.3, 0, 1)
    t = best_fbeta_threshold(y.astype(int), p, beta=2, max_alert_rate=0.3)
    assert (p >= t).mean() <= 0.3


def test_selection_uses_cv_not_test():
    r = {"a": {"cv": {"pr_auc_mean": 0.30, "roc_auc_mean": 0.6, "brier_mean": 0.1}, "test": {"pr_auc": 0.1}},
         "b": {"cv": {"pr_auc_mean": 0.20, "roc_auc_mean": 0.7, "brier_mean": 0.1}, "test": {"pr_auc": 0.9}}}
    assert select_model(r) == "a"


@pytest.mark.parametrize("name", list(PARAMS))
def test_export_parity_and_exact_shap(name, synthetic_raw):
    """The dependency-free engine must reproduce sklearn/xgboost scores and shap's SHAP values."""
    cohort, _ = build_cohort(synthetic_raw)
    X = engineer_frame(cohort); y = cohort[TARGET].values
    est, _ = model_space(name, 0, 7.0); est.set_params(**PARAMS[name])
    pipe = Pipeline([("pre", build_preprocessor(20)), ("clf", est)]).fit(X, y)
    pm = PortableModel(export_pipeline(pipe, X, META))
    S = X.iloc[:60]
    recs = S.astype(object).where(pd.notna(S), None).to_dict("records")
    eng = np.array([pm.raw_score(pm.transform(r)) for r in recs])
    np.testing.assert_allclose(eng, raw_scores(pipe, S), atol=2e-6)
    if pm.kind == "tree_ensemble":
        ref, _ = shap_values(pipe, X.iloc[:200], S.iloc[:15])
        mine = []
        for r in recs[:15]:
            phi = [0.0] * pm.n_outputs
            xs = [_to_f32(v) for v in pm.transform(r)]
            for t in pm.trees:
                t.shap(xs, phi)
            mine.append(phi)
        np.testing.assert_allclose(np.array(mine), ref, atol=1e-5)
    # additivity of explanations
    ex = pm.explain(cohort.iloc[0].to_dict())
    total = sum(c["contribution"] for c in ex["all_contributions"])
    assert abs(ex["base_value"] + total - ex["output_value"]) < 1e-4


@needs_artifacts
def test_deployed_model_loads_and_predicts(valid_patient):
    m = load_model(ROOT / "backend" / "artifacts" / "model.json")
    out = m.predict(valid_patient)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["predicted_class"] in (0, 1)
    assert out["risk_category"] in {b["name"] for b in m.risk_bands}
    meta = json.loads((ROOT / "backend" / "artifacts" / "metadata.json").read_text())
    assert meta["model_version"] == m.version
    assert meta["features"] == MODEL_FEATURES
    assert not set(EXCLUDED_COLUMNS) & set(meta["features"])
    assert meta["validation"]["patient_overlap_train_test"] == 0
    assert meta["export_parity_max_abs_diff"] < 1e-5


@needs_artifacts
def test_reported_metrics_are_internally_consistent():
    ev = json.loads((ROOT / "reports" / "evaluation" / "evaluation_report.json").read_text())
    t = ev["test_metrics"]; cm = t["confusion_matrix"]
    assert cm["tp"] + cm["fn"] + cm["fp"] + cm["tn"] == ev["test_set"]["n"]
    assert abs(t["recall"] - cm["tp"] / (cm["tp"] + cm["fn"])) < 1e-9
    assert abs(t["precision"] - cm["tp"] / (cm["tp"] + cm["fp"])) < 1e-9
    assert len(ev["comparison"]) >= 3
    best = max(ev["comparison"], key=lambda r: r["cv_pr_auc_mean"])
    assert ev["selected_model"] in (best["model"],) or abs(best["cv_pr_auc_mean"] - next(
        r for r in ev["comparison"] if r["selected"])["cv_pr_auc_mean"]) <= 0.002


@needs_data
@pytest.mark.slow
def test_quick_training_run(tmp_path):
    from ml.train import run
    out = run(quick=True, out_root=tmp_path)
    vdir = out["dir"]
    assert (vdir / "model.json").exists() and (vdir / "metadata.json").exists()
    md = json.loads((vdir / "metadata.json").read_text())
    assert md["dataset"]["train_patients"] > 0 and md["export_parity_max_abs_diff"] < 1e-5
    assert set(out["results"]) == {"logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "xgboost"}
    m = load_model(vdir / "model.json")
    assert m.version.startswith("quick-")


@needs_artifacts
def test_offline_inference_matches_api_engine(tmp_path):
    from ml.inference import load_registered_model, predict_frame
    df = pd.DataFrame([{"record_id": "a", "time_in_hospital": "3", "num_lab_procedures": "40", "num_procedures": "0",
                        "num_medications": "10", "number_outpatient": "0", "number_emergency": "0",
                        "number_inpatient": "0", "number_diagnoses": "5", "gender": "Male", "age": "[50-60)",
                        "admission_type_id": "1", "discharge_disposition_id": "1", "admission_source_id": "7",
                        "diag_1": "414", "max_glu_serum": "None", "A1Cresult": "None", "change": "No",
                        "diabetesMed": "Yes", "insulin": "Steady", "metformin": "No"}])
    out = predict_frame(df, load_registered_model())
    assert out.loc[0, "errors"] == "" and 0 < out.loc[0, "predicted_probability"] < 1
