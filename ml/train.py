"""Phases 4-6 + 9 - leakage-safe training, model comparison, evaluation,
explainability, export and registration.

    python -m ml.train            # full run (≈20-40 min on 2 CPUs)
    python -m ml.train --quick    # smoke run on a patient-grouped subsample (tests / CI)

Outputs
    models/<version>/model.json, pipeline.joblib, metadata.json, evaluation.json
    models/registry.json                          (all versions)
    models/metadata/feature_schema.json
    reports/evaluation/*.json|md, reports/figures/*.png
    backend/artifacts/*                           (promoted runtime artefacts)
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config
from .data_ingestion import build_cohort, file_sha256, load_raw
from .evaluate import (
    best_fbeta_threshold, curves, metrics_at_threshold, risk_band_summary, subgroup_metrics,
    threshold_free_metrics, threshold_table,
)
from .explainability import global_explanations
from .export import export_pipeline
from .feature_engineering import (
    EXCLUDED_COLUMNS, FEATURE_DESCRIPTIONS, MODEL_CATEGORICAL_FEATURES, MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES, TARGET, engineer_frame,
)
from .monitoring import baseline_drift, reference_profile
from .preprocessing import build_preprocessor, group_cv, patient_group_split
from healthrisk_runtime.engine import PortableModel
from healthrisk_runtime.features import CATEGORICAL_INPUTS, NUMERIC_INPUTS, OPTIONAL_INPUTS, REQUIRED_INPUTS

MODEL_LABELS = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
                "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
                "xgboost": "XGBoost"}


def model_space(name: str, rs: int, pos_ratio: float) -> Tuple[Any, Dict[str, Any]]:
    if name == "logistic_regression":
        return (LogisticRegression(max_iter=3000, random_state=rs),
                {"clf__C": loguniform(1e-3, 3), "clf__class_weight": [None, "balanced"]})
    if name == "decision_tree":
        return (DecisionTreeClassifier(random_state=rs),
                {"clf__max_depth": [3, 4, 5, 6, 8, 10], "clf__min_samples_leaf": [20, 50, 100, 200, 400],
                 "clf__criterion": ["gini", "entropy"], "clf__class_weight": [None, "balanced"]})
    if name == "random_forest":
        return (RandomForestClassifier(random_state=rs, n_jobs=-1),
                {"clf__n_estimators": [150, 200], "clf__max_depth": [6, 8, 10],
                 "clf__min_samples_leaf": [20, 50, 100], "clf__max_features": ["sqrt", 0.3],
                 "clf__class_weight": [None, "balanced_subsample"]})
    if name == "gradient_boosting":
        return (GradientBoostingClassifier(random_state=rs, subsample=0.7),
                {"clf__n_estimators": [150, 250], "clf__learning_rate": [0.05, 0.1],
                 "clf__max_depth": [2, 3, 4], "clf__min_samples_leaf": [50, 100]})
    if name == "xgboost":
        return (XGBClassifier(random_state=rs, n_jobs=-1, tree_method="hist", eval_metric="logloss"),
                {"clf__n_estimators": [200, 300, 400, 600], "clf__learning_rate": [0.02, 0.05, 0.1],
                 "clf__max_depth": [3, 4, 5], "clf__min_child_weight": [1, 5, 10],
                 "clf__subsample": [0.7, 0.9], "clf__colsample_bytree": [0.5, 0.8],
                 "clf__reg_lambda": [1.0, 5.0],
                 "clf__scale_pos_weight": [1.0, round(float(np.sqrt(pos_ratio)), 2), round(pos_ratio, 2)]})
    raise KeyError(name)


def score_oof(pipe: Pipeline, X, y, groups, cv) -> np.ndarray:
    """Out-of-fold uncalibrated scores (margin for linear/boosting, probability for bagged trees)."""
    clf = pipe.named_steps["clf"]
    method = "predict_proba" if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier)) else "decision_function"
    if isinstance(clf, XGBClassifier):
        method = "predict_proba"
    s = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method=method, n_jobs=1)
    if method == "predict_proba":
        s = s[:, 1]
        if isinstance(clf, XGBClassifier):  # convert to margin
            s = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7)))
    return s


def raw_scores(pipe: Pipeline, X) -> np.ndarray:
    clf = pipe.named_steps["clf"]
    if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier)):
        return pipe.predict_proba(X)[:, 1]
    if isinstance(clf, XGBClassifier):
        return clf.predict(pipe.named_steps["pre"].transform(X), output_margin=True)
    return pipe.decision_function(X)


def platt_fit(scores: np.ndarray, y: np.ndarray, prob_space: bool) -> Dict[str, float]:
    s = _to_logit(scores) if prob_space else scores
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(s.reshape(-1, 1), y)
    return {"type": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0]),
            "fitted_on": "out-of-fold training predictions (patient-group CV)"}


def _to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_apply(scores: np.ndarray, cal: Dict[str, float], prob_space: bool) -> np.ndarray:
    s = _to_logit(scores) if prob_space else scores
    return 1 / (1 + np.exp(-(cal["a"] * s + cal["b"])))


def is_prob_space(pipe: Pipeline) -> bool:
    return isinstance(pipe.named_steps["clf"], (RandomForestClassifier, DecisionTreeClassifier))


def _jsonable(o):
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def select_model(results: Dict[str, Dict[str, Any]]) -> str:
    ranked = sorted(results, key=lambda k: -results[k]["cv"]["pr_auc_mean"])
    best = ranked[0]
    for cand in ranked[1:]:
        a, b = results[best]["cv"], results[cand]["cv"]
        if abs(a["pr_auc_mean"] - b["pr_auc_mean"]) <= 0.002:
            if (b["roc_auc_mean"], -b["brier_mean"]) > (a["roc_auc_mean"], -a["brier_mean"]):
                best = cand
    return best


def run(quick: bool = False, out_root=None) -> Dict[str, Any]:
    t0 = time.time()
    cfg = config.load_config()
    rs = cfg["random_state"]
    started = datetime.now(timezone.utc)

    raw = load_raw()
    cohort, cohort_log = build_cohort(raw)
    if quick:  # patient-grouped subsample for fast CI runs
        keep = pd.Series(cohort["patient_nbr"].unique()).sample(frac=0.15, random_state=rs)
        cohort = cohort[cohort["patient_nbr"].isin(keep)].reset_index(drop=True)
    X = engineer_frame(cohort)
    y = cohort[TARGET].values
    groups = cohort["patient_nbr"].values
    assert not set(EXCLUDED_COLUMNS) & set(X.columns), "excluded column leaked into features"

    tr, te = patient_group_split(pd.Series(y), pd.Series(groups), cfg["test_size_folds"], rs)
    assert not set(groups[tr]) & set(groups[te]), "patient overlap between train and test"
    X_tr, X_te, y_tr, y_te, g_tr = X.iloc[tr], X.iloc[te], y[tr], y[te], groups[tr]
    pos_ratio = float((y_tr == 0).sum() / (y_tr == 1).sum())
    cv = group_cv(cfg["cv_folds"], rs)
    print(f"[split] train={len(tr):,} ({len(set(g_tr)):,} patients)  test={len(te):,} "
          f"({len(set(groups[te])):,} patients)  prevalence train={y_tr.mean():.4f} test={y_te.mean():.4f}")

    results: Dict[str, Dict[str, Any]] = {}
    fitted: Dict[str, Pipeline] = {}
    for name, mcfg in cfg["models"].items():
        ts = time.time()
        est, space = model_space(name, rs, pos_ratio)
        pipe = Pipeline([("pre", build_preprocessor(cfg["one_hot_min_frequency"])), ("clf", est)])
        search = RandomizedSearchCV(
            pipe, space, n_iter=2 if quick else mcfg["n_iter"], cv=cv, random_state=rs, n_jobs=1,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc", "brier": "neg_brier_score"},
            refit="pr_auc", return_train_score=True, error_score="raise")
        search.fit(X_tr, y_tr, groups=g_tr)
        i = search.best_index_
        cvr = search.cv_results_
        best = search.best_estimator_
        prob = is_prob_space(best)
        oof = score_oof(clone(best), X_tr, y_tr, g_tr, cv)
        cal = platt_fit(oof, y_tr, prob)
        p_oof = platt_apply(oof, cal, prob)
        thr = best_fbeta_threshold(y_tr, p_oof, beta=cfg["threshold_beta"], max_alert_rate=cfg["max_alert_rate"])
        p_te_raw = raw_scores(best, X_te)
        p_te = platt_apply(p_te_raw, cal, prob)
        p_tr = platt_apply(raw_scores(best, X_tr), cal, prob)
        results[name] = {
            "label": MODEL_LABELS[name],
            "best_params": {k.replace("clf__", ""): _jsonable(v) for k, v in search.best_params_.items()},
            "cv": {"pr_auc_mean": float(cvr["mean_test_pr_auc"][i]), "pr_auc_std": float(cvr["std_test_pr_auc"][i]),
                   "roc_auc_mean": float(cvr["mean_test_roc_auc"][i]), "roc_auc_std": float(cvr["std_test_roc_auc"][i]),
                   "brier_mean": float(-cvr["mean_test_brier"][i]),
                   "train_pr_auc_mean": float(cvr["mean_train_pr_auc"][i]),
                   "n_candidates": int(len(cvr["params"]))},
            "oof_calibrated": threshold_free_metrics(y_tr, p_oof),
            "calibration": cal, "threshold": thr,
            "train": threshold_free_metrics(y_tr, p_tr),
            "test": {**threshold_free_metrics(y_te, p_te), "uncalibrated_brier": None,
                     **{k: v for k, v in metrics_at_threshold(y_te, p_te, thr).items()}},
            "test_at_0_5": metrics_at_threshold(y_te, p_te, 0.5),
            "fit_seconds": round(time.time() - ts, 1),
        }
        if not prob:
            results[name]["test"]["uncalibrated_brier"] = float(np.mean((1 / (1 + np.exp(-p_te_raw)) - y_te) ** 2))
        else:
            results[name]["test"]["uncalibrated_brier"] = float(np.mean((p_te_raw - y_te) ** 2))
        fitted[name] = best
        r = results[name]
        print(f"[{name}] cv PR-AUC={r['cv']['pr_auc_mean']:.4f}±{r['cv']['pr_auc_std']:.4f} "
              f"ROC-AUC={r['cv']['roc_auc_mean']:.4f} | test PR-AUC={r['test']['pr_auc']:.4f} "
              f"ROC-AUC={r['test']['roc_auc']:.4f} recall@{thr}={r['test']['recall']:.3f} ({r['fit_seconds']}s)")

    selected = select_model(results)
    best_pipe = fitted[selected]
    sel = results[selected]
    prob = is_prob_space(best_pipe)
    p_te_raw = raw_scores(best_pipe, X_te)
    p_te = platt_apply(p_te_raw, sel["calibration"], prob)
    p_tr = platt_apply(raw_scores(best_pipe, X_tr), sel["calibration"], prob)

    # ---------------- versioning -------------------------------------------------
    registry = json.loads(config.REGISTRY.read_text()) if config.REGISTRY.exists() else {"versions": []}
    n_prev = len(registry["versions"])
    version = (f"quick-{started:%Y%m%d%H%M%S}" if quick else f"{cfg['model_version_prefix']}.{n_prev}")
    vdir = (out_root or config.MODELS_DIR) / version
    fig_dir = vdir / "figures" if quick else config.FIGURES_DIR
    eval_dir = vdir / "reports" if quick else config.EVAL_DIR
    data_hash = file_sha256(config.RAW_CSV)
    cfg_hash = __import__("hashlib").sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

    # ---------------- detailed evaluation of the selected model -----------------
    race = cohort["race"].fillna("Missing").values[te]
    ages = cohort["age"].values[te]
    gender = cohort["gender"].values[te]
    evaluation = {
        "model_version": version, "selected_model": selected, "selected_label": sel["label"],
        "selection_rule": cfg["selection_rule"],
        "threshold_strategy": f"Threshold maximising F{cfg['threshold_beta']:g} (recall-weighted) on calibrated "
                              f"out-of-fold training predictions, restricted to flagging at most "
                              f"{cfg['max_alert_rate'] * 100:.0f}% of encounters; never tuned on the test set.",
        "decision_threshold": sel["threshold"],
        "test_metrics": sel["test"], "train_metrics": sel["train"], "test_at_0_5": sel["test_at_0_5"],
        "curves": curves(y_te, p_te, p_uncal=(p_te_raw if prob else 1 / (1 + np.exp(-p_te_raw)))),
        "threshold_table": threshold_table(y_te, p_te),
        "risk_bands": risk_band_summary(y_te, p_te, cfg["risk_bands"]),
        "subgroups": {"gender": subgroup_metrics(y_te, p_te, gender, sel["threshold"]),
                      "age": subgroup_metrics(y_te, p_te, ages, sel["threshold"]),
                      "race": subgroup_metrics(y_te, p_te, race, sel["threshold"])},
        "comparison": [{"model": k, "label": v["label"], **{f"cv_{m}": v["cv"][m] for m in
                        ("pr_auc_mean", "pr_auc_std", "roc_auc_mean", "brier_mean", "train_pr_auc_mean")},
                        **{f"test_{m}": v["test"][m] for m in ("pr_auc", "roc_auc", "brier", "ece", "precision",
                                                                "recall", "specificity", "f1", "accuracy")},
                        "threshold": v["threshold"], "best_params": v["best_params"],
                        "fit_seconds": v["fit_seconds"], "selected": k == selected}
                       for k, v in results.items()],
        "test_set": {"n": int(len(te)), "patients": int(len(set(groups[te]))), "prevalence": float(y_te.mean())},
        "caveat": "Retrospective, single-dataset evaluation (1999-2008 US hospitals). These metrics do not "
                  "establish clinical usefulness or performance in any other population.",
    }
    print(f"[select] {selected}  (test PR-AUC={sel['test']['pr_auc']:.4f}, ROC-AUC={sel['test']['roc_auc']:.4f})")

    # ---------------- explainability ---------------------------------------------
    xai = global_explanations(best_pipe, X_tr, X_te, y_te, 400 if quick else cfg["shap_sample_size"], rs,
                              fig_dir=fig_dir)

    # ---------------- export portable model + parity check ------------------------
    meta = {
        "model_version": version, "model_name": sel["label"], "model_key": selected,
        "trained_at": started.isoformat(timespec="seconds"),
        "calibration": sel["calibration"], "decision_threshold": sel["threshold"],
        "risk_bands": cfg["risk_bands"],
    }
    spec = export_pipeline(best_pipe, X_tr, meta)
    pm = PortableModel(spec)
    sample = X_te.sample(n=min(500, len(X_te)), random_state=rs)
    eng = np.array([pm._calibrate(pm.raw_score(pm.transform(r))) for r in
                    sample.astype(object).where(pd.notna(sample), None).to_dict("records")])
    ref = platt_apply(raw_scores(best_pipe, sample), sel["calibration"], prob)
    parity = float(np.max(np.abs(eng - ref)))
    print(f"[export] max |engine - sklearn| probability difference on 500 test rows = {parity:.2e}")
    assert parity < 1e-5, "exported model does not reproduce the pipeline"

    # ---------------- monitoring reference --------------------------------------
    profile = reference_profile(X_tr, p_tr)
    late_mask = cohort["encounter_id"].rank(pct=True).values > 0.8
    X_late = X[late_mask]
    p_late = platt_apply(raw_scores(best_pipe, X_late), sel["calibration"], prob)
    drift = baseline_drift(profile, X_te, X_late, p_te, p_late)

    # ---------------- metadata / schema / registry -----------------------------
    feature_schema = {
        "prediction_point": "At hospital discharge (index encounter).",
        "target": {"name": TARGET, "definition": "1 if raw 'readmitted' == '<30' (readmitted within 30 days); "
                                                  "0 if '>30' or 'NO'."},
        "required_inputs": REQUIRED_INPUTS, "optional_inputs": OPTIONAL_INPUTS,
        "numeric_inputs": NUMERIC_INPUTS,
        "categorical_inputs": {k: v for k, v in CATEGORICAL_INPUTS.items()},
        "model_features": {"numeric": MODEL_NUMERIC_FEATURES, "categorical": MODEL_CATEGORICAL_FEATURES},
        "feature_descriptions": FEATURE_DESCRIPTIONS, "excluded_columns": EXCLUDED_COLUMNS,
        "transformed_feature_count": spec["preprocessor"]["n_outputs"],
    }
    metadata = {
        "model_version": version, "model_name": sel["label"], "model_key": selected,
        "trained_at": meta["trained_at"], "training_seconds": round(time.time() - t0, 1), "quick_run": quick,
        "dataset": {"file": config.RAW_CSV.name, "sha256": data_hash, "raw_rows": cohort_log["raw_rows"],
                    "cohort": cohort_log, "train_rows": int(len(tr)), "test_rows": int(len(te)),
                    "train_patients": int(len(set(g_tr))), "test_patients": int(len(set(groups[te])))},
        "config": cfg, "config_sha256": cfg_hash,
        "validation": {"split": f"StratifiedGroupKFold(n_splits={cfg['test_size_folds']}) first fold held out by patient_nbr",
                       "cv": f"StratifiedGroupKFold(n_splits={cfg['cv_folds']}) grouped by patient_nbr",
                       "patient_overlap_train_test": 0},
        "best_params": sel["best_params"], "decision_threshold": sel["threshold"],
        "calibration": sel["calibration"], "risk_bands": cfg["risk_bands"],
        "metrics": {"cv": sel["cv"], "test": sel["test"]},
        "features": MODEL_FEATURES, "export_parity_max_abs_diff": parity,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "xgboost": __import__("xgboost").__version__, "pandas": pd.__version__,
                        "numpy": np.__version__},
    }
    vdir.mkdir(parents=True, exist_ok=True)
    dump = lambda p, o: p.write_text(json.dumps(o, indent=1, default=_jsonable))  # noqa: E731
    (vdir / "model.json").write_text(json.dumps(spec, separators=(",", ":"), default=_jsonable))
    joblib.dump(best_pipe, vdir / "pipeline.joblib", compress=3)
    dump(vdir / "metadata.json", metadata)
    dump(vdir / "evaluation.json", evaluation)
    dump(vdir / "feature_schema.json", feature_schema)
    if not quick:
        config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
        dump(config.METADATA_DIR / "feature_schema.json", feature_schema)
        dump(config.METADATA_DIR / f"model_metadata_{version}.json", metadata)
    registry["versions"] = [v for v in registry["versions"] if v["model_version"] != version]
    registry["versions"].append({"model_version": version, "model_name": sel["label"], "trained_at": meta["trained_at"],
                                 "dataset_sha256": data_hash, "config_sha256": cfg_hash, "quick": quick,
                                 "test_pr_auc": sel["test"]["pr_auc"], "test_roc_auc": sel["test"]["roc_auc"],
                                 "test_recall": sel["test"]["recall"], "test_brier": sel["test"]["brier"],
                                 "decision_threshold": sel["threshold"], "artifact": f"models/{version}/model.json"})
    if not quick:  # quick smoke runs are never registered
        registry["production"] = version
        dump(config.REGISTRY, registry)

    eval_dir.mkdir(parents=True, exist_ok=True)
    dump(eval_dir / "evaluation_report.json", evaluation)
    dump(eval_dir / "model_comparison.json", evaluation["comparison"])
    dump(eval_dir / "explainability.json", xai)
    dump(eval_dir / "drift_baseline.json", drift)
    (eval_dir / "evaluation_report.md").write_text(render_report(evaluation, metadata, xai))
    from .plots import evaluation_figures
    evaluation_figures(evaluation, fig_dir)

    if not quick:
        promote(version, spec, metadata, evaluation, xai, profile, drift, feature_schema, registry)
    print(f"[done] version {version} in {time.time() - t0:.0f}s")
    return {"version": version, "selected": selected, "results": results, "metadata": metadata, "dir": vdir}


def promote(version, spec, metadata, evaluation, xai, profile, drift, schema, registry) -> None:
    """Copy runtime artefacts into backend/artifacts (what the serverless API ships)."""
    from .analytics import build_analytics
    out = config.BACKEND_ARTIFACTS
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(spec, separators=(",", ":"), default=_jsonable))
    w = lambda n, o: (out / n).write_text(json.dumps(o, separators=(",", ":"), default=_jsonable))  # noqa: E731
    w("metadata.json", metadata)
    w("feature_schema.json", schema)
    w("reference_profile.json", profile)
    w("registry.json", registry)
    w("analytics.json", build_analytics(metadata, evaluation, xai, drift))
    public = config.ROOT / "frontend" / "public" / "figures"
    public.mkdir(parents=True, exist_ok=True)
    for f in config.FIGURES_DIR.glob("*.png"):
        shutil.copy2(f, public / f.name)


def render_report(ev: Dict[str, Any], md: Dict[str, Any], xai: Dict[str, Any]) -> str:
    t = ev["test_metrics"]
    cm = t["confusion_matrix"]
    rows = "\n".join(
        f"| {c['label']}{' ✅' if c['selected'] else ''} | {c['cv_pr_auc_mean']:.4f} ± {c['cv_pr_auc_std']:.4f} | "
        f"{c['cv_roc_auc_mean']:.4f} | {c['test_pr_auc']:.4f} | {c['test_roc_auc']:.4f} | {c['test_brier']:.4f} | "
        f"{c['threshold']:.3f} | {c['test_recall']:.3f} | {c['test_precision']:.3f} | {c['test_f1']:.3f} |"
        for c in sorted(ev["comparison"], key=lambda c: -c["cv_pr_auc_mean"]))
    bands = "\n".join(f"| {b['band']} | {b['lower']:.2f}–{b['upper']:.2f} | {b['count']:,} | {b['share'] * 100:.1f}% | "
                      f"{(b['observed_rate'] or 0) * 100:.1f}% |" for b in ev["risk_bands"])
    top = "\n".join(f"{i + 1}. {r['label']} — mean |SHAP| {r['mean_abs_shap']:.4f}"
                    for i, r in enumerate(xai["mean_abs_shap"][:10]))
    return f"""# Model evaluation report — v{ev['model_version']}

*Generated automatically by `python -m ml.train`. Every number below is computed from the data.*

- Dataset sha256: `{md['dataset']['sha256'][:16]}…`; cohort {md['dataset']['cohort']['cohort_rows']:,} encounters
- Train: {md['dataset']['train_rows']:,} encounters / {md['dataset']['train_patients']:,} patients;
  Test: {md['dataset']['test_rows']:,} encounters / {md['dataset']['test_patients']:,} patients (no patient overlap)
- Test prevalence: {ev['test_set']['prevalence'] * 100:.2f}%

## Model comparison

Selection rule: {ev['selection_rule']}

| Model | CV PR-AUC | CV ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier (calibrated) | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
{rows}

Thresholds: {ev['threshold_strategy']}

## Selected model: {ev['selected_label']}

| Metric (test, threshold {ev['decision_threshold']:.3f}) | Value |
|---|---|
| ROC-AUC | {t['roc_auc']:.4f} |
| PR-AUC (average precision) | {t['pr_auc']:.4f} (prevalence baseline {t['prevalence']:.4f}) |
| Recall / sensitivity | {t['recall']:.4f} |
| Precision (PPV) | {t['precision']:.4f} |
| Specificity | {t['specificity']:.4f} |
| F1 | {t['f1']:.4f} |
| Accuracy | {t['accuracy']:.4f} |
| Brier score | {t['brier']:.4f} (uncalibrated {t['uncalibrated_brier']:.4f}) |
| Expected calibration error | {t['ece']:.4f} |
| Alert rate | {t['alert_rate'] * 100:.1f}% of encounters flagged |

Confusion matrix: TP {cm['tp']:,} · FP {cm['fp']:,} · FN {cm['fn']:,} · TN {cm['tn']:,}

## Risk bands (test set)

| Band | Probability range | Encounters | Share | Observed 30-day readmission |
|---|---|---|---|---|
{bands}

## Global explanation (top 10 by mean |SHAP|, {xai['method']})

{top}

{xai['interpretation_note']}

## Caveat

{ev['caveat']}
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run on a 15% patient subsample")
    args = ap.parse_args(argv)
    run(quick=args.quick)


if __name__ == "__main__":
    main()
