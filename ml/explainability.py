"""Phase 6 - global explainability (offline).

* SHAP values from the ``shap`` library on a held-out test sample (TreeExplainer for
  tree models, LinearExplainer for logistic regression), grouped from one-hot
  columns back to engineered features (additivity makes this exact).
* Permutation importance (drop in PR-AUC) on the test set as a model-agnostic check.

Per-patient explanations at serving time come from the dependency-free engine
(exact TreeSHAP / linear SHAP); the test-suite checks they agree with ``shap``.

SHAP describes how the *model* uses features (predictive association). It does
not identify causes of readmission and must not be read as clinical causation.
"""
from __future__ import annotations

from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, make_scorer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from . import config  # noqa: E402
from .preprocessing import output_feature_sources  # noqa: E402
from healthrisk_runtime.features import FEATURE_LABELS  # noqa: E402


def shap_values(pipe: Pipeline, X_bg: pd.DataFrame, X_explain: pd.DataFrame) -> tuple[np.ndarray, float]:
    import shap
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    Xe = pre.transform(X_explain)
    if isinstance(clf, LogisticRegression):
        Xb = pre.transform(X_bg)
        expl = shap.LinearExplainer(clf, shap.maskers.Independent(Xb, max_samples=len(Xb)))
        vals = expl.shap_values(Xe)
        base = float(np.ravel(expl.expected_value)[0])
    else:
        expl = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
        vals = expl.shap_values(Xe.astype(np.float32) if hasattr(clf, "get_booster") else Xe)
        base = expl.expected_value
        if isinstance(vals, list):  # older shap: per-class list
            vals, base = vals[1], base[1]
        vals = np.asarray(vals)
        if vals.ndim == 3:  # (n, features, classes)
            vals = vals[:, :, 1]
            base = np.ravel(base)[1]
        base = float(np.ravel(base)[0]) if np.ndim(base) else float(base)
    return np.asarray(vals), base


def group_by_source(vals: np.ndarray, sources: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(vals, columns=range(vals.shape[1]))
    grouped = {}
    for src in dict.fromkeys(sources):
        cols = [i for i, s in enumerate(sources) if s == src]
        grouped[src] = df[cols].sum(axis=1).values
    return pd.DataFrame(grouped)


def summary_plot(grouped: pd.DataFrame, X_feat: pd.DataFrame, path, top: int = 15) -> None:
    order = grouped.abs().mean().sort_values(ascending=False).index[:top][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * top + 1.5))
    rng = np.random.default_rng(0)
    for i, f in enumerate(order):
        sv = grouped[f].values
        col = X_feat[f]
        if pd.api.types.is_numeric_dtype(col):
            v = col.values.astype(float)
            lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
            c = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
        else:
            c = np.full(len(sv), 0.5)
        y = i + rng.uniform(-0.3, 0.3, len(sv))
        sc = ax.scatter(sv, y, c=c, cmap="coolwarm", s=6, alpha=0.6, vmin=0, vmax=1, linewidths=0)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_yticks(range(len(order)), [FEATURE_LABELS.get(f, f) for f in order])
    ax.set_xlabel("SHAP value (impact on model log-odds of 30-day readmission)")
    ax.set_title("SHAP summary - held-out test sample", fontweight="bold")
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"]); cb.set_label("Feature value (numeric)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def global_explanations(pipe: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_test: np.ndarray,
                        sample_size: int, random_state: int, fig_dir=None) -> Dict[str, Any]:
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_test), size=min(sample_size, len(X_test)), replace=False)
    Xs = X_test.iloc[idx]
    bg = X_train.sample(n=min(1000, len(X_train)), random_state=random_state)
    vals, base = shap_values(pipe, bg, Xs)
    sources = output_feature_sources(pipe.named_steps["pre"])
    grouped = group_by_source(vals, sources)
    fig_dir = fig_dir or config.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_path = fig_dir / "shap_summary.png"
    summary_plot(grouped, Xs.reset_index(drop=True), plot_path)

    mean_abs = grouped.abs().mean().sort_values(ascending=False)
    importance = [{"feature": f, "label": FEATURE_LABELS.get(f, f), "mean_abs_shap": round(float(v), 5)}
                  for f, v in mean_abs.items()]
    dist = []
    for f in mean_abs.index[:15]:
        sv = grouped[f].values
        q = np.percentile(sv, [5, 25, 50, 75, 95])
        dist.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "p5": float(q[0]), "p25": float(q[1]),
                     "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]),
                     "share_positive": round(float((sv > 0).mean()), 4)})
    # numeric dependence: mean SHAP by feature value (top numeric features)
    dependence = []
    for f in [f for f in mean_abs.index if pd.api.types.is_numeric_dtype(Xs[f])][:4]:
        frame = pd.DataFrame({"x": Xs[f].values, "s": grouped[f].values})
        frame["bin"] = frame["x"].clip(upper=frame["x"].quantile(0.98)).round(0 if frame["x"].nunique() < 40 else -1)
        g = frame.groupby("bin")["s"].agg(["mean", "size"]).reset_index()
        g = g[g["size"] >= 15]
        dependence.append({"feature": f, "label": FEATURE_LABELS.get(f, f),
                           "points": [{"value": float(r["bin"]), "mean_shap": round(float(r["mean"]), 4),
                                       "n": int(r["size"])} for _, r in g.iterrows()]})

    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    pi_idx = rng.choice(len(X_test), size=min(6000, len(X_test)), replace=False)
    pi = permutation_importance(pipe, X_test.iloc[pi_idx], y_test[pi_idx], scoring=scorer, n_repeats=5,
                                random_state=random_state, n_jobs=1)
    perm = sorted([{"feature": f, "label": FEATURE_LABELS.get(f, f), "importance": round(float(m), 5),
                    "std": round(float(s), 5)} for f, m, s in zip(X_test.columns, pi.importances_mean, pi.importances_std)],
                  key=lambda r: -r["importance"])
    return {
        "method": "shap.LinearExplainer" if isinstance(pipe.named_steps["clf"], LogisticRegression)
        else "shap.TreeExplainer (tree_path_dependent)",
        "sample_size": int(len(Xs)), "base_value_log_odds_uncalibrated": base,
        "mean_abs_shap": importance, "shap_distribution": dist, "dependence": dependence,
        "permutation_importance_pr_auc": perm,
        "summary_plot": "figures/shap_summary.png",
        "interpretation_note": "SHAP values describe how the model's prediction depends on each input "
                               "(predictive association learned from historical data). They are not causal "
                               "effects: changing a feature in reality will not necessarily change risk.",
    }
