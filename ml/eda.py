"""Phase 2 - exploratory data analysis.

Computes every statistic from the data at run time (nothing is hard-coded),
writes static figures to ``reports/figures`` and a JSON summary consumed by the
API / dashboard (aggregates only - no row-level data).

    python -m ml.eda
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import config  # noqa: E402
from .data_ingestion import build_cohort, load_raw  # noqa: E402
from healthrisk_runtime.features import (  # noqa: E402
    AGE_BRACKETS, NUMERIC_INPUTS, TARGET, engineer_records, icd9_group,
)

NAVY, TEAL, GREY = "#0B2545", "#13A89E", "#8D99AE"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "font.size": 10})


def _rate_table(df: pd.DataFrame, col: str, order: List[Any] | None = None, min_n: int = 0) -> List[Dict[str, Any]]:
    g = df.groupby(col, observed=True)[TARGET].agg(["size", "mean"]).reset_index()
    g = g[g["size"] >= min_n]
    if order is not None:
        g[col] = pd.Categorical(g[col], categories=order, ordered=True)
        g = g.sort_values(col)
    return [{"group": str(r[col]), "count": int(r["size"]), "rate": round(float(r["mean"]), 4)} for _, r in g.iterrows()]


def _cap(s: pd.Series, cap: int) -> pd.Series:
    return s.clip(upper=cap).map(lambda v: f"{int(v)}+" if v >= cap else str(int(v)))


def _cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    k = min(table.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else 0.0


def compute_eda(df: pd.DataFrame) -> Dict[str, Any]:
    feats = pd.DataFrame(engineer_records(df.to_dict("records")))
    d = df.copy()
    d["age"] = pd.Categorical(d["age"], categories=AGE_BRACKETS, ordered=True)
    d["prior_inpatient"] = _cap(d["number_inpatient"], 5)
    d["prior_emergency"] = _cap(d["number_emergency"], 3)
    d["prior_outpatient"] = _cap(d["number_outpatient"], 3)
    d["los"] = d["time_in_hospital"].astype(int).astype(str)
    d["med_bin"] = pd.cut(d["num_medications"], [0, 5, 10, 15, 20, 25, 30, 40, 100], right=False).astype(str)
    d["lab_bin"] = pd.cut(d["num_lab_procedures"], [0, 10, 20, 30, 40, 50, 60, 70, 80, 200], right=False).astype(str)
    d["diag_1_group"] = feats["diag_1_group"].values
    d["admission_type"] = feats["admission_type"].values
    d["discharge_disposition"] = feats["discharge_disposition"].values
    d["specialty"] = d["medical_specialty"].fillna("Missing")

    pos_rate = float(d[TARGET].mean())
    numeric_cols = list(NUMERIC_INPUTS)
    desc = {c: {"mean": round(float(d[c].mean()), 3), "median": float(d[c].median()),
                "std": round(float(d[c].std()), 3), "min": float(d[c].min()), "max": float(d[c].max()),
                "q1": float(d[c].quantile(0.25)), "q3": float(d[c].quantile(0.75)),
                "missing_pct": round(float(d[c].isna().mean() * 100), 2)} for c in numeric_cols}

    # numeric vs target: Mann-Whitney U + mean by class
    numeric_tests = []
    for c in numeric_cols:
        a, b = d.loc[d[TARGET] == 1, c], d.loc[d[TARGET] == 0, c]
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        numeric_tests.append({"feature": c, "mean_readmitted": round(float(a.mean()), 3),
                              "mean_not_readmitted": round(float(b.mean()), 3),
                              "median_readmitted": float(a.median()), "median_not_readmitted": float(b.median()),
                              "rank_biserial": round(float(1 - 2 * u / (len(a) * len(b))) * -1, 4), "p_value": float(p)})
    cat_tests = []
    for c in ["age", "gender", "race", "admission_type", "discharge_disposition", "diag_1_group",
              "insulin", "A1Cresult", "max_glu_serum", "change", "diabetesMed", "metformin", "specialty"]:
        tab = pd.crosstab(d[c].astype(str).fillna("Missing"), d[TARGET])
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        cat_tests.append({"feature": c, "chi2": round(float(chi2), 2), "dof": int(dof), "p_value": float(p),
                          "cramers_v": round(_cramers_v(tab), 4)})
    cat_tests.sort(key=lambda r: r["cramers_v"], reverse=True)

    corr_cols = numeric_cols + [TARGET]
    corr = d[corr_cols].corr(method="spearman").round(3)

    age_adm = (d[d["admission_type"].isin(["Emergency", "Urgent", "Elective"])]
               .groupby(["age", "admission_type"], observed=True)[TARGET].agg(["mean", "size"]).reset_index())
    age_inp = d.groupby(["age", "prior_inpatient"], observed=True)[TARGET].mean().unstack().round(4)

    summary: Dict[str, Any] = {
        "overview": {
            "encounters": int(len(d)), "patients": int(d["patient_nbr"].nunique()),
            "positives": int(d[TARGET].sum()), "readmission_rate_30d": round(pos_rate, 4),
            "imbalance_ratio": round((1 - pos_rate) / pos_rate, 2),
            "raw_label_distribution": {k: int(v) for k, v in d["readmitted"].value_counts().items()},
        },
        "descriptive_stats": desc,
        "univariate": {
            "age": _rate_table(d, "age", AGE_BRACKETS),
            "gender": _rate_table(d, "gender"),
            "time_in_hospital": _rate_table(d, "los", [str(i) for i in range(1, 15)]),
            "num_medications": _rate_table(d, "med_bin"),
            "num_lab_procedures": _rate_table(d, "lab_bin"),
            "number_inpatient": _rate_table(d, "prior_inpatient", ["0", "1", "2", "3", "4", "5+"]),
            "number_emergency": _rate_table(d, "prior_emergency", ["0", "1", "2", "3+"]),
            "number_outpatient": _rate_table(d, "prior_outpatient", ["0", "1", "2", "3+"]),
        },
        "bivariate": {
            "insulin": _rate_table(d, "insulin", ["No", "Steady", "Up", "Down"]),
            "A1Cresult": _rate_table(d, "A1Cresult", ["None", "Norm", ">7", ">8"]),
            "max_glu_serum": _rate_table(d, "max_glu_serum", ["None", "Norm", ">200", ">300"]),
            "medical_specialty": sorted(_rate_table(d, "specialty", min_n=500), key=lambda r: -r["count"])[:12],
            "admission_type": sorted(_rate_table(d, "admission_type", min_n=100), key=lambda r: -r["count"]),
            "discharge_disposition": sorted(_rate_table(d, "discharge_disposition", min_n=300), key=lambda r: -r["rate"]),
            "diag_1_group": sorted(_rate_table(d, "diag_1_group"), key=lambda r: -r["rate"]),
            "diabetesMed": _rate_table(d, "diabetesMed"),
            "change": _rate_table(d, "change"),
        },
        "multivariate": {
            "spearman_correlation": {"columns": corr_cols, "matrix": corr.values.tolist()},
            "age_by_admission_type": [{"age": str(r["age"]), "admission_type": r["admission_type"],
                                       "rate": round(float(r["mean"]), 4), "count": int(r["size"])}
                                      for _, r in age_adm.iterrows()],
            "age_by_prior_inpatient": {"rows": [str(i) for i in age_inp.index], "columns": list(age_inp.columns),
                                       "matrix": [[None if pd.isna(v) else float(v) for v in row] for row in age_inp.values]},
        },
        "statistical_tests": {"numeric_mann_whitney": numeric_tests, "categorical_chi_square": cat_tests,
                              "caveat": "Encounters from the same patient are not independent, so p-values are "
                                        "optimistic; use effect sizes (Cramér's V, rank-biserial) for ranking."},
    }
    summary["findings"] = generate_findings(summary)
    return summary


def generate_findings(s: Dict[str, Any]) -> List[str]:
    """Plain-language findings derived from the computed summary (never hand-written numbers)."""
    o = s["overview"]
    f = [f"{o['positives']:,} of {o['encounters']:,} cohort encounters ({o['readmission_rate_30d'] * 100:.1f}%) "
         f"were readmitted within 30 days - a class imbalance of about {o['imbalance_ratio']:.1f}:1, "
         "so accuracy alone would be misleading."]
    inp = s["univariate"]["number_inpatient"]
    f.append(f"The readmission rate rises from {inp[0]['rate'] * 100:.1f}% for patients with no inpatient stay in the "
             f"prior year to {inp[-1]['rate'] * 100:.1f}% for those with {inp[-1]['group']} prior stays.")
    em = s["univariate"]["number_emergency"]
    f.append(f"Prior emergency visits show a similar gradient: {em[0]['rate'] * 100:.1f}% (none) vs "
             f"{em[-1]['rate'] * 100:.1f}% ({em[-1]['group']} visits).")
    ages = s["univariate"]["age"]
    big = [a for a in ages if a["count"] >= 1000]
    lo, hi = min(big, key=lambda a: a["rate"]), max(big, key=lambda a: a["rate"])
    f.append(f"Among age brackets with at least 1,000 encounters, the rate ranges from {lo['rate'] * 100:.1f}% "
             f"({lo['group']}) to {hi['rate'] * 100:.1f}% ({hi['group']}).")
    los = s["univariate"]["time_in_hospital"]
    f.append(f"Readmission is {los[0]['rate'] * 100:.1f}% after a 1-day stay and {los[-1]['rate'] * 100:.1f}% after "
             f"a {los[-1]['group']}-day stay.")
    dd = [r for r in s["bivariate"]["discharge_disposition"] if r["count"] >= 1000]
    if dd:
        f.append(f"Discharge destination matters: '{dd[0]['group']}' has a {dd[0]['rate'] * 100:.1f}% rate vs "
                 f"'{dd[-1]['group']}' at {dd[-1]['rate'] * 100:.1f}% (groups with ≥1,000 encounters).")
    a1c = {r["group"]: r for r in s["bivariate"]["A1Cresult"]}
    f.append(f"HbA1c was not measured in {a1c['None']['count'] / o['encounters'] * 100:.0f}% of encounters; "
             f"readmission rate is {a1c['None']['rate'] * 100:.1f}% when untested vs {a1c['>8']['rate'] * 100:.1f}% when >8%.")
    ins = {r["group"]: r for r in s["bivariate"]["insulin"]}
    f.append(f"Encounters where insulin was adjusted (Up {ins['Up']['rate'] * 100:.1f}%, Down {ins['Down']['rate'] * 100:.1f}%) "
             f"show higher rates than no insulin ({ins['No']['rate'] * 100:.1f}%).")
    top = s["statistical_tests"]["categorical_chi_square"][0]
    f.append(f"By effect size, the categorical variable most associated with readmission is '{top['feature']}' "
             f"(Cramér's V = {top['cramers_v']:.3f}); all associations are weak individually, which foreshadows "
             "modest discrimination for any model on this dataset.")
    return f


def make_figures(df: pd.DataFrame, s: Dict[str, Any]) -> List[str]:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = config.FIGURES_DIR / name
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(p.relative_to(config.ROOT)))

    u = s["univariate"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, key, title in zip(axes.ravel(),
                              ["age", "gender", "time_in_hospital", "num_medications", "num_lab_procedures",
                               "number_inpatient", "number_emergency", "number_outpatient"],
                              ["Age bracket", "Gender", "Length of stay (days)", "Medications", "Lab procedures",
                               "Prior inpatient", "Prior emergency", "Prior outpatient"]):
        rows = u[key]
        ax.bar([r["group"] for r in rows], [r["count"] for r in rows], color=NAVY)
        ax2 = ax.twinx()
        ax2.plot([r["group"] for r in rows], [r["rate"] * 100 for r in rows], color=TEAL, marker="o")
        ax2.set_ylabel("30-day readmission %", color=TEAL)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Encounters")
    save(fig, "eda_univariate.png")

    b = s["bivariate"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    for ax, key, title in zip(axes, ["insulin", "A1Cresult", "diag_1_group", "admission_type"],
                              ["Insulin status", "HbA1c result", "Primary diagnosis group", "Admission type"]):
        rows = b[key]
        ax.bar([r["group"] for r in rows], [r["rate"] * 100 for r in rows], color=TEAL)
        ax.axhline(s["overview"]["readmission_rate_30d"] * 100, color=NAVY, ls="--", lw=1, label="Overall rate")
        ax.set_title(title)
        ax.set_ylabel("30-day readmission %")
        ax.tick_params(axis="x", rotation=45)
    axes[0].legend()
    save(fig, "eda_bivariate.png")

    corr = s["multivariate"]["spearman_correlation"]
    fig, ax = plt.subplots(figsize=(8, 7))
    m = np.array(corr["matrix"])
    im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr["columns"])), corr["columns"], rotation=60, ha="right")
    ax.set_yticks(range(len(corr["columns"])), corr["columns"])
    for i in range(len(m)):
        for j in range(len(m)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(m[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation (numeric features + target)")
    save(fig, "eda_correlation_heatmap.png")

    ai = s["multivariate"]["age_by_prior_inpatient"]
    fig, ax = plt.subplots(figsize=(7, 5))
    m = np.array([[np.nan if v is None else v * 100 for v in row] for row in ai["matrix"]])
    im = ax.imshow(m, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(ai["columns"])), ai["columns"])
    ax.set_yticks(range(len(ai["rows"])), ai["rows"])
    ax.set_xlabel("Inpatient admissions in prior year")
    ax.set_ylabel("Age bracket")
    fig.colorbar(im, ax=ax, label="30-day readmission %")
    ax.set_title("Readmission rate by age and prior inpatient use")
    save(fig, "eda_age_by_prior_inpatient.png")
    return paths


def run() -> Dict[str, Any]:
    raw = load_raw()
    cohort, _ = build_cohort(raw)
    summary = compute_eda(cohort)
    summary["figures"] = make_figures(cohort, summary)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


if __name__ == "__main__":
    s = run()
    print("\n".join(f"- {x}" for x in s["findings"]))
