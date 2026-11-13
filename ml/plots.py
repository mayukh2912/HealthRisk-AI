"""Static evaluation figures (ROC, PR, calibration, confusion matrix, model comparison)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAVY, TEAL, GREY = "#0B2545", "#13A89E", "#8D99AE"


def evaluation_figures(ev: Dict[str, Any], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    c, t = ev["curves"], ev["test_metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    ax.plot([p["fpr"] for p in c["roc"]], [p["tpr"] for p in c["roc"]], color=NAVY, lw=2,
            label=f"{ev['selected_label']} (AUC {t['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, label="Chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve (test)")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot([p["recall"] for p in c["pr"]], [p["precision"] for p in c["pr"]], color=TEAL, lw=2,
            label=f"AP {t['pr_auc']:.3f}")
    ax.axhline(t["prevalence"], ls="--", color=GREY, label=f"Prevalence {t['prevalence']:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
    ax.legend()
    ax = axes[2]
    ax.plot([0, 0.6], [0, 0.6], ls="--", color=GREY, label="Perfect calibration")
    ax.plot([p["mean_predicted"] for p in c["calibration"]], [p["observed_rate"] for p in c["calibration"]],
            "o-", color=NAVY, label="Platt-calibrated")
    if "calibration_uncalibrated" in c:
        ax.plot([p["mean_predicted"] for p in c["calibration_uncalibrated"]],
                [p["observed_rate"] for p in c["calibration_uncalibrated"]], "s--", color=TEAL, alpha=0.7,
                label="Uncalibrated")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed readmission rate", title="Calibration (test, deciles)")
    ax.legend()
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "evaluation_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    cm = t["confusion_matrix"]
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=13,
                    color="white" if m[i, j] > m.max() / 2 else NAVY)
    ax.set_xticks([0, 1], ["Predicted: not readmitted", "Predicted: readmitted"], fontsize=8)
    ax.set_yticks([0, 1], ["Actual: no", "Actual: yes"])
    ax.set_title(f"Confusion matrix (threshold {ev['decision_threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    comp = sorted(ev["comparison"], key=lambda r: -r["cv_pr_auc_mean"])
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(comp))
    ax.bar(x - 0.2, [r["cv_pr_auc_mean"] for r in comp], 0.4, yerr=[r["cv_pr_auc_std"] for r in comp],
           color=NAVY, label="CV PR-AUC (train folds)")
    ax.bar(x + 0.2, [r["test_pr_auc"] for r in comp], 0.4, color=TEAL, label="Test PR-AUC")
    ax.axhline(t["prevalence"], ls="--", color=GREY, label="Prevalence (no-skill)")
    ax.set_xticks(x, [r["label"] for r in comp])
    ax.set_ylabel("PR-AUC")
    ax.set_title("Model comparison")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
