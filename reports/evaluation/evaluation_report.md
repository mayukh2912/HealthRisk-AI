# Model evaluation report — v1.0.0

*Generated automatically by `python -m ml.train`. Every number below is computed from the data.*

- Dataset sha256: `0689e7ec031237dc…`; cohort 99,340 encounters
- Train: 79,473 encounters / 56,036 patients;
  Test: 19,867 encounters / 13,951 patients (no patient overlap)
- Test prevalence: 11.39%

## Model comparison

Selection rule: Highest mean group-aware CV PR-AUC; ties (within 0.002) broken by higher ROC-AUC then lower Brier score. The test set is never used for selection.

| Model | CV PR-AUC | CV ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Brier (calibrated) | Threshold | Recall | Precision | F1 |
|---|---|---|---|---|---|---|---|---|---|
| Gradient Boosting ✅ | 0.2265 ± 0.0071 | 0.6723 | 0.2310 | 0.6662 | 0.0960 | 0.110 | 0.618 | 0.177 | 0.275 |
| XGBoost | 0.2251 ± 0.0058 | 0.6713 | 0.2268 | 0.6651 | 0.0962 | 0.115 | 0.594 | 0.182 | 0.279 |
| Random Forest | 0.2220 ± 0.0060 | 0.6689 | 0.2225 | 0.6607 | 0.0965 | 0.115 | 0.597 | 0.177 | 0.272 |
| Logistic Regression | 0.2119 ± 0.0083 | 0.6634 | 0.2192 | 0.6610 | 0.0966 | 0.110 | 0.593 | 0.177 | 0.272 |
| Decision Tree | 0.1999 ± 0.0047 | 0.6517 | 0.2018 | 0.6462 | 0.0973 | 0.115 | 0.574 | 0.173 | 0.265 |

Thresholds: Threshold maximising F2 (recall-weighted) on calibrated out-of-fold training predictions, restricted to flagging at most 40% of encounters; never tuned on the test set.

## Selected model: Gradient Boosting

| Metric (test, threshold 0.110) | Value |
|---|---|
| ROC-AUC | 0.6662 |
| PR-AUC (average precision) | 0.2310 (prevalence baseline 0.1139) |
| Recall / sensitivity | 0.6182 |
| Precision (PPV) | 0.1772 |
| Specificity | 0.6309 |
| F1 | 0.2754 |
| Accuracy | 0.6295 |
| Brier score | 0.0960 (uncalibrated 0.0960) |
| Expected calibration error | 0.0057 |
| Alert rate | 39.7% of encounters flagged |

Confusion matrix: TP 1,399 · FP 6,497 · FN 864 · TN 11,107

## Risk bands (test set)

| Band | Probability range | Encounters | Share | Observed 30-day readmission |
|---|---|---|---|---|
| Low | 0.00–0.10 | 10,569 | 53.2% | 6.9% |
| Moderate | 0.10–0.20 | 7,339 | 36.9% | 13.7% |
| High | 0.20–1.00 | 1,959 | 9.9% | 27.0% |

## Global explanation (top 10 by mean |SHAP|, shap.TreeExplainer (tree_path_dependent))

1. Discharge disposition — mean |SHAP| 0.2150
2. Prior inpatient admissions — mean |SHAP| 0.1975
3. Total prior visits — mean |SHAP| 0.0680
4. Number of diagnoses — mean |SHAP| 0.0465
5. Any prior inpatient stay — mean |SHAP| 0.0432
6. Primary diagnosis group — mean |SHAP| 0.0419
7. Age — mean |SHAP| 0.0393
8. Lab procedures — mean |SHAP| 0.0334
9. Diabetes medications — mean |SHAP| 0.0277
10. Length of stay — mean |SHAP| 0.0240

SHAP values describe how the model's prediction depends on each input (predictive association learned from historical data). They are not causal effects: changing a feature in reality will not necessarily change risk.

## Caveat

Retrospective, single-dataset evaluation (1999-2008 US hospitals). These metrics do not establish clinical usefulness or performance in any other population.
