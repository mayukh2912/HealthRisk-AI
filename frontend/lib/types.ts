export type RateRow = { group: string; count: number; rate: number };

export type Contribution = {
  feature: string;
  label: string;
  value: string | number | null;
  contribution: number;
  direction: "increases" | "decreases" | "neutral";
};

export type Explanation = {
  method: string;
  units: string;
  base_value: number;
  output_value: number;
  top_contributors: Contribution[];
  increasing: Contribution[];
  decreasing: Contribution[];
};

export type PredictionResponse = {
  prediction: number;
  prediction_label: string;
  probability: number;
  risk_category: string;
  decision_threshold: number;
  model_name: string;
  model_version: string;
  timestamp: string;
  explanation: Explanation | null;
  disclaimer: string;
};

export type BatchPrediction = {
  record_id: string;
  predicted_probability: number;
  predicted_class: number;
  risk_category: string;
  model_version: string;
};

export type BatchResponse = {
  summary: {
    rows_received: number;
    rows_scored: number;
    rows_rejected: number;
    flagged: number;
    risk_categories: Record<string, number>;
    mean_probability: number | null;
    decision_threshold: number;
    model_version: string;
    timestamp: string;
  };
  predictions: BatchPrediction[];
  errors: { row: number; record_id: string; errors: string[] }[];
  disclaimer: string;
};

export type ConfusionMatrix = { tn: number; fp: number; fn: number; tp: number };

export type TestMetrics = {
  roc_auc: number;
  pr_auc: number;
  brier: number;
  log_loss: number;
  ece: number;
  prevalence: number;
  uncalibrated_brier: number | null;
  threshold: number;
  accuracy: number;
  precision: number;
  recall: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  f2: number;
  alert_rate: number;
  confusion_matrix: ConfusionMatrix;
};

export type ComparisonRow = {
  model: string;
  label: string;
  cv_pr_auc_mean: number;
  cv_pr_auc_std: number;
  cv_roc_auc_mean: number;
  cv_brier_mean: number;
  cv_train_pr_auc_mean: number;
  test_pr_auc: number;
  test_roc_auc: number;
  test_brier: number;
  test_ece: number;
  test_precision: number;
  test_recall: number;
  test_specificity: number;
  test_f1: number;
  test_accuracy: number;
  threshold: number;
  best_params: Record<string, unknown>;
  fit_seconds: number;
  selected: boolean;
};

export type ThresholdRow = {
  threshold: number;
  precision: number;
  recall: number;
  specificity: number;
  f1: number;
  alert_rate: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
};

export type DriftFeature = {
  feature: string;
  kind: "numeric" | "categorical";
  psi: number;
  status: string;
  bins: { bin: string; reference: number; current: number }[];
};

export type DriftSet = {
  n: number;
  features: DriftFeature[];
  prediction_hist: { edges: number[]; counts: number[] };
  mean_prediction: number;
};

export type SubgroupRow = {
  group: string;
  n: number;
  prevalence: number;
  roc_auc: number;
  pr_auc: number;
  recall: number;
  precision: number;
  mean_predicted: number;
};

export type Analytics = {
  generated_from: { model_version: string; dataset_sha256: string };
  dataset: {
    historical: {
      encounters: number;
      patients: number;
      positives: number;
      readmission_rate_30d: number;
      imbalance_ratio: number;
      raw_label_distribution: Record<string, number>;
    };
    quality: Record<string, unknown> & { rows: number; missing_pct: Record<string, number> };
    cohort: Record<string, number>;
    note: string;
  };
  eda: {
    overview: Analytics["dataset"]["historical"];
    descriptive_stats: Record<string, Record<string, number>>;
    univariate: Record<string, RateRow[]>;
    bivariate: Record<string, RateRow[]>;
    multivariate: {
      spearman_correlation: { columns: string[]; matrix: number[][] };
      age_by_admission_type: { age: string; admission_type: string; rate: number; count: number }[];
      age_by_prior_inpatient: { rows: string[]; columns: string[]; matrix: (number | null)[][] };
    };
    statistical_tests: {
      numeric_mann_whitney: { feature: string; mean_readmitted: number; mean_not_readmitted: number; rank_biserial: number; p_value: number }[];
      categorical_chi_square: { feature: string; chi2: number; dof: number; p_value: number; cramers_v: number }[];
      caveat: string;
    };
    findings: string[];
  };
  model: {
    name: string;
    version: string;
    trained_at: string;
    decision_threshold: number;
    risk_bands: { name: string; upper: number; guidance: string }[];
    selection_rule: string;
    threshold_strategy: string;
    test_set: { n: number; patients: number; prevalence: number };
    best_params: Record<string, unknown>;
    note: string;
  };
  evaluation: {
    test_metrics: TestMetrics;
    test_at_0_5: TestMetrics;
    curves: {
      roc: { fpr: number; tpr: number }[];
      pr: { recall: number; precision: number }[];
      calibration: { mean_predicted: number; observed_rate: number }[];
      calibration_uncalibrated?: { mean_predicted: number; observed_rate: number }[];
    };
    threshold_table: ThresholdRow[];
    risk_bands: { band: string; lower: number; upper: number; count: number; share: number; observed_rate: number | null }[];
    subgroups: Record<string, SubgroupRow[]>;
    comparison: ComparisonRow[];
    caveat: string;
  };
  explainability: {
    method: string;
    sample_size: number;
    mean_abs_shap: { feature: string; label: string; mean_abs_shap: number }[];
    shap_distribution: { feature: string; label: string; p5: number; p25: number; median: number; p75: number; p95: number; share_positive: number }[];
    dependence: { feature: string; label: string; points: { value: number; mean_shap: number; n: number }[] }[];
    permutation_importance_pr_auc: { feature: string; label: string; importance: number; std: number }[];
    summary_plot: string;
    interpretation_note: string;
  };
  monitoring: {
    baseline_drift: {
      held_out_test: DriftSet;
      latest_20pct_encounters: DriftSet;
      notes: string;
    };
  };
};

export type ModelInfo = {
  model_name: string;
  model_version: string;
  trained_at: string;
  decision_threshold: number;
  risk_bands: { name: string; upper: number; guidance: string }[];
  features: string[];
  metrics: { cv: Record<string, number>; test: TestMetrics };
  dataset: Record<string, unknown> & { sha256: string; train_rows: number; test_rows: number };
  registry: {
    production?: string;
    versions: {
      model_version: string;
      model_name: string;
      trained_at: string;
      dataset_sha256: string;
      config_sha256: string;
      test_pr_auc: number;
      test_roc_auc: number;
      test_recall: number;
      test_brier: number;
      decision_threshold: number;
    }[];
  };
  environment: Record<string, string>;
  config_sha256: string;
  disclaimer: string;
};

export type DriftResponse = {
  n_rows: number;
  n_valid: number;
  model_version: string;
  features: DriftFeature[];
  prediction_hist: { edges: number[]; counts: number[] };
  reference_prediction_hist: { edges: number[]; counts: number[] };
  mean_prediction: number;
  note: string;
};
