export const pct = (v: number | null | undefined, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);

export const int = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : Math.round(v).toLocaleString("en-US");

export const dateTime = (iso: string | undefined) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" });
};

export const riskColor = (category: string) =>
  category === "High" ? "var(--color-risk-high)" : category === "Moderate" ? "var(--color-risk-moderate)" : "var(--color-risk-low)";

export const FEATURE_NAMES: Record<string, string> = {
  time_in_hospital: "Length of stay", num_lab_procedures: "Lab procedures", num_procedures: "Procedures",
  num_medications: "Medications", number_outpatient: "Prior outpatient", number_emergency: "Prior emergency",
  number_inpatient: "Prior inpatient", number_diagnoses: "Diagnoses", age_years: "Age (years)",
  total_prior_visits: "Total prior visits", meds_per_day: "Medications / day", n_diabetes_meds: "Diabetes meds",
  gender: "Gender", admission_type: "Admission type", discharge_disposition: "Discharge disposition",
  admission_source: "Admission source", diag_1_group: "Primary diagnosis group", A1Cresult: "HbA1c result",
  insulin: "Insulin", los_category: "Length-of-stay band", readmitted_30d: "Readmitted ≤30d",
};
