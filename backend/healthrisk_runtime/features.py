"""Single source of truth for the feature contract.

This module is pure Python (no pandas / numpy) so that exactly the same code runs
in the offline training pipeline (``ml/``) and in the lightweight serverless
inference API (``backend/``). Every transformation here is *stateless*: it only
looks at one encounter record and uses no statistics learned from data, so it can
never leak information across the train / test split. Learned preprocessing
(imputation medians, scaling, one-hot categories) lives in the scikit-learn
ColumnTransformer that is fitted on training data only.

Prediction point: **at hospital discharge**. Every raw field used below is
recorded during or before the index encounter and is therefore available when a
discharge-time risk estimate would be produced.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional

TARGET_RAW = "readmitted"
TARGET = "readmitted_30d"
POSITIVE_RAW_LABEL = "<30"

# ---------------------------------------------------------------------------
# Raw columns
# ---------------------------------------------------------------------------
EXPECTED_RAW_COLUMNS: List[str] = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
    "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide-metformin", "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone", "change", "diabetesMed",
    "readmitted",
]

DIABETES_MEDICATIONS: List[str] = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Columns that are never used as model inputs, and why. Surfaced in the docs,
# metadata and the leakage test-suite.
EXCLUDED_COLUMNS: Dict[str, str] = {
    "readmitted": "Original outcome column (the target itself) - direct leakage.",
    "readmitted_30d": "Derived binary target.",
    "encounter_id": "Row identifier; carries no clinical signal and may encode time order.",
    "patient_nbr": "Patient identifier; used ONLY as the grouping key for patient-aware splits.",
    "weight": "~97% missing; unusable and inconsistently recorded.",
    "payer_code": "~40% missing administrative/insurance field; a socioeconomic proxy that "
                  "adds fairness risk without being a clinical factor at discharge.",
    "race": "Sensitive attribute. Excluded from model inputs to avoid encoding demographic "
            "bias; retained only for subgroup (fairness) evaluation.",
    "examide": "Constant column (single value in the entire dataset).",
    "citoglipton": "Constant column (single value in the entire dataset).",
}

# Discharge dispositions that remove a patient from the at-risk population.
# 11 = Expired, 19/20/21 = Expired (hospice), 13/14 = Hospice. These patients
# cannot (or are not expected to) be readmitted, so they are removed from the
# cohort rather than labelled "not readmitted".
EXCLUDED_DISPOSITIONS = {11, 13, 14, 19, 20, 21}

AGE_BRACKETS = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
MED_STATUS = ["No", "Steady", "Up", "Down"]
GLU_SERUM = ["None", "Norm", ">200", ">300"]
A1C = ["None", "Norm", ">7", ">8"]
GENDERS = ["Female", "Male"]

# ---------------------------------------------------------------------------
# Model input contract (what the API / batch CSV must provide)
# ---------------------------------------------------------------------------
NUMERIC_INPUTS: Dict[str, Dict[str, Any]] = {
    "time_in_hospital":   {"min": 1, "max": 14, "label": "Length of stay (days)"},
    "num_lab_procedures": {"min": 0, "max": 200, "label": "Lab procedures"},
    "num_procedures":     {"min": 0, "max": 10, "label": "Non-lab procedures"},
    "num_medications":    {"min": 0, "max": 100, "label": "Distinct medications"},
    "number_outpatient":  {"min": 0, "max": 60, "label": "Outpatient visits (prior year)"},
    "number_emergency":   {"min": 0, "max": 80, "label": "Emergency visits (prior year)"},
    "number_inpatient":   {"min": 0, "max": 30, "label": "Inpatient admissions (prior year)"},
    "number_diagnoses":   {"min": 1, "max": 16, "label": "Diagnoses recorded"},
}

CATEGORICAL_INPUTS: Dict[str, Dict[str, Any]] = {
    "gender": {"values": GENDERS, "label": "Gender"},
    "age": {"values": AGE_BRACKETS, "label": "Age bracket"},
    "admission_type_id": {"values": None, "label": "Admission type ID"},
    "discharge_disposition_id": {"values": None, "label": "Discharge disposition ID"},
    "admission_source_id": {"values": None, "label": "Admission source ID"},
    "medical_specialty": {"values": None, "label": "Admitting physician specialty"},
    "diag_1": {"values": None, "label": "Primary diagnosis (ICD-9)"},
    "diag_2": {"values": None, "label": "Secondary diagnosis (ICD-9)"},
    "diag_3": {"values": None, "label": "Additional diagnosis (ICD-9)"},
    "max_glu_serum": {"values": GLU_SERUM, "label": "Max glucose serum test"},
    "A1Cresult": {"values": A1C, "label": "HbA1c result"},
    "change": {"values": ["No", "Ch"], "label": "Diabetes medication changed"},
    "diabetesMed": {"values": ["No", "Yes"], "label": "Diabetes medication prescribed"},
}

REQUIRED_INPUTS: List[str] = list(NUMERIC_INPUTS) + [
    "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "diag_1", "max_glu_serum", "A1Cresult", "change",
    "diabetesMed", "insulin", "metformin",
]
OPTIONAL_INPUTS: List[str] = ["medical_specialty", "diag_2", "diag_3"] + [
    m for m in DIABETES_MEDICATIONS if m not in ("insulin", "metformin")
]

# ---------------------------------------------------------------------------
# Engineered model features
# ---------------------------------------------------------------------------
MODEL_NUMERIC_FEATURES: List[str] = [
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
    "age_years", "total_prior_visits", "any_prior_inpatient", "meds_per_day",
    "n_diabetes_meds", "n_med_dose_changes", "insulin_treated", "on_diabetes_med",
    "a1c_tested", "glucose_tested",
]
MODEL_CATEGORICAL_FEATURES: List[str] = [
    "gender", "admission_type", "discharge_disposition", "admission_source",
    "medical_specialty", "diag_1_group", "diag_2_group", "diag_3_group",
    "max_glu_serum", "A1Cresult", "insulin", "metformin", "change", "los_category",
]
MODEL_FEATURES: List[str] = MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "time_in_hospital": "Length of the index stay in days.",
    "num_lab_procedures": "Number of lab tests performed during the encounter.",
    "num_procedures": "Number of non-lab procedures during the encounter.",
    "num_medications": "Number of distinct generic medications administered.",
    "number_outpatient": "Outpatient visits in the year before the encounter.",
    "number_emergency": "Emergency visits in the year before the encounter.",
    "number_inpatient": "Inpatient admissions in the year before the encounter.",
    "number_diagnoses": "Number of diagnoses entered for the encounter.",
    "age_years": "Midpoint of the 10-year age bracket (ordinal age).",
    "total_prior_visits": "Engineered: outpatient + emergency + inpatient visits in the prior year.",
    "any_prior_inpatient": "Engineered: 1 if any inpatient admission in the prior year.",
    "meds_per_day": "Engineered medication burden: medications / length of stay.",
    "n_diabetes_meds": "Engineered: count of diabetes medications prescribed (not 'No').",
    "n_med_dose_changes": "Engineered: count of diabetes medications with dose Up/Down.",
    "insulin_treated": "Engineered: 1 if insulin was prescribed.",
    "on_diabetes_med": "Engineered: 1 if any diabetes medication was prescribed.",
    "a1c_tested": "Engineered: 1 if an HbA1c test was performed.",
    "glucose_tested": "Engineered: 1 if a max glucose serum test was performed.",
    "gender": "Recorded gender.",
    "admission_type": "Admission type (emergency, urgent, elective ...).",
    "discharge_disposition": "Where the patient was discharged to.",
    "admission_source": "Where the patient was admitted from.",
    "medical_specialty": "Specialty of the admitting physician (rare values grouped).",
    "diag_1_group": "Engineered: ICD-9 chapter group of the primary diagnosis.",
    "diag_2_group": "Engineered: ICD-9 chapter group of the secondary diagnosis.",
    "diag_3_group": "Engineered: ICD-9 chapter group of the additional diagnosis.",
    "max_glu_serum": "Max glucose serum test result category.",
    "A1Cresult": "HbA1c test result category.",
    "insulin": "Insulin status during the encounter (No/Steady/Up/Down).",
    "metformin": "Metformin status during the encounter (No/Steady/Up/Down).",
    "change": "Whether any diabetes medication was changed.",
    "los_category": "Engineered: length-of-stay band (1-2, 3-6, 7+ days).",
}

FEATURE_LABELS: Dict[str, str] = {
    "time_in_hospital": "Length of stay", "num_lab_procedures": "Lab procedures",
    "num_procedures": "Procedures", "num_medications": "Medications",
    "number_outpatient": "Prior outpatient visits", "number_emergency": "Prior emergency visits",
    "number_inpatient": "Prior inpatient admissions", "number_diagnoses": "Number of diagnoses",
    "age_years": "Age", "total_prior_visits": "Total prior visits",
    "any_prior_inpatient": "Any prior inpatient stay", "meds_per_day": "Medications per day",
    "n_diabetes_meds": "Diabetes medications", "n_med_dose_changes": "Diabetes dose changes",
    "insulin_treated": "Insulin treated", "on_diabetes_med": "On diabetes medication",
    "a1c_tested": "HbA1c tested", "glucose_tested": "Glucose serum tested",
    "gender": "Gender", "admission_type": "Admission type",
    "discharge_disposition": "Discharge disposition", "admission_source": "Admission source",
    "medical_specialty": "Physician specialty", "diag_1_group": "Primary diagnosis group",
    "diag_2_group": "Secondary diagnosis group", "diag_3_group": "Additional diagnosis group",
    "max_glu_serum": "Glucose serum result", "A1Cresult": "HbA1c result",
    "insulin": "Insulin status", "metformin": "Metformin status", "change": "Medication changed",
    "los_category": "Length-of-stay band",
}

ADMISSION_TYPE = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn", 5: "Not available",
                  6: "Not available", 7: "Trauma center", 8: "Not available"}
DISCHARGE_DISPOSITION = {
    1: "Home", 2: "Short-term hospital", 3: "Skilled nursing facility", 4: "Intermediate care facility",
    5: "Other inpatient care", 6: "Home with home health", 7: "Left against medical advice",
    8: "Home IV provider", 9: "Admitted as inpatient", 10: "Neonatal aftercare",
    12: "Expected to return (outpatient)", 15: "Swing bed", 16: "Outpatient services (other)",
    17: "Outpatient services (this)", 18: "Not available", 22: "Rehab facility",
    23: "Long-term care hospital", 24: "Medicaid nursing facility", 25: "Not available",
    26: "Not available", 27: "Federal facility", 28: "Psychiatric hospital",
    29: "Critical access hospital", 30: "Other institution",
}
ADMISSION_SOURCE = {
    1: "Physician referral", 2: "Clinic referral", 3: "HMO referral", 4: "Transfer from hospital",
    5: "Transfer from SNF", 6: "Transfer from other facility", 7: "Emergency room",
    8: "Court/law enforcement", 9: "Not available", 10: "Transfer from critical access hospital",
    11: "Normal delivery", 12: "Premature delivery", 13: "Sick baby", 14: "Extramural birth",
    15: "Not available", 17: "Not available", 18: "Transfer from home health agency",
    19: "Readmission to home health agency", 20: "Not available", 21: "Not available",
    22: "Transfer (same facility, separate claim)", 23: "Born inside this hospital",
    24: "Born outside this hospital", 25: "Transfer from ambulatory surgery",
    26: "Transfer from hospice",
}

_MISSING_TOKENS = {"", "?", "none_missing", "nan", "NaN", "NULL", "null"}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in _MISSING_TOKENS:
        return True
    return False


def _clean_str(value: Any) -> Optional[str]:
    if is_missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _num(value: Any) -> Optional[float]:
    if is_missing(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_code(value: Any) -> Optional[int]:
    v = _num(value)
    return None if v is None else int(v)


def age_midpoint(bracket: Any) -> Optional[float]:
    b = _clean_str(bracket)
    if b is None or b not in AGE_BRACKETS:
        return None
    lo = int(b[1:].split("-")[0])
    return lo + 5.0


def icd9_group(code: Any) -> str:
    """Map an ICD-9 code to a chapter-level group (Strack et al., 2014)."""
    c = _clean_str(code)
    if c is None:
        return "Missing"
    if c[0] in ("V", "E", "v", "e"):
        return "Other"
    try:
        v = float(c)
    except ValueError:
        return "Other"
    if math.floor(v) == 250:
        return "Diabetes"
    if 390 <= v <= 459 or math.floor(v) == 785:
        return "Circulatory"
    if 460 <= v <= 519 or math.floor(v) == 786:
        return "Respiratory"
    if 520 <= v <= 579 or math.floor(v) == 787:
        return "Digestive"
    if 800 <= v <= 999:
        return "Injury"
    if 710 <= v <= 739:
        return "Musculoskeletal"
    if 580 <= v <= 629 or math.floor(v) == 788:
        return "Genitourinary"
    if 140 <= v <= 239:
        return "Neoplasms"
    return "Other"


def los_category(days: Optional[float]) -> str:
    if days is None:
        return "Missing"
    if days <= 2:
        return "1-2 days"
    if days <= 6:
        return "3-6 days"
    return "7+ days"


def _coded(mapping: Mapping[int, str], value: Any) -> str:
    code = _int_code(value)
    if code is None:
        return "Not available"
    return mapping.get(code, "Other")


def engineer_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Transform one raw encounter record into the model feature dictionary.

    Missing numeric values are returned as ``None`` (imputed later by the fitted
    preprocessor). Missing categoricals become the explicit category ``"Missing"``.
    """
    out: Dict[str, Any] = {}
    for col in NUMERIC_INPUTS:
        out[col] = _num(raw.get(col))

    inp = out["number_inpatient"] or 0.0
    total = sum((out[c] or 0.0) for c in ("number_outpatient", "number_emergency", "number_inpatient"))
    los = out["time_in_hospital"]
    meds = out["num_medications"]

    out["age_years"] = age_midpoint(raw.get("age"))
    out["total_prior_visits"] = total
    out["any_prior_inpatient"] = 1.0 if inp > 0 else 0.0
    out["meds_per_day"] = (meds / los) if (meds is not None and los) else None

    statuses = [(_clean_str(raw.get(m)) or "No") for m in DIABETES_MEDICATIONS]
    out["n_diabetes_meds"] = float(sum(1 for s in statuses if s != "No"))
    out["n_med_dose_changes"] = float(sum(1 for s in statuses if s in ("Up", "Down")))
    insulin = _clean_str(raw.get("insulin")) or "No"
    metformin = _clean_str(raw.get("metformin")) or "No"
    diabetes_med = _clean_str(raw.get("diabetesMed")) or "No"
    out["insulin_treated"] = 1.0 if insulin != "No" else 0.0
    out["on_diabetes_med"] = 1.0 if diabetes_med == "Yes" else 0.0
    a1c = _clean_str(raw.get("A1Cresult")) or "None"
    glu = _clean_str(raw.get("max_glu_serum")) or "None"
    out["a1c_tested"] = 0.0 if a1c == "None" else 1.0
    out["glucose_tested"] = 0.0 if glu == "None" else 1.0

    out["gender"] = _clean_str(raw.get("gender")) or "Missing"
    out["admission_type"] = _coded(ADMISSION_TYPE, raw.get("admission_type_id"))
    out["discharge_disposition"] = _coded(DISCHARGE_DISPOSITION, raw.get("discharge_disposition_id"))
    out["admission_source"] = _coded(ADMISSION_SOURCE, raw.get("admission_source_id"))
    out["medical_specialty"] = _clean_str(raw.get("medical_specialty")) or "Missing"
    out["diag_1_group"] = icd9_group(raw.get("diag_1"))
    out["diag_2_group"] = icd9_group(raw.get("diag_2"))
    out["diag_3_group"] = icd9_group(raw.get("diag_3"))
    out["max_glu_serum"] = glu
    out["A1Cresult"] = a1c
    out["insulin"] = insulin
    out["metformin"] = metformin
    out["change"] = _clean_str(raw.get("change")) or "No"
    out["los_category"] = los_category(los)
    return out


def engineer_records(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [engineer_record(r) for r in rows]


def encode_target(raw_label: Any) -> int:
    """1 = readmitted within 30 days ('<30'); 0 = '>30' or 'NO'."""
    label = _clean_str(raw_label)
    if label not in ("<30", ">30", "NO"):
        raise ValueError(f"Unexpected readmitted label: {raw_label!r}")
    return 1 if label == POSITIVE_RAW_LABEL else 0
