"""Shared fixtures.

`synthetic_raw` builds a small *synthetic* DataFrame in the raw dataset schema purely to
unit-test pipeline mechanics. It is random test data, not patient data, and is never used
for reported results. Tests needing the real dataset are skipped when it is absent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import ml  # noqa: E402,F401  (adds backend/ to sys.path)
from ml import config  # noqa: E402
from healthrisk_runtime.features import AGE_BRACKETS, DIABETES_MEDICATIONS, EXPECTED_RAW_COLUMNS  # noqa: E402

HAS_DATA = config.RAW_CSV.exists()
needs_data = pytest.mark.skipif(not HAS_DATA, reason="real dataset not present in data/raw/")
HAS_ARTIFACTS = (ROOT / "backend" / "artifacts" / "model.json").exists()
needs_artifacts = pytest.mark.skipif(not HAS_ARTIFACTS, reason="run `python -m ml.train` first")


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 1500
    patients = rng.integers(1, 600, n)
    df = pd.DataFrame({
        "encounter_id": np.arange(1, n + 1), "patient_nbr": patients,
        "race": rng.choice(["Caucasian", "AfricanAmerican", None], n), "gender": rng.choice(["Female", "Male"], n),
        "age": rng.choice(AGE_BRACKETS[3:], n), "weight": None,
        "admission_type_id": rng.choice([1, 2, 3, 6], n), "discharge_disposition_id": rng.choice([1, 3, 6, 11, 22], n),
        "admission_source_id": rng.choice([1, 7, 4], n), "time_in_hospital": rng.integers(1, 15, n),
        "payer_code": None, "medical_specialty": rng.choice(["InternalMedicine", "Cardiology", None], n),
        "num_lab_procedures": rng.integers(1, 90, n), "num_procedures": rng.integers(0, 6, n),
        "num_medications": rng.integers(1, 40, n), "number_outpatient": rng.poisson(0.3, n),
        "number_emergency": rng.poisson(0.2, n), "number_inpatient": rng.poisson(0.6, n),
        "diag_1": rng.choice(["428", "250.02", "414", "V45", "786"], n), "diag_2": rng.choice(["401", "250", None], n),
        "diag_3": rng.choice(["276", "E888", None], n), "number_diagnoses": rng.integers(1, 17, n),
        "max_glu_serum": rng.choice(["None", "Norm", ">200"], n), "A1Cresult": rng.choice(["None", ">8", "Norm"], n),
        "change": rng.choice(["No", "Ch"], n), "diabetesMed": rng.choice(["No", "Yes"], n),
    })
    for m in DIABETES_MEDICATIONS:
        df[m] = rng.choice(["No", "Steady", "Up", "Down"], n, p=[0.7, 0.2, 0.05, 0.05]) if m in ("insulin", "metformin") else "No"
    risk = 0.08 + 0.06 * df["number_inpatient"].clip(upper=4)
    df["readmitted"] = np.where(rng.random(n) < risk, "<30", rng.choice([">30", "NO"], n))
    return df[EXPECTED_RAW_COLUMNS]


@pytest.fixture
def valid_patient() -> dict:
    return {"gender": "Female", "age": "[70-80)", "admission_type_id": 1, "discharge_disposition_id": 6,
            "admission_source_id": 7, "time_in_hospital": 6, "num_lab_procedures": 52, "num_procedures": 1,
            "num_medications": 18, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2,
            "number_diagnoses": 9, "diag_1": "428", "diag_2": "250.02", "diag_3": "403",
            "medical_specialty": "InternalMedicine", "max_glu_serum": "None", "A1Cresult": "None",
            "insulin": "Up", "metformin": "No", "change": "Ch", "diabetesMed": "Yes"}
