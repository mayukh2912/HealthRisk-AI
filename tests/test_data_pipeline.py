import numpy as np
import pandas as pd
import pytest

from conftest import needs_data
from ml.data_ingestion import DatasetNotFoundError, build_cohort, load_raw, validate_columns
from ml.feature_engineering import EncounterFeatureEngineer, engineer_frame
from ml.preprocessing import build_preprocessor, patient_group_split
from ml.validation import quality_checks
from healthrisk_runtime.features import (
    EXCLUDED_COLUMNS, EXCLUDED_DISPOSITIONS, MODEL_FEATURES, age_midpoint, encode_target, engineer_record,
    icd9_group, los_category,
)


def test_load_raw_missing_file_gives_instructions(tmp_path):
    with pytest.raises(DatasetNotFoundError) as e:
        load_raw(tmp_path / "nope.csv")
    assert "kaggle.com/datasets/brandao/diabetes" in str(e.value)


def test_validate_columns_rejects_missing(synthetic_raw):
    with pytest.raises(ValueError, match="readmitted"):
        validate_columns(synthetic_raw.drop(columns=["readmitted"]))


@pytest.mark.parametrize("raw,expected", [("<30", 1), (">30", 0), ("NO", 0)])
def test_target_encoding(raw, expected):
    assert encode_target(raw) == expected


def test_target_encoding_rejects_unknown():
    with pytest.raises(ValueError):
        encode_target("YES")


def test_cohort_exclusions(synthetic_raw):
    df = synthetic_raw.copy()
    df.loc[0, "gender"] = "Unknown/Invalid"
    cohort, log = build_cohort(df)
    assert not cohort["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS).any()
    assert set(cohort["gender"]) <= {"Female", "Male"}
    assert log["removed_invalid_gender"] == 1
    assert log["cohort_rows"] == len(cohort)
    assert cohort["readmitted_30d"].isin([0, 1]).all()
    assert (cohort["readmitted_30d"] == (cohort["readmitted"] == "<30")).all()


def test_quality_checks_detect_missing_and_duplicates(synthetic_raw):
    df = pd.concat([synthetic_raw, synthetic_raw.iloc[[0]]])
    q = quality_checks(df)
    assert q["duplicate_encounter_ids"] == 1
    assert q["missing_pct"]["weight"] == 100.0
    assert q["patients_with_multiple_encounters"] > 0


def test_patient_group_split_has_no_overlap(synthetic_raw):
    cohort, _ = build_cohort(synthetic_raw)
    tr, te = patient_group_split(cohort["readmitted_30d"], cohort["patient_nbr"], 5, 42)
    assert not set(cohort["patient_nbr"].iloc[tr]) & set(cohort["patient_nbr"].iloc[te])
    assert len(tr) + len(te) == len(cohort)
    assert 0.1 < len(te) / len(cohort) < 0.3


def test_feature_engineering_values():
    assert icd9_group("250.83") == "Diabetes"
    assert icd9_group("428") == "Circulatory" and icd9_group("786") == "Respiratory"
    assert icd9_group("V45") == "Other" and icd9_group(None) == "Missing" and icd9_group("?") == "Missing"
    assert los_category(2) == "1-2 days" and los_category(5) == "3-6 days" and los_category(9) == "7+ days"
    assert age_midpoint("[70-80)") == 75.0 and age_midpoint("bad") is None
    f = engineer_record({"number_outpatient": 1, "number_emergency": 2, "number_inpatient": 3, "num_medications": 20,
                         "time_in_hospital": 4, "insulin": "Up", "metformin": "Steady", "A1Cresult": ">8"})
    assert f["total_prior_visits"] == 6 and f["any_prior_inpatient"] == 1
    assert f["meds_per_day"] == 5 and f["n_diabetes_meds"] == 2 and f["n_med_dose_changes"] == 1
    assert f["insulin_treated"] == 1 and f["a1c_tested"] == 1 and f["glucose_tested"] == 0
    assert f["medical_specialty"] == "Missing"


def test_feature_engineering_is_stateless_and_leak_free(synthetic_raw):
    fe = EncounterFeatureEngineer()
    a = fe.fit(synthetic_raw.iloc[:100]).transform(synthetic_raw.iloc[200:300])
    b = fe.fit(synthetic_raw.iloc[500:]).transform(synthetic_raw.iloc[200:300])
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == MODEL_FEATURES
    assert not set(EXCLUDED_COLUMNS) & set(a.columns)


def test_preprocessor_handles_missing_and_unseen(synthetic_raw):
    X = engineer_frame(synthetic_raw)
    pre = build_preprocessor(min_frequency=20).fit(X.iloc[:1000])
    test = X.iloc[1000:1010].copy()
    test.loc[test.index[0], "medical_specialty"] = "Never-Seen-Specialty"
    test.loc[test.index[1], "num_medications"] = np.nan
    out = pre.transform(test)
    assert np.isfinite(out).all()


@needs_data
def test_real_dataset_loads_and_validates():
    df = load_raw()
    assert df.shape == (101766, 50)
    q = quality_checks(df)
    assert q["duplicate_encounter_ids"] == 0
    assert q["constant_columns"] == ["examide", "citoglipton"]
    cohort, log = build_cohort(df)
    assert 0.10 < cohort["readmitted_30d"].mean() < 0.13
