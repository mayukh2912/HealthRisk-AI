"""Pydantic request/response models for the HealthRisk AI API."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

MedStatus = Literal["No", "Steady", "Up", "Down"]
AgeBracket = Literal["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)",
                     "[70-80)", "[80-90)", "[90-100)"]
ICD9 = r"^(?i:V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?|\d{1,3}(\.\d{1,2})?)$"


class PatientInput(BaseModel):
    """Discharge-time encounter features. No direct identifiers are accepted."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "gender": "Female", "age": "[70-80)", "admission_type_id": 1, "discharge_disposition_id": 1,
        "admission_source_id": 7, "time_in_hospital": 5, "num_lab_procedures": 45, "num_procedures": 1,
        "num_medications": 16, "number_outpatient": 0, "number_emergency": 0, "number_inpatient": 1,
        "number_diagnoses": 9, "diag_1": "428", "diag_2": "250.02", "diag_3": "401",
        "medical_specialty": "InternalMedicine", "max_glu_serum": "None", "A1Cresult": "None",
        "insulin": "Steady", "metformin": "No", "change": "No", "diabetesMed": "Yes"}})

    gender: Literal["Female", "Male"]
    age: AgeBracket
    admission_type_id: int = Field(ge=1, le=8)
    discharge_disposition_id: int = Field(ge=1, le=30)
    admission_source_id: int = Field(ge=1, le=26)
    time_in_hospital: int = Field(ge=1, le=14, description="Length of stay in days")
    num_lab_procedures: int = Field(ge=0, le=200)
    num_procedures: int = Field(ge=0, le=10)
    num_medications: int = Field(ge=0, le=100)
    number_outpatient: int = Field(ge=0, le=60)
    number_emergency: int = Field(ge=0, le=80)
    number_inpatient: int = Field(ge=0, le=30)
    number_diagnoses: int = Field(ge=1, le=16)
    diag_1: str = Field(pattern=ICD9, max_length=7)
    diag_2: Optional[str] = Field(default=None, pattern=ICD9, max_length=7)
    diag_3: Optional[str] = Field(default=None, pattern=ICD9, max_length=7)
    medical_specialty: Optional[str] = Field(default=None, max_length=80)
    max_glu_serum: Literal["None", "Norm", ">200", ">300"]
    A1Cresult: Literal["None", "Norm", ">7", ">8"]
    insulin: MedStatus
    metformin: MedStatus
    change: Literal["No", "Ch"]
    diabetesMed: Literal["No", "Yes"]
    # Optional: other diabetes medications (default "No")
    repaglinide: MedStatus = "No"
    nateglinide: MedStatus = "No"
    chlorpropamide: MedStatus = "No"
    glimepiride: MedStatus = "No"
    acetohexamide: MedStatus = "No"
    glipizide: MedStatus = "No"
    glyburide: MedStatus = "No"
    tolbutamide: MedStatus = "No"
    pioglitazone: MedStatus = "No"
    rosiglitazone: MedStatus = "No"
    acarbose: MedStatus = "No"
    miglitol: MedStatus = "No"
    troglitazone: MedStatus = "No"
    tolazamide: MedStatus = "No"
    examide: MedStatus = "No"
    citoglipton: MedStatus = "No"
    glyburide_metformin: MedStatus = Field(default="No", alias="glyburide-metformin")
    glipizide_metformin: MedStatus = Field(default="No", alias="glipizide-metformin")
    glimepiride_pioglitazone: MedStatus = Field(default="No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: MedStatus = Field(default="No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: MedStatus = Field(default="No", alias="metformin-pioglitazone")

    def to_raw(self) -> Dict[str, Any]:
        return self.model_dump(by_alias=True)


class Contribution(BaseModel):
    feature: str
    label: str
    value: Any = None
    contribution: float
    direction: str


class Explanation(BaseModel):
    method: str
    units: str
    base_value: float
    output_value: float
    top_contributors: List[Contribution]
    increasing: List[Contribution]
    decreasing: List[Contribution]


class PredictionResponse(BaseModel):
    prediction: int = Field(description="1 = flagged as elevated 30-day readmission risk at the model threshold")
    prediction_label: str
    probability: float
    risk_category: str
    decision_threshold: float
    model_name: str
    model_version: str
    timestamp: str
    explanation: Optional[Explanation] = None
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str
