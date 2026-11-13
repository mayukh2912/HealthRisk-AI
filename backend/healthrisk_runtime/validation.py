"""Pure-Python record validation shared by the API (single + batch) and tests."""
from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .features import (
    A1C, AGE_BRACKETS, DIABETES_MEDICATIONS, GENDERS, GLU_SERUM, MED_STATUS,
    NUMERIC_INPUTS, OPTIONAL_INPUTS, REQUIRED_INPUTS, is_missing,
)

ID_FIELDS = ("admission_type_id", "discharge_disposition_id", "admission_source_id")
ID_RANGE = {"admission_type_id": (1, 8), "discharge_disposition_id": (1, 30), "admission_source_id": (1, 26)}
ICD9_RE = re.compile(r"^(V\d{2}(\.\d{1,2})?|E\d{3}(\.\d)?|\d{1,3}(\.\d{1,2})?)$", re.IGNORECASE)
FIXED_VALUES = {
    "gender": GENDERS, "age": AGE_BRACKETS, "max_glu_serum": GLU_SERUM, "A1Cresult": A1C,
    "change": ["No", "Ch"], "diabetesMed": ["No", "Yes"],
    **{m: MED_STATUS for m in DIABETES_MEDICATIONS},
}
# Direct identifiers that must never be uploaded.
FORBIDDEN_COLUMNS = {"name", "first_name", "last_name", "patient_name", "phone", "phone_number",
                     "address", "email", "ssn", "dob", "date_of_birth", "mrn"}
RECORD_ID_COLUMN = "record_id"
MAX_BATCH_ROWS = 5000
MAX_BATCH_BYTES = 4_000_000  # stays under Vercel's 4.5 MB request-body limit


def validate_record(raw: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Return (clean_record, errors). ``errors`` is empty when the record is valid."""
    errors: List[str] = []
    clean: Dict[str, Any] = {}
    for field in REQUIRED_INPUTS:
        if field not in raw or is_missing(raw.get(field)):
            errors.append(f"'{field}' is required")
    for field, spec in NUMERIC_INPUTS.items():
        v = raw.get(field)
        if is_missing(v):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a number")
            continue
        if fv != int(fv):
            errors.append(f"'{field}' must be a whole number")
        elif not (spec["min"] <= fv <= spec["max"]):
            errors.append(f"'{field}' must be between {spec['min']} and {spec['max']}")
        clean[field] = int(fv)
    for field in ID_FIELDS:
        v = raw.get(field)
        if is_missing(v):
            continue
        try:
            iv = int(float(v))
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be an integer code")
            continue
        lo, hi = ID_RANGE[field]
        if not lo <= iv <= hi:
            errors.append(f"'{field}' must be between {lo} and {hi}")
        clean[field] = iv
    for field, allowed in FIXED_VALUES.items():
        v = raw.get(field)
        if is_missing(v):
            if field in OPTIONAL_INPUTS:
                clean[field] = "No"
            continue
        sv = str(v).strip()
        if sv not in allowed:
            errors.append(f"'{field}' must be one of {allowed}")
        clean[field] = sv
    for field in ("diag_1", "diag_2", "diag_3"):
        v = raw.get(field)
        if is_missing(v):
            clean[field] = None
            continue
        sv = str(v).strip()
        if not ICD9_RE.match(sv):
            errors.append(f"'{field}' must be an ICD-9 code such as 428, 250.01, V45 or E888")
        clean[field] = sv
    spec = raw.get("medical_specialty")
    clean["medical_specialty"] = None if is_missing(spec) else str(spec).strip()[:80]
    return clean, errors


def parse_csv(text: str) -> Tuple[List[Dict[str, str]], List[str]]:
    reader = csv.DictReader(io.StringIO(text))
    header = [h.strip() for h in (reader.fieldnames or [])]
    reader.fieldnames = header
    return [dict(r) for r in reader], header


def check_header(header: List[str]) -> List[str]:
    errors = []
    lower = {h.lower() for h in header}
    forbidden = sorted(lower & FORBIDDEN_COLUMNS)
    if forbidden:
        errors.append(f"Remove direct patient identifiers before upload: {forbidden}")
    missing = [c for c in REQUIRED_INPUTS if c not in header]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    return errors


def record_id(row: Mapping[str, Any], index: int) -> str:
    rid = row.get(RECORD_ID_COLUMN)
    return str(rid).strip()[:64] if rid not in (None, "") else f"row_{index + 1}"


def optional_str(v: Any) -> Optional[str]:
    return None if is_missing(v) else str(v)
