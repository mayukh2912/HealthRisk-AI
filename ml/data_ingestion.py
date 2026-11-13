"""Phase 1 - dataset acquisition and loading.

Usage:
    python -m ml.data_ingestion             # validate a local copy + write data-quality report
    python -m ml.data_ingestion --download  # try Kaggle (kagglehub), else print manual steps

The raw file is never committed; it lives in ``data/raw/`` (git-ignored).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from . import config  # noqa: F401  (also sets sys.path for the runtime package)
from healthrisk_runtime.features import (
    EXCLUDED_DISPOSITIONS, EXPECTED_RAW_COLUMNS, TARGET, TARGET_RAW, encode_target,
)

KAGGLE_DATASET = "brandao/diabetes"
MANUAL_INSTRUCTIONS = f"""
Dataset not found at {config.RAW_CSV}.

Download it manually (identical content from either source):
  * Kaggle: https://www.kaggle.com/datasets/brandao/diabetes  (Download -> archive.zip)
  * UCI:    https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
Then unzip so that these files exist:
  {config.RAW_CSV}
  {config.MAPPING_CSV}   (optional, used for documentation only)
Or run with Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY):
  python -m ml.data_ingestion --download
"""


class DatasetNotFoundError(FileNotFoundError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_from_kaggle(dest: Path = config.RAW_DIR) -> Path:
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pip install kagglehub to enable automatic download") from exc
    src = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("diabetic_data.csv", "IDS_mapping.csv"):
        found = next(src.rglob(name), None)
        if found:
            shutil.copy2(found, dest / name)
    return dest / "diabetic_data.csv"


def load_raw(path: Path | str = config.RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV exactly as published. '?' is the dataset's missing marker;
    the literal string 'None' (e.g. A1Cresult) is a valid category, not missing."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(MANUAL_INSTRUCTIONS)
    df = pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False,
                     dtype={"payer_code": "object", "medical_specialty": "object",
                            "diag_1": "object", "diag_2": "object", "diag_3": "object"})
    validate_columns(df)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")


def build_cohort(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply documented cohort exclusions and derive the binary target.

    * remove exact duplicate encounter ids (keep first)
    * remove encounters ending in death or hospice (not at risk of readmission)
    * remove records with invalid gender ('Unknown/Invalid')
    """
    log: Dict[str, int] = {"raw_rows": int(len(df))}
    out = df.drop_duplicates(subset="encounter_id", keep="first")
    log["removed_duplicate_encounters"] = int(len(df) - len(out))
    mask = out["discharge_disposition_id"].isin(EXCLUDED_DISPOSITIONS)
    log["removed_expired_or_hospice"] = int(mask.sum())
    out = out[~mask]
    mask = ~out["gender"].isin(["Female", "Male"])
    log["removed_invalid_gender"] = int(mask.sum())
    out = out[~mask].copy()
    out[TARGET] = out[TARGET_RAW].map(encode_target).astype(int)
    log["cohort_rows"] = int(len(out))
    log["cohort_patients"] = int(out["patient_nbr"].nunique())
    return out.reset_index(drop=True), log


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    if args.download and not config.RAW_CSV.exists():
        try:
            download_from_kaggle()
        except Exception as exc:  # noqa: BLE001
            print(f"Automatic download failed: {exc}")
            print(MANUAL_INSTRUCTIONS)
            return 1
    try:
        df = load_raw()
    except DatasetNotFoundError as exc:
        print(exc)
        return 1
    from .validation import write_quality_report
    report = write_quality_report(df)
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns; sha256={report['dataset_sha256'][:12]}")
    print(f"Data-quality report written to {config.REPORTS_DIR / 'data_quality_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
