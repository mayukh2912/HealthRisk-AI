"""Paths and training configuration. Loaded from ml/training_config.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_CSV = RAW_DIR / "diabetic_data.csv"
MAPPING_CSV = RAW_DIR / "IDS_mapping.csv"
MODELS_DIR = ROOT / "models"
METADATA_DIR = MODELS_DIR / "metadata"
REGISTRY = MODELS_DIR / "registry.json"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EVAL_DIR = REPORTS_DIR / "evaluation"
BACKEND_ARTIFACTS = ROOT / "backend" / "artifacts"
FRONTEND_PUBLIC = ROOT / "frontend" / "public"
CONFIG_PATH = Path(__file__).with_name("training_config.json")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
