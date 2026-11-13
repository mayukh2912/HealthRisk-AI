"""Offline / batch inference using a registered model version.

The serving API uses the same engine (``healthrisk_runtime.engine``); this module adds
registry lookup and DataFrame/CSV helpers for local scoring.

    python -m ml.inference input.csv output.csv [--version 1.0.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config
from healthrisk_runtime.engine import PortableModel, load_model
from healthrisk_runtime.validation import record_id, validate_record


def load_registered_model(version: Optional[str] = None) -> PortableModel:
    reg = json.loads(config.REGISTRY.read_text())
    version = version or reg.get("production")
    entry = next((v for v in reg["versions"] if v["model_version"] == version), None)
    if entry is None:
        raise KeyError(f"Model version {version!r} not in registry ({[v['model_version'] for v in reg['versions']]})")
    return load_model(config.ROOT / entry["artifact"])


def predict_frame(df: pd.DataFrame, model: Optional[PortableModel] = None) -> pd.DataFrame:
    model = model or load_registered_model()
    rows = []
    for i, rec in enumerate(df.astype(object).where(pd.notna(df), None).to_dict("records")):
        _, errors = validate_record(rec)
        base = {"record_id": record_id(rec, i), "model_version": model.version}
        if errors:
            rows.append({**base, "predicted_probability": None, "predicted_class": None,
                         "risk_category": None, "errors": "; ".join(errors)})
            continue
        out = model.predict(rec)
        rows.append({**base, "predicted_probability": round(out["probability"], 6),
                     "predicted_class": out["predicted_class"], "risk_category": out["risk_category"], "errors": ""})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("output"); ap.add_argument("--version")
    a = ap.parse_args(argv)
    res = predict_frame(pd.read_csv(a.input, dtype=str, keep_default_na=False), load_registered_model(a.version))
    res.to_csv(a.output, index=False)
    print(f"Scored {res['predicted_probability'].notna().sum()} / {len(res)} rows -> {Path(a.output)}")


if __name__ == "__main__":
    main()
