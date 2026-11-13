"""HealthRisk AI - FastAPI inference service.

Serverless-friendly: the model is a small JSON artefact loaded once at import
(function initialisation); no training, no disk writes, no server-side session
state. Dependencies: fastapi + pydantic only.

Run locally:   cd backend && uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import APIRouter, FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

from healthrisk_runtime.drift import drift_report, histogram  # noqa: E402
from healthrisk_runtime.engine import PortableModel, load_model, utc_now_iso  # noqa: E402
from healthrisk_runtime.features import engineer_record  # noqa: E402
from healthrisk_runtime.validation import (  # noqa: E402
    MAX_BATCH_BYTES, MAX_BATCH_ROWS, check_header, parse_csv, record_id, validate_record,
)
from schemas import HealthResponse, PatientInput, PredictionResponse  # noqa: E402

ARTIFACTS = Path(os.environ.get("HEALTHRISK_ARTIFACTS", Path(__file__).resolve().parent / "artifacts"))
APP_ENV = os.environ.get("APP_ENV", "development")
DISCLAIMER = ("Educational research prototype trained on retrospective 1999-2008 data. Not a medical device, "
              "not clinically validated, and not a diagnosis. Do not use for patient care decisions.")

# Privacy: log only method, path, status and latency - never bodies or query values.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("healthrisk")


def _load_json(name: str) -> Optional[Dict[str, Any]]:
    p = ARTIFACTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


MODEL: Optional[PortableModel] = load_model(ARTIFACTS / "model.json") if (ARTIFACTS / "model.json").exists() else None
METADATA = _load_json("metadata.json") or {}
SCHEMA = _load_json("feature_schema.json") or {}
REFERENCE = _load_json("reference_profile.json")
REGISTRY = _load_json("registry.json") or {"versions": []}
_ANALYTICS: Optional[Dict[str, Any]] = None


def analytics() -> Dict[str, Any]:
    global _ANALYTICS
    if _ANALYTICS is None:
        _ANALYTICS = _load_json("analytics.json") or {}
    return _ANALYTICS


def require_model() -> PortableModel:
    if MODEL is None:
        raise HTTPException(503, "Model artefact not found. Run `python -m ml.train` first.")
    return MODEL


app = FastAPI(title="HealthRisk AI API", version=METADATA.get("model_version", "unversioned"),
              description="30-day readmission risk estimation (educational prototype). " + DISCLAIMER,
              docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)

origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.middleware("http")
async def access_log(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - t0) * 1000)
    response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        loc = [str(x) for x in e.get("loc", []) if x not in ("body",)]
        errors.append({"field": ".".join(loc) or "body", "message": e.get("msg", "invalid value")})
    return JSONResponse(status_code=422, content={"error": "validation_error", "detail": errors})


router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok" if MODEL else "degraded", model_loaded=MODEL is not None,
                          model_version=MODEL.version if MODEL else None, timestamp=utc_now_iso())


@router.get("/model-info")
def model_info() -> Dict[str, Any]:
    m = require_model()
    md = METADATA
    return {
        "model_name": m.name, "model_version": m.version, "model_key": md.get("model_key"),
        "trained_at": md.get("trained_at"), "decision_threshold": m.threshold, "risk_bands": m.risk_bands,
        "calibration": md.get("calibration"), "best_params": md.get("best_params"),
        "metrics": md.get("metrics"), "dataset": md.get("dataset"), "validation": md.get("validation"),
        "features": md.get("features"), "feature_schema": SCHEMA, "environment": md.get("environment"),
        "config_sha256": md.get("config_sha256"), "registry": REGISTRY, "disclaimer": DISCLAIMER,
    }


@router.get("/schema")
def schema() -> Dict[str, Any]:
    return SCHEMA


def _predict(raw: Dict[str, Any], explain: bool) -> Dict[str, Any]:
    m = require_model()
    res = m.predict(raw)
    out = {
        "prediction": res["predicted_class"],
        "prediction_label": "Elevated risk (flagged)" if res["predicted_class"] else "Not flagged",
        "probability": round(res["probability"], 6), "risk_category": res["risk_category"],
        "decision_threshold": m.threshold, "model_name": m.name, "model_version": m.version,
        "timestamp": utc_now_iso(), "disclaimer": DISCLAIMER, "explanation": None,
    }
    if explain:
        ex = m.explain(raw, top_k=8)
        ex.pop("all_contributions", None)
        out["explanation"] = ex
    return out


@router.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput, explain: bool = Query(True)) -> Dict[str, Any]:
    raw = patient.to_raw()
    _, errors = validate_record(raw)
    if errors:
        raise HTTPException(422, {"error": "validation_error", "detail": [{"field": "body", "message": e} for e in errors]})
    return _predict(raw, explain)


async def _read_csv(request: Request) -> tuple[List[Dict[str, str]], List[str]]:
    body = await request.body()
    if not body:
        raise HTTPException(400, "Empty request body. Send CSV text with Content-Type: text/csv.")
    if len(body) > MAX_BATCH_BYTES:
        raise HTTPException(413, f"CSV larger than {MAX_BATCH_BYTES // 1_000_000} MB. Split the file.")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "CSV must be UTF-8 encoded.")
    rows, header = parse_csv(text)
    errs = check_header(header)
    if errs:
        raise HTTPException(422, {"error": "invalid_columns", "detail": errs})
    if not rows:
        raise HTTPException(422, {"error": "empty_file", "detail": ["The CSV has a header but no rows."]})
    if len(rows) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"At most {MAX_BATCH_ROWS:,} rows per request (got {len(rows):,}).")
    return rows, header


def _csv_escape(v: Any) -> str:
    s = "" if v is None else str(v)
    if s[:1] in ("=", "+", "-", "@") and not s.replace(".", "", 1).lstrip("-").isdigit():
        s = "'" + s  # neutralise spreadsheet formula injection
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


@router.post("/batch-predict")
async def batch_predict(request: Request, format: str = Query("json", pattern="^(json|csv)$")):
    m = require_model()
    rows, _ = await _read_csv(request)
    results, errors = [], []
    for i, row in enumerate(rows):
        rid = record_id(row, i)
        clean, errs = validate_record(row)
        if errs:
            errors.append({"row": i + 1, "record_id": rid, "errors": errs})
            continue
        res = m.predict(row)
        results.append({"record_id": rid, "predicted_probability": round(res["probability"], 6),
                        "predicted_class": res["predicted_class"], "risk_category": res["risk_category"],
                        "model_version": m.version})
    summary = {"rows_received": len(rows), "rows_scored": len(results), "rows_rejected": len(errors),
               "flagged": sum(r["predicted_class"] for r in results),
               "risk_categories": {b["name"]: sum(1 for r in results if r["risk_category"] == b["name"]) for b in m.risk_bands},
               "mean_probability": round(sum(r["predicted_probability"] for r in results) / len(results), 6) if results else None,
               "decision_threshold": m.threshold, "model_version": m.version, "timestamp": utc_now_iso()}
    if format == "csv":
        cols = ["record_id", "predicted_probability", "predicted_class", "risk_category", "model_version"]
        lines = [",".join(cols)] + [",".join(_csv_escape(r[c]) for c in cols) for r in results]
        return Response("\n".join(lines) + "\n", media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="healthrisk_predictions.csv"'})
    return {"summary": summary, "predictions": results, "errors": errors[:200], "disclaimer": DISCLAIMER}


@router.post("/monitoring/drift")
async def monitoring_drift(request: Request):
    m = require_model()
    if REFERENCE is None:
        raise HTTPException(503, "Reference profile missing.")
    rows, _ = await _read_csv(request)
    valid = [r for r in rows if not validate_record(r)[1]]
    if len(valid) < 30:
        raise HTTPException(422, {"error": "too_few_rows", "detail": ["At least 30 valid rows are needed for PSI."]})
    feats = [engineer_record(r) for r in valid]
    probs = [m.predict_features(f)["probability"] for f in feats]
    edges = REFERENCE["prediction_hist"]["edges"]
    return {"n_rows": len(rows), "n_valid": len(valid), "model_version": m.version,
            "features": drift_report(REFERENCE, feats),
            "prediction_hist": {"edges": edges, "counts": histogram(probs, edges)},
            "reference_prediction_hist": REFERENCE["prediction_hist"],
            "mean_prediction": round(sum(probs) / len(probs), 4), "psi_bands": REFERENCE["psi_bands"],
            "note": "Input and prediction drift only. Outcome-based performance monitoring requires collected labels."}


@router.get("/analytics")
def get_analytics(section: Optional[str] = Query(None, pattern="^[a-z_]+$")):
    data = analytics()
    if not data:
        raise HTTPException(503, "Analytics artefact not found. Run `python -m ml.train`.")
    if section:
        if section not in data:
            raise HTTPException(404, f"Unknown section '{section}'. Available: {sorted(data)}")
        return JSONResponse(data[section], headers={"Cache-Control": "public, max-age=300"})
    return JSONResponse(data, headers={"Cache-Control": "public, max-age=300"})


app.include_router(router)
# Defensive alias: if a platform ever strips the /api prefix before forwarding, the same
# handlers still resolve. Hidden from the OpenAPI schema.
_alias = APIRouter()
for _r in router.routes:
    _alias.add_api_route(_r.path[len("/api"):] or "/", _r.endpoint, methods=list(_r.methods),
                         include_in_schema=False, response_model=getattr(_r, "response_model", None))
app.include_router(_alias)
