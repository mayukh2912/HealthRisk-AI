import pytest
from fastapi.testclient import TestClient

from conftest import needs_artifacts

pytestmark = needs_artifacts

HEADER = ("record_id,time_in_hospital,num_lab_procedures,num_procedures,num_medications,number_outpatient,"
          "number_emergency,number_inpatient,number_diagnoses,gender,age,admission_type_id,discharge_disposition_id,"
          "admission_source_id,diag_1,max_glu_serum,A1Cresult,change,diabetesMed,insulin,metformin")
ROW = "{rid},3,40,0,10,0,0,{inp},5,Male,[50-60),1,1,7,414,None,None,No,Yes,Steady,No"


@pytest.fixture(scope="module")
def client():
    import main
    return TestClient(main.app)


def csv(n=3, extra=""):
    return "\n".join([HEADER] + [ROW.format(rid=f"r{i}", inp=i % 4) for i in range(n)]) + "\n" + extra


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_model_info(client):
    d = client.get("/api/model-info").json()
    for k in ("model_name", "model_version", "trained_at", "features", "metrics"):
        assert d[k]
    assert "test" in d["metrics"] and 0 < d["metrics"]["test"]["roc_auc"] <= 1


def test_predict_valid(client, valid_patient):
    r = client.post("/api/predict", json=valid_patient)
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["probability"] <= 1 and d["prediction"] in (0, 1)
    assert d["risk_category"] in ("Low", "Moderate", "High") and d["model_version"]
    ex = d["explanation"]
    assert ex["top_contributors"] and all(c["contribution"] > 0 for c in ex["increasing"])
    assert "not a diagnosis" in d["disclaimer"]


def test_predict_is_deterministic(client, valid_patient):
    a = client.post("/api/predict", json=valid_patient).json()["probability"]
    b = client.post("/api/predict", json=valid_patient).json()["probability"]
    assert a == b


@pytest.mark.parametrize("field,value", [("time_in_hospital", 0), ("time_in_hospital", 30), ("gender", "Other"),
                                         ("age", "75"), ("diag_1", "DROP TABLE"), ("insulin", "Maybe"),
                                         ("number_inpatient", -1), ("admission_type_id", 99)])
def test_predict_rejects_invalid_values(client, valid_patient, field, value):
    r = client.post("/api/predict", json={**valid_patient, field: value})
    assert r.status_code == 422
    assert any(e["field"] == field for e in r.json()["detail"])


def test_predict_rejects_missing_required(client, valid_patient):
    p = dict(valid_patient); p.pop("diag_1")
    r = client.post("/api/predict", json=p)
    assert r.status_code == 422 and r.json()["detail"][0]["field"] == "diag_1"


def test_predict_rejects_identifiers(client, valid_patient):
    r = client.post("/api/predict", json={**valid_patient, "patient_name": "x"})
    assert r.status_code == 422


def test_batch_predict_json_and_csv(client):
    r = client.post("/api/batch-predict", content=csv(5), headers={"content-type": "text/csv"})
    assert r.status_code == 200
    d = r.json()
    assert d["summary"]["rows_scored"] == 5 and len(d["predictions"]) == 5
    assert {p["record_id"] for p in d["predictions"]} == {f"r{i}" for i in range(5)}
    assert all(0 <= p["predicted_probability"] <= 1 for p in d["predictions"])
    r = client.post("/api/batch-predict?format=csv", content=csv(2), headers={"content-type": "text/csv"})
    lines = r.text.strip().splitlines()
    assert lines[0] == "record_id,predicted_probability,predicted_class,risk_category,model_version" and len(lines) == 3


def test_batch_reports_invalid_rows(client):
    bad = ROW.format(rid="bad", inp=0).replace(",3,40,", ",99,40,", 1)
    d = client.post("/api/batch-predict", content=csv(2) + bad + "\n", headers={"content-type": "text/csv"}).json()
    assert d["summary"]["rows_rejected"] == 1 and d["errors"][0]["record_id"] == "bad"


def test_batch_rejects_missing_columns_and_identifiers(client):
    r = client.post("/api/batch-predict", content="record_id,gender\nx,Male\n")
    assert r.status_code == 422 and "Missing required columns" in str(r.json())
    r = client.post("/api/batch-predict", content=csv(1).replace("record_id", "patient_name", 1))
    assert r.status_code == 422 and "identifiers" in str(r.json())


def test_batch_rejects_empty_and_oversized(client):
    assert client.post("/api/batch-predict", content="").status_code == 400
    assert client.post("/api/batch-predict", content=HEADER + "\n").status_code == 422
    assert client.post("/api/batch-predict", content=csv(5001)).status_code == 413


def test_analytics_has_no_row_level_data(client):
    d = client.get("/api/analytics").json()
    assert {"dataset", "eda", "evaluation", "explainability", "monitoring"} <= set(d)

    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield k
                yield from keys(v)
        elif isinstance(o, list):
            for v in o:
                yield from keys(v)
    assert not {"patient_nbr", "encounter_id"} & set(keys(d))  # no row-level identifiers anywhere
    assert client.get("/api/analytics?section=model").status_code == 200
    assert client.get("/api/analytics?section=nope").status_code == 404


def test_drift_endpoint(client):
    r = client.post("/api/monitoring/drift", content=csv(60))
    assert r.status_code == 200
    d = r.json()
    assert d["n_valid"] == 60 and all(f["psi"] >= 0 for f in d["features"])
    assert client.post("/api/monitoring/drift", content=csv(5)).status_code == 422
