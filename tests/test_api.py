import copy
import json
from pathlib import Path
import shutil
import pytest
from fastapi.testclient import TestClient
from esp_pdm.api import create_app
from esp_pdm.io import read_json
from esp_pdm.model import RiskModel

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def payload():
    return read_json(ROOT / "examples/request.json")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(ROOT / "models")) as connection:
        yield connection


def test_health_schema_and_offline_prediction_parity(client, payload):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["input_stage"] == "normalized_BASIC_window"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    expected = RiskModel(ROOT / "models").predict(**payload)
    assert response.json() == expected
    assert response.json()["horizon_hours"] == 168
    assert client.get("/openapi.json").status_code == 200


@pytest.mark.parametrize("change", ["rows", "columns", "order", "id", "string", "boolean", "extra", "overflow"])
def test_invalid_requests_are_422(client, payload, change):
    body = copy.deepcopy(payload)
    if change == "rows":
        body["samples"].pop()
    elif change == "columns":
        body["samples"][0].pop()
    elif change == "order":
        body["feature_names"] = list(reversed(body["feature_names"]))
    elif change == "id":
        body["preprocessing_id"] = "different"
    elif change == "extra":
        body["raw_sensor_values"] = []
    else:
        body["samples"][0][0] = {"string": "0", "boolean": True, "overflow": 1e300}[change]
    assert client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_returns_422_without_error_handler_crash(client, payload, invalid):
    body = copy.deepcopy(payload)
    body["samples"][0][0] = invalid
    response = client.post("/predict", content=json.dumps(body), headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_artifact_corruption_fails_startup(tmp_path):
    target = tmp_path / "models"
    shutil.copytree(ROOT / "models", target)
    artifact = target / "champion.joblib"
    artifact.write_bytes(artifact.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="failed validation"):
        with TestClient(create_app(target)):
            pass


def test_model_reload_is_deterministic(payload):
    assert RiskModel(ROOT / "models").predict(**payload) == RiskModel(ROOT / "models").predict(**payload)
