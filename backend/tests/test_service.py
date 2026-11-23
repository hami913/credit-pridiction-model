"""Startup integrity, explicit failure states, and no-training guarantees."""
import hashlib
import json
import shutil

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier

from app.main import create_app
from app.services.prediction_service import InputDataError, PredictionService


ORIGINAL_MODEL_SHA256 = "5f7068b16bf1393578f7fd6e1b37e9c6905a1fca993e20c9775424738006c261"


def copy_artifacts(source, destination):
    shutil.copytree(source / "models", destination / "models")
    shutil.copytree(source / "config", destination / "config")
    return destination


def assert_unavailable(root, raw_customer):
    with TestClient(create_app(root)) as client:
        health = client.get("/api/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert client.post("/api/predict", json=raw_customer).status_code == 503
        for path in ["/api/model/info", "/api/model/metrics", "/api/model/feature-importance"]:
            assert client.get(path).status_code == 503
        assert client.get("/docs").status_code == 200


def test_original_model_bytes_and_metadata_are_consistent(backend_root, saved_artifact):
    model = backend_root / "models/credit_default_model_v1.joblib"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == ORIGINAL_MODEL_SHA256
    metadata = json.loads((backend_root / "models/model_metadata.json").read_text())
    assert metadata["model_sha256"] == ORIGINAL_MODEL_SHA256
    assert metadata["features"] == saved_artifact["features"]
    assert metadata["threshold"] == saved_artifact["threshold"] == 0.5
    assert metadata["target"] == saved_artifact["target"]
    assert metadata["model_name"] == saved_artifact["model_name"]
    assert metadata["model_version"] == saved_artifact["model_version"]


def test_service_loads_the_same_saved_classifier(backend_root, raw_customer):
    service = PredictionService(backend_root)
    service.load()
    assert service.model is not None
    result = service.predict(raw_customer)
    assert result["default_probability"] == pytest.approx(0.11318492128352413, abs=1e-12)


@pytest.mark.parametrize("probability,prediction,ui_band,score_level", [
    (0.0, 0, "Low", "Low"),
    (0.299999, 0, "Low", "Low"),
    (0.30, 0, "Moderate", "Medium"),
    (0.499999, 0, "Moderate", "Medium"),
    (0.50, 1, "High", "Medium"),
    (0.60, 1, "High", "High"),
    (1.0, 1, "High", "High"),
])
def test_distinct_original_classification_and_risk_boundaries(probability, prediction, ui_band, score_level):
    # Regression table copied from the original app's UI band and risk_scoring
    # rules. Binary threshold, UI band, and illustrative risk score differ.
    service = PredictionService()
    service.threshold = 0.5
    service.model_version = "1.0"
    summary = service._summary(probability)
    assert summary["prediction"] == prediction
    assert summary["risk_band"] == ui_band
    assert summary["risk_level"] == score_level
    assert summary["risk_index"] == round(probability * 100, 2)


def test_startup_and_multiple_requests_never_fit_or_reload(backend_root, raw_customer, monkeypatch):
    original_load = joblib.load
    loaded = []

    def record_load(*args, **kwargs):
        loaded.append(str(args[0]))
        return original_load(*args, **kwargs)

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Serving must never retrain the saved classifier")

    monkeypatch.setattr(joblib, "load", record_load)
    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden_fit)
    with TestClient(create_app(backend_root)) as client:
        for _ in range(2):
            assert client.post("/api/predict", json=raw_customer).status_code == 200
        assert client.get("/api/model/info").status_code == 200
    assert len(loaded) == 1


def test_missing_artifacts_leave_a_readable_health_endpoint(tmp_path, raw_customer):
    assert_unavailable(tmp_path, raw_customer)


@pytest.mark.parametrize("missing", ["credit_default_model_v1.joblib", "model_metadata.json"])
def test_each_missing_model_artifact_fails_closed(tmp_path, backend_root, raw_customer, missing):
    copy_artifacts(backend_root, tmp_path)
    (tmp_path / "models" / missing).unlink()
    assert_unavailable(tmp_path, raw_customer)


def test_corrupt_model_is_not_loaded_or_replaced(tmp_path, backend_root, raw_customer):
    copy_artifacts(backend_root, tmp_path)
    path = tmp_path / "models/credit_default_model_v1.joblib"
    path.write_bytes(b"not a serialized model")
    assert_unavailable(tmp_path, raw_customer)
    assert path.read_bytes() == b"not a serialized model"


@pytest.mark.parametrize("contents", ["{not-json", "[]", "null"])
def test_invalid_metadata_fails_closed(tmp_path, backend_root, raw_customer, contents):
    copy_artifacts(backend_root, tmp_path)
    (tmp_path / "models/model_metadata.json").write_text(contents)
    assert_unavailable(tmp_path, raw_customer)


@pytest.mark.parametrize("field,replacement", [
    ("model_sha256", "0" * 64),
    ("threshold", 0.25),
    ("target", "different_target"),
    ("model_version", "different_version"),
    ("sklearn_version", "0.0.0"),
    ("raw_feature_count", 22),
    ("engineered_feature_count", 37),
    ("processed_feature_count", 40),
    ("classes", [0, 1, 2]),
])
def test_metadata_mismatches_fail_closed(tmp_path, backend_root, raw_customer, field, replacement):
    copy_artifacts(backend_root, tmp_path)
    path = tmp_path / "models/model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = replacement
    path.write_text(json.dumps(metadata))
    assert_unavailable(tmp_path, raw_customer)


@pytest.mark.parametrize("group,key,value", [
    ("binary", "threshold", 0.9),
    ("risk_score", "medium_below", 0.99),
])
def test_inconsistent_risk_definitions_fail_closed(tmp_path, backend_root, raw_customer, group, key, value):
    copy_artifacts(backend_root, tmp_path)
    path = tmp_path / "models/model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata["risk_definitions"][group][key] = value
    path.write_text(json.dumps(metadata))
    assert_unavailable(tmp_path, raw_customer)


@pytest.mark.parametrize("field,value", [("SEX", 3), ("AGE", 1), ("PAY_AMT1", -5000)])
def test_direct_probability_service_rejects_invalid_customer_domain(app_client, raw_customer, field, value):
    raw_customer[field] = value
    with pytest.raises(InputDataError):
        app_client.app.state.service.predict_probabilities(pd.DataFrame([raw_customer]))


@pytest.mark.parametrize("field", ["raw_features", "features", "processed_features"])
def test_feature_order_mismatches_fail_closed(tmp_path, backend_root, raw_customer, field):
    copy_artifacts(backend_root, tmp_path)
    path = tmp_path / "models/model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata[field] = list(reversed(metadata[field]))
    path.write_text(json.dumps(metadata))
    assert_unavailable(tmp_path, raw_customer)


def test_prediction_does_not_require_historical_reports(tmp_path, backend_root, raw_customer):
    copy_artifacts(backend_root, tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health").status_code == 200
        response = client.post("/api/predict", json=raw_customer)
        assert response.status_code == 200
        assert response.json()["default_probability"] == pytest.approx(0.11318492128352413, abs=1e-12)
        # Like the professor's self-contained metadata artifact, the original
        # historical report is embedded with provenance; the reports directory
        # is therefore optional for inference and displaying supplied metrics.
        metrics = client.get("/api/model/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["verified_in_this_integration"] is False


def test_missing_historical_metrics_does_not_disable_predictions(tmp_path, backend_root, raw_customer):
    copy_artifacts(backend_root, tmp_path)
    path = tmp_path / "models/model_metadata.json"
    metadata = json.loads(path.read_text())
    metadata.pop("historical_report", None)
    path.write_text(json.dumps(metadata))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/health").status_code == 200
        assert client.post("/api/predict", json=raw_customer).status_code == 200
        assert client.get("/api/model/metrics").status_code == 503
