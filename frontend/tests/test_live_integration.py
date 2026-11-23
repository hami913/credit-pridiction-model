"""Actual HTTP inference and Streamlit flows; opt in with CREDIT_TEST_LIVE=1.

Streamlit 1.51 AppTest has no file-upload action. These tests supply in-memory
uploaded-file objects at that UI boundary. CSV bytes, HTTP multipart requests,
FastAPI validation, the real saved model and rendering are all exercised.
"""
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from frontend.services.api import CreditAPI

pytestmark = pytest.mark.live
APP = Path(__file__).resolve().parents[1] / "app.py"


class UploadedCSV:
    def __init__(self, contents: bytes, name: str = "test_upload.csv"):
        self.contents = contents
        self.name = name

    def getvalue(self):
        return self.contents


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_live_api_client_preserves_predictions_and_supports_analytics(live_server):
    api = CreditAPI(live_server)
    assert api.health()["model_loaded"] is True
    info = api.model_info()
    assert (info["raw_feature_count"], info["engineered_feature_count"], info["processed_feature_count"]) == (23, 38, 41)
    template = api.template_csv()
    frame = pd.read_csv(BytesIO(template))
    row = frame[info["raw_features"]].iloc[0].to_dict()
    single = api.predict(row)
    batch = api.batch_csv(template)
    assert single["default_probability"] == pytest.approx(batch["results"][0]["default_probability"], abs=1e-14)
    scenarios = api.scenarios(row)
    current = next(item for item in scenarios["scenarios"] if item["factor"] == 1)
    assert current["default_probability"] == pytest.approx(single["default_probability"], abs=1e-14)
    explanation = api.explain(row)
    assert len(explanation["features"]) == 10
    assert all(np.isfinite(item["shap_value"]) for item in explanation["features"])
    assert len(api.feature_importance()["features"]) == 41
    assert api.model_metrics()["verified_in_this_integration"] is False
    sample = api.sample_csv(100)
    audit = api.audit_csv(sample)
    assert audit["total_customers"] == 100
    assert audit["regulatory_compliance"] == "not_assessed"
    monitoring = api.monitor_csv(sample, sample)
    assert monitoring["probability_drift"]["psi"] == pytest.approx(0.0, abs=1e-12)
    assert monitoring["actual_labels_available"] is True


def test_live_streamlit_single_scenarios_shap_and_model_page(live_server, monkeypatch):
    monkeypatch.setenv("CREDIT_API_BASE_URL", live_server)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    button(app, "Load sample history").click().run()
    button(app, "Run credit risk assessment").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.get("plotly_chart")) == 6
    result = app.session_state["assessment"]
    assert len(result["engineered_features"]) == 38
    expected = CreditAPI(live_server).predict(app.session_state["customer_data"])
    # Parallel Random Forest summation can differ at floating-point epsilon.
    assert result["default_probability"] == pytest.approx(expected["default_probability"], abs=1e-14, rel=0)
    assert len(app.session_state["scenarios"]["scenarios"]) == 5
    assert len(app.session_state["explanation"]["features"]) == 10
    app.switch_page("pages/2_Model_Information.py").run()
    assert not app.exception
    assert not app.error
    assert any(item.label == "ROC-AUC" for item in app.metric)


def test_live_streamlit_batch_upload_and_invalid_row_handling(live_server, monkeypatch):
    monkeypatch.setenv("CREDIT_API_BASE_URL", live_server)
    api = CreditAPI(live_server)
    sample = pd.read_csv(BytesIO(api.template_csv()))
    sample["Customer_ID"] = [f"{i:04}" for i in range(len(sample))]
    invalid = sample.iloc[[0]].copy()
    invalid["AGE"] = 5
    all_rows = pd.concat([sample, invalid], ignore_index=True)
    uploaded = UploadedCSV(all_rows.to_csv(index=False).encode())
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: uploaded if label == "Upload customer CSV" else None)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.switch_page("pages/1_Batch_Assessment.py").run()
    assert not app.exception
    assert any("Customer_ID" in item.value and item.value["Customer_ID"].iloc[0] == "0000" for item in app.dataframe)
    button(app, "🚀 Run batch scoring").click().run()
    assert not app.exception
    assert any("Input validation failed" in item.value for item in app.error)
    next(item for item in app.radio if item.label == "Invalid-row handling").set_value("Score valid rows and report skipped rows").run()
    button(app, "🚀 Run batch scoring").click().run()
    assert not app.exception
    assert not app.error
    response = app.session_state["batch_response"]
    assert response["scored_rows"] == len(sample)
    assert response["skipped_rows"] == 1
    assert response["results"][0]["Customer_ID"] == "0000"
    assert response["errors"][0]["row_number"] == len(sample) + 2


def test_live_streamlit_fairness_reporting_and_monitoring(live_server, monkeypatch):
    monkeypatch.setenv("CREDIT_API_BASE_URL", live_server)
    sample = CreditAPI(live_server).sample_csv(100)
    upload = UploadedCSV(sample, "uci_sample_100.csv")
    monkeypatch.setattr(st, "file_uploader", lambda label, **kwargs: upload)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    app.switch_page("pages/3_Analytics.py").run()
    assert not app.exception
    button(app, "Run fairness and validation audit").click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["audit_report"]["total_customers"] == 100
    button(app, "Run monitoring comparison").click().run()
    assert not app.exception
    assert not app.error
    report = app.session_state["monitoring_report"]
    assert report["probability_drift"]["psi"] == pytest.approx(0.0, abs=1e-12)
    assert report["actual_labels_available"] is True
    assert len(app.get("download_button")) >= 3
    # An edited audit setting invalidates the previous audit, while a changed
    # upload invalidates both reports rather than showing stale measurements.
    next(item for item in app.number_input if item.label == "Minimum group size").set_value(10).run()
    assert "audit_report" not in app.session_state
    assert "monitoring_report" in app.session_state
    upload.contents += b"\n"
    app.run()
    assert "monitoring_report" not in app.session_state
