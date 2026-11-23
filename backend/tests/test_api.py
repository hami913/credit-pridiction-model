"""HTTP contract, complete inference flows, and malformed-input handling.

The tests exercise the shipped Random Forest. Small labeled UCI samples verify
analytics plumbing; their computed metrics are not held-out performance claims.
"""
import io
import json

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import prepare_model_input
from src.evaluate import evaluate_model
from src.risk_scoring import assess_risk


RAW_FEATURES = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    *[f"BILL_AMT{i}" for i in range(1, 7)],
    *[f"PAY_AMT{i}" for i in range(1, 7)],
]


def original_probabilities(rows, artifact):
    frame = prepare_model_input(pd.DataFrame(rows), artifact["features"])
    return artifact["pipeline"].predict_proba(frame)[:, 1]


def upload_csv(client, frame, endpoint="/api/predict/batch-csv", **kwargs):
    return client.post(
        endpoint,
        files={"file": ("customers.csv", frame.to_csv(index=False).encode(), "text/csv")},
        **kwargs,
    )


def test_health_openapi_and_model_contract(app_client, saved_artifact):
    health = app_client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"
    assert health.json()["model_loaded"] is True
    assert health.json()["threshold"] == saved_artifact["threshold"] == 0.5
    info = app_client.get("/api/model/info")
    assert info.status_code == 200
    body = info.json()
    assert body["raw_features"] == RAW_FEATURES
    assert body["features"] == saved_artifact["features"]
    assert body["raw_feature_count"] == 23
    assert body["engineered_feature_count"] == 38
    assert body["processed_feature_count"] == 41
    assert body["classes"] == [0, 1]
    assert body["algorithm"] == "RandomForestClassifier"
    assert body["sklearn_version"] == "1.6.1"
    assert body["classifier_parameters"]["n_estimators"] == 400
    assert body["target"] == "default.payment.next.month"
    assert body["metadata_provenance"]
    assert body["artifact_metadata"]["threshold"] == 0.5
    assert app_client.get("/docs").status_code == 200
    contract = app_client.get("/openapi.json").json()
    required = {"/api/health", "/api/predict", "/api/predict/batch",
                "/api/predict/batch-csv", "/api/scenario", "/api/explain",
                "/api/model/info", "/api/model/metrics",
                "/api/model/feature-importance", "/api/analytics/audit",
                "/api/analytics/monitor", "/api/data/template", "/api/data/sample"}
    assert required <= set(contract["paths"])


def test_historical_metrics_are_labeled_and_unchanged(app_client, backend_root):
    supplied = json.loads((backend_root / "reports/credit_risk_regulatory_report.json").read_text())
    response = app_client.get("/api/model/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["source"]
    assert body["verified_in_this_integration"] is False
    assert body["report"] == supplied
    mapping = {"accuracy": "Test_Accuracy", "precision": "Test_Precision",
               "recall": "Test_Recall", "f1": "Test_F1",
               "roc_auc": "Test_ROC_AUC", "pr_auc": "Test_PR_AUC"}
    assert body["metrics"] == {key: supplied[source] for key, source in mapping.items()}


def test_single_prediction_matches_saved_original_logic(app_client, raw_customer, saved_artifact):
    response = app_client.post("/api/predict", json=raw_customer)
    assert response.status_code == 200, response.text
    result = response.json()
    expected = float(original_probabilities([raw_customer], saved_artifact)[0])
    assert expected == pytest.approx(0.11318492128352413, abs=1e-12)
    assert result["default_probability"] == pytest.approx(expected, abs=1e-12)
    assert result["default_probability_rounded"] == round(expected, 4)
    assert result["default_probability_percent"] == round(expected * 100, 2)
    assert result["non_default_probability"] == pytest.approx(1 - expected)
    assert result["prediction"] == int(expected >= saved_artifact["threshold"])
    assert result["prediction_label"] == "NO DEFAULT / LOWER RISK"
    assert result["risk_band"] == "Low"
    assert result["risk_index"] == assess_risk(expected)["risk_index"]
    assert result["risk_level"] == assess_risk(expected)["risk_level"]
    assert len(result["engineered_features"]) == 38
    assert list(result["engineered_features"]) == saved_artifact["features"]
    expected_features = prepare_model_input(pd.DataFrame([raw_customer]), saved_artifact["features"])
    for name, value in result["engineered_features"].items():
        assert value == pytest.approx(expected_features[name].iloc[0])
    assert result["indicators"]["avg_utilization"] == pytest.approx(0.1)
    assert result["indicators"]["avg_payment"] == 5000
    assert result["indicators"]["delayed_months"] == 0


@pytest.mark.parametrize("feature", RAW_FEATURES)
def test_every_raw_input_is_required(app_client, raw_customer, feature):
    raw_customer.pop(feature)
    response = app_client.post("/api/predict", json=raw_customer)
    assert response.status_code == 422


@pytest.mark.parametrize("name,value", [
    ("LIMIT_BAL", 0), ("LIMIT_BAL", -1), ("LIMIT_BAL", "200000"),
    ("LIMIT_BAL", True), ("BILL_AMT1", True), ("BILL_AMT1", "20000"),
    ("SEX", True), ("SEX", 0), ("SEX", 3), ("SEX", 1.5),
    ("EDUCATION", -1), ("EDUCATION", 7), ("MARRIAGE", 4),
    ("AGE", 17), ("AGE", 101), ("AGE", 25.5), ("AGE", "30"),
    ("PAY_0", -3), ("PAY_2", 9), ("PAY_3", 1.5), ("PAY_4", False),
    ("PAY_AMT1", -1), ("PAY_AMT2", None), ("PAY_AMT3", "5000"),
])
def test_strict_input_validation(app_client, raw_customer, name, value):
    raw_customer[name] = value
    assert app_client.post("/api/predict", json=raw_customer).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_has_safe_validation_response(app_client, raw_customer, value):
    raw_customer["BILL_AMT1"] = value
    response = app_client.post("/api/predict", content=json.dumps(raw_customer),
                               headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    json.dumps(response.json(), allow_nan=False)


def test_extra_single_input_is_forbidden(app_client, raw_customer):
    assert app_client.post("/api/predict", json={**raw_customer, "ID": "001"}).status_code == 422


def test_original_public_route_preserves_rounded_probability(app_client, raw_customer):
    response = app_client.post("/predict", json=raw_customer)
    canonical = app_client.post("/api/predict", json=raw_customer)
    assert response.status_code == canonical.status_code == 200
    assert response.json()["default_probability"] == round(canonical.json()["default_probability"], 4)
    assert response.json()["prediction"] == canonical.json()["prediction"]
    assert app_client.get("/health").json() == app_client.get("/api/health").json()


@pytest.mark.parametrize("education", [0, 5, 6])
def test_original_category_mapping_is_accepted(app_client, raw_customer, education):
    raw_customer.update(EDUCATION=education, MARRIAGE=0)
    response = app_client.post("/api/predict", json=raw_customer)
    reference = app_client.post("/api/predict", json={**raw_customer, "EDUCATION": 4, "MARRIAGE": 3})
    assert response.status_code == reference.status_code == 200
    assert response.json()["default_probability"] == pytest.approx(reference.json()["default_probability"], abs=1e-12)
    assert response.json()["engineered_features"]["EDUCATION"] == 4
    assert response.json()["engineered_features"]["MARRIAGE"] == 3


def test_negative_bills_and_negative_repayment_codes_are_preserved(app_client, raw_customer, saved_artifact):
    raw_customer.update({f"BILL_AMT{i}": -5000 for i in range(1, 7)})
    raw_customer.update(PAY_0=-2, PAY_2=-1)
    response = app_client.post("/api/predict", json=raw_customer)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["default_probability"] == pytest.approx(original_probabilities([raw_customer], saved_artifact)[0], abs=1e-12)
    assert result["engineered_features"]["PAY_TO_BILL_RATIO"] == 0
    assert result["indicators"]["avg_delay"] == -0.5
    assert result["indicators"]["avg_utilization"] < 0


def test_all_five_scenarios_match_original_inference(app_client, raw_customer, saved_artifact):
    response = app_client.post("/api/scenario", json={"customer": raw_customer})
    assert response.status_code == 200, response.text
    result = response.json()
    scenarios = result["scenarios"]
    assert [row["label"] for row in scenarios] == ["No payments", "Half", "Current", "+50%", "Double"]
    assert [row["factor"] for row in scenarios] == [0, 0.5, 1, 1.5, 2]
    changed = [{**raw_customer, **{f"PAY_AMT{i}": raw_customer[f"PAY_AMT{i}"] * factor for i in range(1, 7)}}
               for factor in [0, 0.5, 1, 1.5, 2]]
    expected = original_probabilities(changed, saved_artifact)
    np.testing.assert_allclose([r["default_probability"] for r in scenarios], expected, atol=1e-12, rtol=0)
    assert [r["prediction"] for r in scenarios] == (expected >= 0.5).astype(int).tolist()
    assert result["threshold"] == 0.5


@pytest.mark.parametrize("factors", [[-1], [True], ["2"], []])
def test_invalid_scenario_factors_rejected(app_client, raw_customer, factors):
    assert app_client.post("/api/scenario", json={"customer": raw_customer, "factors": factors}).status_code == 422


def test_batch_json_parity_order_and_passthrough(app_client, raw_customer, saved_artifact):
    rows = [{**raw_customer, "ID": "0002", "notes": "First customer", "actual_default": 0},
            {**raw_customer, "ID": "0001", "PAY_0": 4, "PAY_AMT1": 0, "notes": "Second customer", "actual_default": 1}]
    response = app_client.post("/api/predict/batch", json={"customers": rows})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["total_rows"], body["scored_rows"], body["skipped_rows"]) == (2, 2, 0)
    assert body["errors"] == []
    assert [row["row_number"] for row in body["results"]] == [2, 3]
    assert [row["ID"] for row in body["results"]] == ["0002", "0001"]
    assert [row["notes"] for row in body["results"]] == ["First customer", "Second customer"]
    np.testing.assert_allclose([row["default_probability"] for row in body["results"]], original_probabilities(rows, saved_artifact), atol=1e-12, rtol=0)


def test_batch_csv_matches_json_and_preserves_identifiers(app_client, raw_customer):
    frame = pd.DataFrame([{**raw_customer, "ID": "0007"}, {**raw_customer, "ID": "NA", "PAY_0": 3}])
    csv_response = upload_csv(app_client, frame)
    json_response = app_client.post("/api/predict/batch", json={"customers": frame.to_dict("records")})
    assert csv_response.status_code == json_response.status_code == 200
    csv_rows, json_rows = csv_response.json()["results"], json_response.json()["results"]
    assert [r["ID"] for r in csv_rows] == ["0007", "NA"]
    np.testing.assert_allclose([r["default_probability"] for r in csv_rows], [r["default_probability"] for r in json_rows], atol=1e-12, rtol=0)


def test_csv_blank_lines_preserve_source_row_numbers(app_client, raw_customer):
    rows = [raw_customer, {**raw_customer, "LIMIT_BAL": 0}, raw_customer]
    lines = pd.DataFrame(rows).to_csv(index=False).splitlines()
    # Header line 1; valid line 2; blank line 3; invalid line 4; valid line 5.
    content = "\n".join([lines[0], lines[1], "", lines[2], lines[3]]) + "\n"
    response = app_client.post("/api/predict/batch-csv?skip_invalid=true", files={
        "file": ("blank_lines.csv", content, "text/csv"),
    })
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_rows"] == 3
    assert [row["row_number"] for row in result["results"]] == [2, 5]
    assert [row["row_number"] for row in result["errors"]] == [4]


def test_csv_quoted_multiline_fields_preserve_record_start_rows(app_client, raw_customer):
    rows = [{**raw_customer, "notes": "first line\nsecond line"},
            {**raw_customer, "notes": "invalid", "LIMIT_BAL": 0},
            {**raw_customer, "notes": "last"}]
    # First quoted record spans physical lines 2–3; later records begin 4 and 5.
    response = upload_csv(app_client, pd.DataFrame(rows), params={"skip_invalid": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["total_rows"] == 3
    assert [row["row_number"] for row in result["results"]] == [2, 5]
    assert [row["row_number"] for row in result["errors"]] == [4]
    assert result["results"][0]["notes"] == "first line\nsecond line"


@pytest.mark.parametrize("csv", [False, True])
def test_batch_strict_and_explicit_skip_modes(app_client, raw_customer, csv):
    rows = [{**raw_customer, "ID": "good"}, {**raw_customer, "ID": "bad", "LIMIT_BAL": 0}]
    def send(skip):
        if csv:
            return upload_csv(app_client, pd.DataFrame(rows), params={"skip_invalid": str(skip).lower()})
        return app_client.post("/api/predict/batch", json={"customers": rows, "skip_invalid": skip})
    assert send(False).status_code == 422
    response = send(True)
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["total_rows"], body["scored_rows"], body["skipped_rows"]) == (2, 1, 1)
    assert body["results"][0]["ID"] == "good"
    assert body["errors"][0]["row_number"] == 3
    assert body["errors"][0]["errors"]
    rows[:] = [rows[1]]
    assert send(True).status_code == 422


def test_batch_json_rejects_numeric_strings(app_client, raw_customer):
    raw_customer["LIMIT_BAL"] = "200000"
    assert app_client.post("/api/predict/batch", json={"customers": [raw_customer]}).status_code == 422


def test_batch_preserves_columns_colliding_with_outputs(app_client, raw_customer):
    original = {**raw_customer, "ID": "009", "prediction": "legacy", "default_probability": "provided",
                "row_number": "original-row", "input__prediction": "already-present"}
    response = app_client.post("/api/predict/batch", json={"customers": [original]})
    assert response.status_code == 200, response.text
    body = response.json()
    result, renames = body["results"][0], body["renamed_input_columns"]
    for column in ["prediction", "default_probability", "row_number"]:
        assert column in renames
        assert result[renames[column]] == original[column]
    assert result["input__prediction"] == "already-present"
    assert result["prediction"] in [0, 1]
    assert 0 <= result["default_probability"] <= 1


@pytest.mark.parametrize("content", [b"", b"foo,bar\n1,2\n", b"1,2,3\n4,5,6\n", b'"unterminated\n', b"\xff\xfe\x00bad"])
def test_invalid_csv_rejected_cleanly(app_client, content):
    response = app_client.post("/api/predict/batch-csv", files={"file": ("bad.csv", content, "text/csv")})
    assert response.status_code == 422, response.text
    json.dumps(response.json(), allow_nan=False)


def test_header_only_and_duplicate_header_csv_rejected(app_client, raw_customer):
    no_rows = pd.DataFrame(columns=RAW_FEATURES)
    assert upload_csv(app_client, no_rows).status_code == 422
    content = pd.DataFrame([raw_customer]).to_csv(index=False)
    lines = content.splitlines()
    duplicate = lines[0] + ",LIMIT_BAL\n" + lines[1] + ",1\n"
    assert app_client.post("/api/predict/batch-csv", files={"file": ("duplicate.csv", duplicate, "text/csv")}).status_code == 422


def test_batch_row_limit_and_upload_byte_limit(app_client, raw_customer, monkeypatch):
    import app.api.routes as routes
    import app.services.prediction_service as service_module

    monkeypatch.setattr(service_module, "MAX_BATCH_ROWS", 1)
    response = app_client.post("/api/predict/batch", json={"customers": [raw_customer, raw_customer]})
    assert response.status_code in (413, 422)
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 8)
    response = app_client.post("/api/predict/batch-csv", files={"file": ("large.csv", b"x" * 9, "text/csv")})
    assert response.status_code == 413


def test_empty_json_batch_is_rejected(app_client):
    assert app_client.post("/api/predict/batch", json={"customers": []}).status_code == 422


def test_feature_importance_is_from_the_actual_random_forest(app_client, saved_artifact):
    response = app_client.get("/api/model/feature-importance")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["method"]
    features = body["features"]
    assert len(features) == 41
    assert len(body["aggregated_features"]) == 38
    assert sum(row["importance"] for row in features) == pytest.approx(1.0)
    assert sum(row["importance"] for row in body["aggregated_features"]) == pytest.approx(1.0)
    actual = saved_artifact["pipeline"].named_steps["classifier"].feature_importances_
    np.testing.assert_allclose(sorted(r["importance"] for r in features), sorted(actual), atol=1e-12, rtol=0)


def test_shap_explanation_has_real_model_probability(app_client, raw_customer):
    response = app_client.post("/api/explain", json={"customer": raw_customer, "top_n": 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert "SHAP" in body["method"].upper()
    assert body["positive_class"] == 1
    assert len(body["features"]) == 5
    assert body["default_probability"] == pytest.approx(0.11318492128352413, abs=1e-12)
    impacts = [r["absolute_impact"] for r in body["features"]]
    assert impacts == sorted(impacts, reverse=True)
    assert all(np.isfinite(r["shap_value"]) and r["absolute_impact"] == pytest.approx(abs(r["shap_value"])) for r in body["features"])


@pytest.mark.parametrize("group", ["SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP"])
def test_audit_fairness_and_reporting_with_observed_labels(app_client, labeled_uci_sample, saved_artifact, group):
    response = upload_csv(app_client, labeled_uci_sample, "/api/analytics/audit", data={"group_column": group, "min_group_size": "5", "data_source": "uci_functionality_test"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report_type"] == "credit_risk_model_audit"
    assert body["total_customers"] == len(labeled_uci_sample)
    assert body["data_source"] == "uci_functionality_test"
    assert body["regulatory_compliance"] == "not_assessed"
    assert body["fairness_analysis"]["group_column"] == group
    assert body["fairness_analysis"]["groups"]
    assert body["limitations"]
    probabilities = original_probabilities(labeled_uci_sample.to_dict("records"), saved_artifact)
    expected = evaluate_model(labeled_uci_sample["default.payment.next.month"], probabilities, threshold=0.5)
    for name, value in expected.items():
        if isinstance(value, (int, float)):
            assert body["model_evaluation"][name] == pytest.approx(value)
        else:
            assert body["model_evaluation"][name] == value
    json.dumps(body, allow_nan=False)


def test_audit_requires_labels_and_valid_group(app_client, labeled_uci_sample):
    no_labels = labeled_uci_sample.drop(columns=["default.payment.next.month"])
    assert upload_csv(app_client, no_labels, "/api/analytics/audit").status_code == 422
    bad_labels = labeled_uci_sample.copy()
    bad_labels.loc[0, "default.payment.next.month"] = 2
    assert upload_csv(app_client, bad_labels, "/api/analytics/audit").status_code == 422
    assert upload_csv(app_client, labeled_uci_sample, "/api/analytics/audit", data={"group_column": "ID"}).status_code == 422


def test_audit_rejects_conflicting_observed_label_columns(app_client, labeled_uci_sample):
    conflicting = labeled_uci_sample.copy()
    conflicting["actual_default"] = 1 - conflicting["default.payment.next.month"]
    assert upload_csv(app_client, conflicting, "/api/analytics/audit").status_code == 422


def test_age_audit_preserves_original_notebook_boundary_groups(app_client, raw_customer):
    # Notebook cell 162 uses right-closed intervals (0,30], (30,40], ...,
    # (60,inf]. Preserve them even though the original final label is "60+".
    ages = [30, 31, 40, 41, 50, 51, 60, 61]
    frame = pd.DataFrame([
        {**raw_customer, "AGE": age, "actual_default": index % 2}
        for index, age in enumerate(ages)
    ])
    response = upload_csv(app_client, frame, "/api/analytics/audit", data={
        "group_column": "AGE_GROUP", "min_group_size": "1",
        "data_source": "synthetic_age_boundary_regression",
    })
    assert response.status_code == 200, response.text
    groups = response.json()["fairness_analysis"]["groups"]
    assert {group["group"]: group["sample_size"] for group in groups} == {
        "≤30": 1, "31-40": 2, "41-50": 2, "51-60": 2, "60+": 1,
    }


@pytest.mark.parametrize("labels", [False, True])
def test_monitoring_same_population_and_optional_observed_labels(app_client, labeled_uci_sample, labels):
    reference = labeled_uci_sample.drop(columns=["default.payment.next.month"])
    current = labeled_uci_sample if labels else reference
    response = app_client.post("/api/analytics/monitor", files={
        "reference_file": ("reference.csv", reference.to_csv(index=False), "text/csv"),
        "current_file": ("current.csv", current.to_csv(index=False), "text/csv"),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report_type"] == "credit_risk_model_monitoring"
    assert body["probability_drift"]["reference_count"] == len(reference)
    assert body["probability_drift"]["current_count"] == len(current)
    assert body["probability_drift"]["psi"] == pytest.approx(0.0, abs=1e-12)
    assert body["probability_drift"]["drift_status"] == "low"
    assert body["actual_labels_available"] is labels
    assert (body["performance"] is not None) is labels
    assert body["limitations"]
    json.dumps(body, allow_nan=False)


def test_download_template_and_restored_data_sample(app_client):
    template = app_client.get("/api/data/template")
    assert template.status_code == 200
    assert "attachment" in template.headers["content-disposition"]
    frame = pd.read_csv(io.StringIO(template.text))
    assert len(frame) == 2 and set(RAW_FEATURES) <= set(frame.columns)
    assert upload_csv(app_client, frame).status_code == 200
    sample = app_client.get("/api/data/sample", params={"limit": 12})
    assert sample.status_code == 200
    restored = pd.read_csv(io.StringIO(sample.text))
    assert len(restored) == 12
    assert {"ID", "default.payment.next.month", *RAW_FEATURES} == set(restored.columns)
    assert restored["default.payment.next.month"].isin([0, 1]).all()
    assert app_client.get("/api/data/sample", params={"limit": 1001}).status_code == 422
