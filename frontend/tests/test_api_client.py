import json

import httpx
import pytest

from frontend.services.api import APIError, CreditAPI


def test_predict_passes_exact_raw_record_to_backend(customer):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"default_probability": 0.123456789})

    client = CreditAPI("http://backend.test", transport=httpx.MockTransport(handler))
    result = client.predict(customer)
    assert json.loads(captured[0].content) == customer
    assert captured[0].url.path == "/api/predict"
    assert result["default_probability"] == 0.123456789


@pytest.mark.parametrize("name,path", [("health", "/api/health"), ("model_info", "/api/model/info"), ("model_metrics", "/api/model/metrics"), ("feature_importance", "/api/model/feature-importance")])
def test_read_endpoints_use_configured_base(name, path):
    def handler(request):
        assert request.url.path == path
        return httpx.Response(200, json={"ok": True})

    assert getattr(CreditAPI("http://backend.test/api/", transport=httpx.MockTransport(handler)), name)() == {"ok": True}


def test_scenario_and_explanation_request_contract(customer):
    requests = []
    client = CreditAPI("http://backend.test", transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(200, json={})))
    client.scenarios(customer, [0, 1, 2])
    client.explain(customer, 8)
    assert json.loads(requests[0].content) == {"customer": customer, "factors": [0, 1, 2]}
    assert requests[0].url.path == "/api/scenario"
    assert json.loads(requests[1].content) == {"customer": customer, "top_n": 8}
    assert requests[1].url.path == "/api/explain"


def test_batch_upload_retains_bytes_and_policy():
    captured = []
    csv = b"Customer_ID,LIMIT_BAL\naccount-1,100000\n"
    client = CreditAPI("http://backend.test", transport=httpx.MockTransport(lambda request: captured.append(request) or httpx.Response(200, json={"scored_rows": 1})))
    assert client.batch_csv(csv, filename="my_upload.csv", skip_invalid=True)["scored_rows"] == 1
    request = captured[0]
    assert request.url.params["skip_invalid"] == "true"
    assert csv in request.content
    assert b'filename="my_upload.csv"' in request.content


def test_analytics_upload_contract():
    captured = []
    client = CreditAPI("http://backend.test", transport=httpx.MockTransport(lambda request: captured.append(request) or httpx.Response(200, json={})))
    client.audit_csv(b"audit-data", group_column="EDUCATION", min_group_size=12, data_source="review.csv")
    client.monitor_csv(b"reference-data", b"current-data")
    assert captured[0].url.path == "/api/analytics/audit"
    assert all(value in captured[0].content for value in (b"EDUCATION", b"12", b"review.csv", b"audit-data"))
    assert captured[1].url.path == "/api/analytics/monitor"
    assert all(value in captured[1].content for value in (b'name="reference_file"', b'name="current_file"', b"reference-data", b"current-data"))


def test_template_and_sample_return_download_bytes():
    captured = []
    client = CreditAPI("http://backend.test", transport=httpx.MockTransport(lambda request: captured.append(request) or httpx.Response(200, content=b"a,b\n1,2\n")))
    assert client.template_csv() == b"a,b\n1,2\n"
    assert client.sample_csv(50) == b"a,b\n1,2\n"
    assert captured[1].url.params["limit"] == "50"


def test_validation_error_keeps_structured_row_details():
    detail = [{"loc": ["body", "PAY_0"], "msg": "Invalid repayment status", "type": "value_error"}]
    client = CreditAPI("http://backend.test", transport=httpx.MockTransport(lambda _: httpx.Response(422, json={"detail": detail})))
    with pytest.raises(APIError, match="body.PAY_0") as caught:
        client.predict({})
    assert caught.value.status_code == 422
    assert caught.value.detail == detail


def test_missing_backend_has_actionable_error():
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(APIError, match="Start the backend"):
        CreditAPI("http://backend.test", transport=httpx.MockTransport(handler)).health()


def test_timeout_has_actionable_error():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(APIError, match="CREDIT_API_TIMEOUT"):
        CreditAPI("http://backend.test", transport=httpx.MockTransport(handler)).health()


def test_invalid_json_is_not_reported_as_a_prediction():
    with pytest.raises(APIError, match="invalid JSON"):
        CreditAPI("http://backend.test", transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"oops"))).predict({})


@pytest.mark.parametrize("value", ["file:///tmp/model", "backend.test", "ftp://backend.test"])
def test_invalid_api_configuration(value):
    with pytest.raises(APIError):
        CreditAPI(value)
