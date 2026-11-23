"""Synthetic UI fixtures and an opt-in real HTTP backend fixture."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FEATURES = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] + [f"BILL_AMT{i}" for i in range(1, 7)] + [f"PAY_AMT{i}" for i in range(1, 7)]


@pytest.fixture
def customer():
    result = {key: 0 for key in RAW_FEATURES}
    result.update(LIMIT_BAL=200000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=30)
    for i in range(1, 7):
        result[f"BILL_AMT{i}"] = 10000
        result[f"PAY_AMT{i}"] = 2000
    return result


class FakeAPI:
    """Explicitly synthetic server responses for UI behavior tests only."""
    def __init__(self):
        self.calls = []

    def model_info(self):
        return {
            "model_name": "Synthetic test response", "model_version": "test-only", "threshold": 0.5,
            "raw_feature_count": 23, "engineered_feature_count": 38, "processed_feature_count": 41,
            "raw_features": RAW_FEATURES, "features": RAW_FEATURES + [f"test_derived_{i}" for i in range(15)],
            "target": "default.payment.next.month", "model_sha256": "test-only", "risk_definitions": {},
        }

    def predict(self, customer):
        self.calls.append(("predict", deepcopy(customer)))
        return {
            "default_probability": 0.4, "default_probability_percent": 40.0, "non_default_probability": 0.6,
            "prediction": 0, "prediction_label": "NO DEFAULT / LOWER RISK", "threshold": 0.5,
            "model_version": "test-only", "risk_band": "Moderate", "risk_index": 40.0, "risk_level": "Medium",
            "indicators": {"avg_utilization": 0.2, "max_utilization": 0.3, "avg_delay": 0.0, "max_delay": 0,
                           "delayed_months": 0, "avg_bill": 40000.0, "avg_payment": 5000.0, "pay_to_bill": 0.125, "pay_to_limit": 0.025},
            "engineered_features": {**customer, "test_derived": 0.2},
        }

    def scenarios(self, customer):
        self.calls.append(("scenario", deepcopy(customer)))
        return {"threshold": 0.5, "model_version": "test-only", "scenarios": [
            {"label": label, "factor": factor, "default_probability_percent": value, "default_probability": value / 100, "risk_band": "High" if value >= 50 else "Moderate"}
            for label, factor, value in [("No payments", 0, 55), ("Half", .5, 45), ("Current", 1, 40), ("+50%", 1.5, 36), ("Double", 2, 34)]
        ]}

    def explain(self, customer):
        self.calls.append(("explain", deepcopy(customer)))
        return {"features": [{"feature": "PAY_0", "shap_value": 0.02, "absolute_impact": 0.02}, {"feature": "LIMIT_BAL", "shap_value": -0.01, "absolute_impact": 0.01}], "note": "Synthetic test response only."}

    def template_csv(self):
        return (",".join(RAW_FEATURES) + "\n").encode()

    def sample_csv(self, limit):
        self.calls.append(("sample", limit))
        return b"ID,actual_default\n1,0\n"

    def model_metrics(self):
        return {"source": "synthetic test fixture", "verified_in_this_integration": False,
                "metrics": {key: 0.5 for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
                "report": {"data_source": "synthetic test fixture"}}

    def feature_importance(self):
        return {"method": "synthetic test fixture", "features": [{"feature": f"processed_{i}", "importance": 1 / 41} for i in range(41)],
                "aggregated_features": [{"feature": f"input_{i}", "importance": 1 / 38} for i in range(38)]}


@pytest.fixture
def fake_api(monkeypatch):
    import frontend.components.common as common
    fake = FakeAPI()
    monkeypatch.setattr(common, "CreditAPI", lambda: fake)
    return fake


@pytest.fixture(scope="session")
def live_server(tmp_path_factory):
    """Start FastAPI in the same network namespace as these tests."""
    if os.getenv("CREDIT_TEST_LIVE") != "1":
        pytest.skip("Set CREDIT_TEST_LIVE=1 to execute the real backend/Streamlit integration tests.")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path_factory.mktemp("live_backend") / "server.log"
    with log_path.open("w+") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--app-dir", str(PROJECT_ROOT / "backend"), "--host", "127.0.0.1", "--port", str(port)],
            cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log.flush()
                    pytest.fail("Backend exited during startup:\n" + log_path.read_text())
                try:
                    if httpx.get(f"{base_url}/api/health", timeout=1, trust_env=False).status_code == 200:
                        break
                except httpx.RequestError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("Backend was not healthy within 45 seconds:\n" + log_path.read_text())
            yield base_url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
