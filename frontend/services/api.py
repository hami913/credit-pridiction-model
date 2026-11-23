"""The only HTTP boundary used by the Streamlit frontend.

No model or backend source is imported here. The server owns validation,
feature engineering, inference, risk definitions and analytics calculations.
"""
from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx


class APIError(RuntimeError):
    """Actionable error from the backend or its network connection."""

    def __init__(self, message: str, *, status_code: int | None = None, detail: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def _detail_text(detail: Any) -> str:
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        lines = []
        for item in detail[:10]:
            if isinstance(item, dict):
                location = ".".join(str(value) for value in item.get("loc", []))
                lines.append(f"{location}: {item.get('msg', str(item))}".lstrip(": "))
            else:
                lines.append(str(item))
        if len(detail) > 10:
            lines.append(f"… and {len(detail) - 10} more validation errors.")
        return "\n".join(lines)
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("detail") or detail.get("error")
        return str(message) if message else str(detail)
    return str(detail)


class CreditAPI:
    """Synchronous client, suitable for Streamlit and independent tests."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        value = (base_url or os.getenv("CREDIT_API_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise APIError("CREDIT_API_BASE_URL must be a complete http:// or https:// address.")
        self.base_url = value
        self.api_url = value if value.endswith("/api") else f"{value}/api"
        try:
            seconds = float(timeout if timeout is not None else os.getenv("CREDIT_API_TIMEOUT", "120"))
        except (TypeError, ValueError) as exc:
            raise APIError("CREDIT_API_TIMEOUT must be a positive number of seconds.") from exc
        if seconds <= 0 or seconds == float("inf") or seconds != seconds:
            raise APIError("CREDIT_API_TIMEOUT must be a positive finite number of seconds.")
        self.timeout = httpx.Timeout(seconds, connect=min(seconds, 5.0))
        self.transport = transport

    def _request(self, method: str, path: str, *, binary: bool = False, **kwargs: Any) -> Any:
        try:
            # The configured backend is contacted directly. Unrelated shell
            # proxy settings must not redirect localhost application requests.
            with httpx.Client(timeout=self.timeout, transport=self.transport, trust_env=False) as client:
                response = client.request(method, f"{self.api_url}{path}", **kwargs)
        except httpx.TimeoutException as exc:
            raise APIError("The backend timed out. Retry the operation or increase CREDIT_API_TIMEOUT.") from exc
        except httpx.RequestError as exc:
            raise APIError(
                f"Cannot connect to the FastAPI backend at {self.base_url}. "
                "Start the backend using the README instructions and check CREDIT_API_BASE_URL."
            ) from exc

        if not response.is_success:
            try:
                payload = response.json()
            except ValueError:
                payload = response.text[:1000] or response.reason_phrase
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            prefix = "Input validation failed" if response.status_code == 422 else "Backend request failed"
            if response.status_code == 503:
                prefix = "The backend model is unavailable"
            raise APIError(f"{prefix} ({response.status_code}): {_detail_text(detail)}", status_code=response.status_code, detail=detail)

        if binary:
            return response.content
        try:
            return response.json()
        except ValueError as exc:
            raise APIError("The backend returned an invalid JSON response.", status_code=response.status_code) from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model/info")

    def model_metrics(self) -> dict[str, Any]:
        return self._request("GET", "/model/metrics")

    def feature_importance(self) -> dict[str, Any]:
        return self._request("GET", "/model/feature-importance")

    def template_csv(self) -> bytes:
        return self._request("GET", "/data/template", binary=True)

    def sample_csv(self, limit: int = 100) -> bytes:
        return self._request("GET", "/data/sample", params={"limit": limit}, binary=True)

    def predict(self, customer: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/predict", json=customer)

    def scenarios(self, customer: dict[str, Any], factors: list[float] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"customer": customer}
        if factors is not None:
            payload["factors"] = factors
        return self._request("POST", "/scenario", json=payload)

    def explain(self, customer: dict[str, Any], top_n: int = 10) -> dict[str, Any]:
        return self._request("POST", "/explain", json={"customer": customer, "top_n": top_n})

    def batch(self, customers: list[dict[str, Any]], *, skip_invalid: bool = False) -> dict[str, Any]:
        return self._request("POST", "/predict/batch", json={"customers": customers, "skip_invalid": skip_invalid})

    def batch_csv(self, contents: bytes, *, filename: str = "customers.csv", skip_invalid: bool = False) -> dict[str, Any]:
        return self._request(
            "POST", "/predict/batch-csv",
            params={"skip_invalid": str(skip_invalid).lower()},
            files={"file": (filename, contents, "text/csv")},
        )

    def audit_csv(
        self, contents: bytes, *, filename: str = "audit.csv", group_column: str = "SEX",
        min_group_size: int = 30, data_source: str = "uploaded_csv",
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/analytics/audit",
            data={"group_column": group_column, "min_group_size": str(min_group_size), "data_source": data_source},
            files={"file": (filename, contents, "text/csv")},
        )

    def monitor_csv(
        self, reference: bytes, current: bytes, *, reference_name: str = "reference.csv", current_name: str = "current.csv",
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/analytics/monitor",
            files={
                "reference_file": (reference_name, reference, "text/csv"),
                "current_file": (current_name, current, "text/csv"),
            },
        )
