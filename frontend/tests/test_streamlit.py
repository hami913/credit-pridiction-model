from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.services.api import APIError

APP = Path(__file__).resolve().parents[1] / "app.py"


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_initial_single_form_and_empty_history_guard(fake_api):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    button(app, "Run credit risk assessment").click().run()
    assert not app.exception
    assert any("Insufficient account activity" in item.value for item in app.warning)
    assert fake_api.calls == []


def test_single_scenarios_explanation_and_result_survive_rerun(fake_api):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    button(app, "Load sample history").click().run()
    button(app, "Run credit risk assessment").click().run()
    assert not app.exception
    assert [entry[0] for entry in fake_api.calls] == ["predict", "scenario", "explain"]
    assert len(fake_api.calls[0][1]) == 23
    assert len(app.tabs) == 3
    assert any(item.label == "Risk index · 0–100" and item.value == "40.00" for item in app.metric)
    assert len(app.get("plotly_chart")) == 6
    app.run()
    assert len(fake_api.calls) == 3
    button(app, "Clear history").click().run()
    assert not app.exception
    assert "assessment" not in app.session_state


@pytest.mark.parametrize("page,title", [
    ("pages/1_Batch_Assessment.py", None),
    ("pages/2_Model_Information.py", "Model information & performance"),
    ("pages/3_Analytics.py", "Monitoring, fairness & reporting"),
])
def test_all_pages_render_through_real_streamlit_navigation(fake_api, page, title):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.switch_page(page).run()
    assert not app.exception
    if title:
        assert any(item.value == title for item in app.title)


def test_backend_unavailable_is_visible_without_traceback(monkeypatch):
    import frontend.components.common as common

    def unavailable():
        raise APIError("Cannot connect to the FastAPI backend.")

    monkeypatch.setattr(common, "CreditAPI", unavailable)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any("Cannot connect" in item.value for item in app.error)
    assert any(item.label == "Retry backend connection" for item in app.button)


def test_scenario_comparison_uses_full_precision_not_display_rounding(fake_api):
    def close_probabilities(customer):
        return {"threshold": 0.5, "scenarios": [
            {"label": "Current", "factor": 1, "default_probability": 0.40000001, "default_probability_percent": 40.0, "risk_band": "Moderate"},
            {"label": "Double", "factor": 2, "default_probability": 0.4, "default_probability_percent": 40.0, "risk_band": "Moderate"},
        ]}

    fake_api.scenarios = close_probabilities
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    button(app, "Load sample history").click().run()
    button(app, "Run credit risk assessment").click().run()
    assert not app.exception
    assert any("<b>Double</b>" in item.value for item in app.markdown)
