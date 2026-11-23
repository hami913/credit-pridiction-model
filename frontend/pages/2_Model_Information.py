"""Artifact metadata, source metrics, and actual classifier importance."""
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components.common import backend, footer, navigation, show_api_error
from frontend.components.ui import INDIGO, apply_style, polish_chart, section
from frontend.services.api import APIError

st.set_page_config(page_title="Credit Model Information", page_icon="🔎", layout="wide")
apply_style()
navigation()
api, info = backend()
st.title("Model information & performance")
st.write("The original credit-default Random Forest, its saved metadata, and the feature importance of the deployed artifact.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", "Random Forest")
c2.metric("Version", str(info["model_version"]))
c3.metric("Decision threshold", f"{info['threshold']:.0%}")
c4.metric("Raw inputs", info["raw_feature_count"])
st.caption(
    f"{info['engineered_feature_count']} total model-input columns after feature engineering "
    f"({info['raw_feature_count']} original inputs plus {info['engineered_feature_count'] - info['raw_feature_count']} derived columns); "
    f"{info['processed_feature_count']} columns after preprocessing. Target: {info['target']}."
)
with st.expander("Feature contract and model metadata"):
    raw, engineered = st.columns(2)
    raw.dataframe(pd.DataFrame({"Raw feature": info["raw_features"]}), hide_index=True, width="stretch")
    engineered.dataframe(pd.DataFrame({"Model input feature": info["features"]}), hide_index=True, width="stretch")
    st.write("**Artifact SHA-256**")
    st.code(info["model_sha256"], language=None)
    st.json(info)
    st.download_button("Download model information", json.dumps(info, indent=2, allow_nan=False), "model_information.json", "application/json")

section("1", "Original risk definitions", "The original model decision and the original risk-index module use different bands.")
t = float(info["threshold"])
st.table(pd.DataFrame([
    {"Definition": "Dashboard risk_band", "Low": f"p < {0.6 * t:g}", "Middle": f"Moderate: {0.6 * t:g} ≤ p < {t:g}", "High": f"p ≥ {t:g}"},
    {"Definition": "Risk-index risk_level", "Low": "p < 0.30", "Middle": "Medium: 0.30 ≤ p < 0.60", "High": "p ≥ 0.60"},
]))
st.caption(f"Binary default prediction: p ≥ {t:g}. Risk index: probability × 100, rounded to two decimals. Bands are decision-support conventions.")

section("2", "Historical model performance", "Metrics retained from the supplied project's historical report.")
try:
    historical = api.model_metrics()
    st.info("These are the original reported metrics. They are not newly measured integration-test results.")
    st.caption(f"Source: {historical['source']}")
    metrics = historical["metrics"]
    labels = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    for offset in (0, 3):
        columns = st.columns(3)
        for col, (key, label) in zip(columns, list(labels.items())[offset:offset + 3]):
            value = metrics.get(key)
            col.metric(label, "Unavailable" if value is None else f"{value:.2%}")
    with st.expander("Original report and governance statements"):
        st.json(historical["report"])
        st.caption("This legacy report describes the original project. Its filename and governance statements do not establish regulatory certification.")
    st.download_button("Download original historical report", json.dumps(historical["report"], indent=2, allow_nan=False), "credit_risk_regulatory_report.json", "application/json")
except APIError as exc:
    show_api_error(exc)

section("3", "Feature importance", "Importance values calculated from the unchanged deployed Random Forest.")
try:
    payload = api.feature_importance()
    mode = st.radio("Feature grouping", ["Model input groups", "Processed columns"], horizontal=True)
    rows = payload["aggregated_features"] if mode == "Model input groups" else payload["features"]
    data = pd.DataFrame(rows)
    top_n = st.slider("Number of features", min_value=5, max_value=len(data), value=min(15, len(data)))
    selected = data.sort_values("importance", ascending=False).head(top_n).sort_values("importance")
    figure = go.Figure(go.Bar(x=selected["importance"], y=selected["feature"], orientation="h", marker={"color": INDIGO, "cornerradius": 6}))
    polish_chart(figure, "Most influential features across the forest", height=max(400, top_n * 24))
    figure.update_layout(xaxis_title="Impurity-based importance", showlegend=False)
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption(f"Method: {payload['method']}. Global importance does not show the direction of a specific customer's prediction or establish causation.")
    st.dataframe(data.sort_values("importance", ascending=False), hide_index=True, width="stretch")
    st.download_button("Download feature importance", data.to_csv(index=False), "feature_importance.csv", "text/csv")
except APIError as exc:
    show_api_error(exc)
footer()
