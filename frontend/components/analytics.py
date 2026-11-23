"""Presentation of the original monitoring, fairness and reporting functions."""
from hashlib import sha256
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components.common import show_api_error
from frontend.components.ui import AQUA, INDIGO, polish_chart
from frontend.services.api import APIError, CreditAPI


def evaluation_metrics(metrics: dict, *, key_prefix: str) -> None:
    labels = [("Accuracy", "Accuracy"), ("Precision", "Precision"), ("Recall", "Recall"), ("F1", "F1"), ("ROC_AUC", "ROC-AUC"), ("PR_AUC", "PR-AUC")]
    for offset in (0, 3):
        for col, (key, label) in zip(st.columns(3), labels[offset:offset + 3]):
            value = metrics.get(key)
            col.metric(label, "Undefined" if value is None else f"{value:.2%}")
    st.caption("ROC-AUC and PR-AUC require both observed outcome classes. Undefined metrics are not replaced with zero.")
    matrix = [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]]
    figure = go.Figure(go.Heatmap(
        z=matrix, x=["Predicted no default", "Predicted default"], y=["Observed no default", "Observed default"],
        colorscale="Purples", text=matrix, texttemplate="%{text}", showscale=False,
    ))
    polish_chart(figure, "Confusion matrix for the uploaded outcomes", height=340)
    figure.update_yaxes(autorange="reversed")
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False}, key=f"{key_prefix}_confusion")
    with st.expander("All evaluation values"):
        st.json(metrics)


def render_audit_report(report: dict) -> None:
    st.subheader("Validation and fairness report")
    st.caption(f"Data source: {report['data_source']} · {report['total_customers']:,} customers · threshold {report['threshold']:.0%}")
    st.info("These measurements describe the uploaded labeled records. They are not automatically a held-out validation or proof of regulatory compliance.")
    evaluation_metrics(report["model_evaluation"], key_prefix="audit")
    fairness = report["fairness_analysis"]
    groups = pd.DataFrame(fairness["groups"])
    st.subheader(f"Fairness analysis by {fairness['group_column']}")
    if report.get("age_group_definition"):
        st.caption(str(report["age_group_definition"]))
    if not groups.empty:
        st.dataframe(groups, hide_index=True, width="stretch")
        if not groups["sufficient_sample"].all():
            st.warning(f"Some groups contain fewer than {fairness['min_group_size']} records. The original fairness logic excludes them from disparity calculations.")
        figure = go.Figure()
        figure.add_bar(x=groups["group"], y=groups["prediction_rate"], name="Predicted default rate", marker_color=INDIGO)
        figure.add_bar(x=groups["group"], y=groups["observed_default_rate"], name="Observed default rate", marker_color=AQUA)
        figure.update_layout(barmode="group")
        polish_chart(figure, "Predicted and observed default rates by group", "Rate", 380)
        figure.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
        st.download_button("Download group metrics CSV", groups.to_csv(index=False), "fairness_group_metrics.csv", "text/csv")
    st.write("**Group disparities (maximum minus minimum across eligible groups)**")
    st.dataframe(pd.DataFrame([fairness["disparities"]]).T.rename(columns={0: "Gap"}), width="stretch")
    for limitation in report.get("limitations", []):
        st.caption(limitation)
    st.caption(f"Regulatory compliance: {report.get('regulatory_compliance', 'not_assessed')}")
    st.download_button("Download complete audit report", json.dumps(report, indent=2, allow_nan=False), "credit_risk_audit_report.json", "application/json")


def render_monitoring_report(report: dict) -> None:
    drift = report["probability_drift"]
    st.subheader("Probability drift monitoring")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reference customers", drift["reference_count"])
    c2.metric("Current customers", drift["current_count"])
    c3.metric("Population stability index", f"{drift['psi']:.4f}")
    c4.metric("Drift status", drift["drift_status"].replace("_", " ").title())
    if drift["drift_status"] == "insufficient_samples":
        st.warning(f"Both datasets need at least {drift['minimum_samples']} records for the original monitoring status thresholds.")
    figure = go.Figure(go.Bar(
        x=["Reference", "Current"], y=[drift["reference_mean_probability"], drift["current_mean_probability"]],
        marker_color=[INDIGO, AQUA], text=[f"{drift['reference_mean_probability']:.2%}", f"{drift['current_mean_probability']:.2%}"],
        textposition="outside",
    ))
    polish_chart(figure, "Mean predicted default probability", "Probability", 340)
    figure.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
    st.caption(f"Mean probability change: {drift['mean_probability_change'] * 100:+.2f} percentage points.")
    if report.get("actual_labels_available") and report.get("performance") is not None:
        st.subheader("Performance on current observed outcomes")
        evaluation_metrics(report["performance"], key_prefix="monitor")
    else:
        st.info("No observed current outcomes were supplied. Distribution drift can be measured, but prediction performance cannot.")
    for limitation in report.get("limitations", []):
        st.caption(limitation)
    with st.expander("Complete monitoring values"):
        st.json(report)
    st.download_button("Download monitoring report", json.dumps(report, indent=2, allow_nan=False), "credit_risk_monitoring_report.json", "application/json")


def render_analytics(api: CreditAPI) -> None:
    st.title("Monitoring, fairness & reporting")
    st.write("Evaluate supplied records using the original project's analytics functions and the unchanged model.")
    st.caption("All uploads require the 23 raw feature columns. The backend validates and scores the records before calculating analytics.")
    with st.expander("Get a small UCI sample for a functional demonstration"):
        st.write("This sample contains actual source-dataset rows. It is for exercising the workflow and is not identified as an independent held-out test set.")
        if st.button("Prepare 100 UCI rows"):
            try:
                st.session_state.analytics_sample = api.sample_csv(100)
            except APIError as exc:
                show_api_error(exc)
        if "analytics_sample" in st.session_state:
            st.download_button("Download labeled UCI sample", st.session_state.analytics_sample, "uci_sample_100.csv", "text/csv")
    audit_tab, monitor_tab = st.tabs(["Fairness & audit report", "Monitoring"])
    with audit_tab:
        st.subheader("Upload observed outcomes")
        st.write("Add `actual_default` or `default.payment.next.month` with observed labels 0 or 1 to the 23 raw inputs.")
        file = st.file_uploader("Labeled customer CSV", type=["csv"], key="audit_file")
        col1, col2 = st.columns(2)
        group = col1.selectbox("Audit grouping", ["SEX", "EDUCATION", "MARRIAGE", "AGE_GROUP"])
        minimum = col2.number_input("Minimum group size", min_value=1, max_value=10000, value=30, step=1)
        source = st.text_input("Report data source", value="uploaded_csv")
        audit_signature = None if file is None else sha256(
            file.getvalue() + json.dumps([group, minimum, source], ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if st.session_state.get("audit_input_signature") != audit_signature:
            st.session_state.audit_input_signature = audit_signature
            st.session_state.pop("audit_report", None)
        if st.button("Run fairness and validation audit", type="primary", disabled=file is None):
            try:
                with st.spinner("Scoring records and building the audit report..."):
                    st.session_state.audit_report = api.audit_csv(file.getvalue(), filename=file.name, group_column=group, min_group_size=minimum, data_source=source)
            except APIError as exc:
                st.session_state.pop("audit_report", None)
                show_api_error(exc)
        if "audit_report" in st.session_state:
            render_audit_report(st.session_state.audit_report)
    with monitor_tab:
        st.subheader("Compare reference and current records")
        st.write("Upload a reference population and a current population. Current records can optionally include observed default labels to measure performance.")
        col1, col2 = st.columns(2)
        with col1:
            reference = st.file_uploader("Reference customer CSV", type=["csv"], key="reference_file")
        with col2:
            current = st.file_uploader("Current customer CSV", type=["csv"], key="current_file")
        monitor_signature = None if reference is None or current is None else (
            sha256(reference.getvalue()).hexdigest(), sha256(current.getvalue()).hexdigest()
        )
        if st.session_state.get("monitor_input_signature") != monitor_signature:
            st.session_state.monitor_input_signature = monitor_signature
            st.session_state.pop("monitoring_report", None)
        if st.button("Run monitoring comparison", type="primary", disabled=reference is None or current is None):
            try:
                with st.spinner("Scoring both populations and measuring drift..."):
                    st.session_state.monitoring_report = api.monitor_csv(reference.getvalue(), current.getvalue(), reference_name=reference.name, current_name=current.name)
            except APIError as exc:
                st.session_state.pop("monitoring_report", None)
                show_api_error(exc)
        if "monitoring_report" in st.session_state:
            render_monitoring_report(st.session_state.monitoring_report)
