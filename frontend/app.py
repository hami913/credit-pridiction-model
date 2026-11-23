"""Credit Risk Intelligence: original Streamlit experience backed by FastAPI."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components.common import backend, footer, navigation, show_api_error
from frontend.components.ui import (
    CORAL, MINT, DEFAULT_HISTORY, DEMO_HISTORY, STATUS_MAP,
    apply_style, build_customer_data, build_insights, monthly_financial_chart,
    polish_chart, repayment_status_chart, risk_gauge_chart, scenario_chart,
    section, utilization_chart, validate_inputs,
)
from frontend.services.api import APIError


st.set_page_config(page_title="Credit Risk Intelligence", page_icon="💳", layout="wide", initial_sidebar_state="collapsed")
apply_style()
navigation()
api, model_info = backend()
threshold = float(model_info["threshold"])

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge"><span class="hero-dot"></span>AI credit risk assessment</div>
        <h1 class="hero-title">Know the risk before the next statement.</h1>
        <p class="hero-copy">Enter a customer profile and six months of repayment behavior to estimate
        next-month default probability, with a clear explanation of what drives it.</p>
        <div class="hero-card">
            <div class="hc-row"><span>CREDIT RISK</span><span class="hc-pill">AI SCORED</span></div>
            <div><div class="hc-chip"></div></div><div class="hc-num">•••• •••• •••• 4821</div>
        </div>
    </div>
    """, unsafe_allow_html=True,
)
nav1, nav2, nav3 = st.columns(3)
nav1.page_link("pages/1_Batch_Assessment.py", label="Batch processing", icon="📊", width="stretch")
nav2.page_link("pages/2_Model_Information.py", label="Model & performance", icon="🔎", width="stretch")
nav3.page_link("pages/3_Analytics.py", label="Monitoring & fairness", icon="📈", width="stretch")
st.caption(
    f"{model_info['model_name']} · version {model_info['model_version']} · "
    f"decision threshold {threshold:.0%} · {model_info['raw_feature_count']} raw inputs"
)

if "history_data" not in st.session_state:
    st.session_state.history_data = DEFAULT_HISTORY.copy()


def clear_assessment() -> None:
    for key in ("assessment", "scenarios", "explanation", "scenario_error", "explanation_error", "credit_history_editor"):
        st.session_state.pop(key, None)


with st.expander("Demo controls"):
    demo_col, reset_col = st.columns(2)
    if demo_col.button("Load sample history", width="stretch"):
        st.session_state.history_data = DEMO_HISTORY.copy()
        clear_assessment()
        st.rerun()
    if reset_col.button("Clear history", width="stretch"):
        st.session_state.history_data = DEFAULT_HISTORY.copy()
        clear_assessment()
        st.rerun()
    st.caption("Sample values are for interface testing only. Use real account values for an actual assessment.")

with st.form("risk_assessment_form"):
    section("1", "Customer information", "Basic demographic and credit-limit inputs used by the model.")
    p1, p2, p3 = st.columns(3)
    limit_bal = p1.number_input("Credit limit", min_value=10000.0, max_value=1000000.0, value=200000.0, step=10000.0)
    age = p2.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = p3.selectbox("Sex", [1, 2], format_func=lambda value: {1: "Male", 2: "Female"}[value])
    p4, p5 = st.columns(2)
    education = p4.selectbox("Education", [1, 2, 3, 4], format_func=lambda value: {
        1: "Graduate School", 2: "University", 3: "High School", 4: "Other",
    }[value])
    marriage = p5.selectbox("Marital status", [1, 2, 3], format_func=lambda value: {1: "Married", 2: "Single", 3: "Other"}[value])
    st.divider()
    section("2", "Six-month account history", "Add the six most recent repayment statuses, statement balances, and payments.")
    edited_history = st.data_editor(
        st.session_state.history_data, hide_index=True, width="stretch",
        disabled=["Month"], num_rows="fixed", key="credit_history_editor",
        column_config={
            "Month": st.column_config.TextColumn("Month", width="small"),
            "Repayment Status": st.column_config.SelectboxColumn("Repayment Status", options=list(STATUS_MAP), required=True, width="medium"),
            "Statement Balance": st.column_config.NumberColumn("Statement Balance", step=1000.0, format="%.0f", width="medium"),
            "Payment Made": st.column_config.NumberColumn("Payment Made", min_value=0.0, step=1000.0, format="%.0f", width="medium"),
        },
    )
    submitted = st.form_submit_button("Run credit risk assessment", type="primary", width="stretch")

if submitted:
    st.session_state.history_data = edited_history.copy()
    try:
        customer_frame, pay_status, bills, payments = build_customer_data(limit_bal, sex, education, marriage, age, edited_history)
        errors = validate_inputs(limit_bal, age, pay_status, bills, payments)
    except (TypeError, ValueError, KeyError):
        st.error("Complete all six history rows with valid status and financial values.")
        st.stop()
    if errors:
        st.error("Please review the supplied customer information.")
        for error in errors:
            st.write(f"- {error}")
        st.stop()
    if all(float(value) == 0 for value in bills + payments):
        st.warning("Insufficient account activity for a meaningful assessment.")
        st.info("Enter at least one statement balance or payment from the six-month account history before running the assessment.")
        st.stop()
    # Preserve input floating-point precision when building the JSON record.
    customer = customer_frame.to_dict(orient="records")[0]
    try:
        with st.spinner("Assessing the customer..."):
            assessment = api.predict(customer)
    except APIError as exc:
        st.session_state.pop("assessment", None)
        show_api_error(exc)
        st.stop()
    for key in ("scenarios", "explanation", "scenario_error", "explanation_error"):
        st.session_state.pop(key, None)
    st.session_state.assessment = assessment
    st.session_state.customer_data = customer
    st.session_state.assessment_history = edited_history.copy()
    st.session_state.assessment_limit = float(limit_bal)
    st.toast("Assessment completed.")

if "assessment" in st.session_state:
    result = st.session_state.assessment
    indicators = result["indicators"]
    probability = result["default_probability"]
    result_threshold = result["threshold"]
    level = result["risk_band"]
    level_class = {"Low": "low", "Moderate": "mid", "High": "high"}[level]
    advice = {
        "Low": "Profile looks healthy. Standard monitoring is sufficient.",
        "Moderate": "Approaching the decision threshold. Consider a limit review and closer monitoring.",
        "High": "At or above the decision threshold. Recommend manual review before extending further credit.",
    }[level]
    section("3", "Estimated default risk", "The model output comes first; supporting behavior and explanation are below.")
    gauge_col, summary_col = st.columns([1.1, 1], gap="large")
    with gauge_col:
        st.plotly_chart(risk_gauge_chart(probability, result_threshold, level), width="stretch", config={"displayModeBar": False})
    with summary_col:
        st.markdown(
            f"""<div class="result-shell {level_class}">
            <div class="result-top"><div class="result-eyebrow">Next-month default probability</div>
            <div class="result-status">● {level} risk</div></div>
            <div class="result-number">{probability * 100:.1f}%</div>
            <div class="meter"><div class="meter-fill" style="width:{probability * 100:.1f}%"></div>
            <div class="meter-mark" style="left:{result_threshold * 100:.1f}%"></div></div>
            <div class="meter-scale"><span>0%</span><span>Threshold {result_threshold:.0%}</span><span>100%</span></div>
            </div><div class="advice"><b>Recommendation:</b> {advice}</div>""", unsafe_allow_html=True,
        )
    score1, score2, score3 = st.columns(3)
    score1.metric("Risk index · 0–100", f"{result['risk_index']:.2f}")
    score2.metric("Risk-index band", result["risk_level"])
    score3.metric("Non-default probability", f"{result['non_default_probability']:.2%}")
    st.caption(
        f"The original dashboard band is Low below {result_threshold * 0.6:.0%}, Moderate below {result_threshold:.0%}, "
        f"and High from {result_threshold:.0%}. The original risk-index module separately uses Low below 30%, "
        "Medium below 60%, and High from 60%. Both original definitions are preserved; the binary prediction uses the saved model threshold."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Delayed months", indicators["delayed_months"])
    m2.metric("Average utilization", f"{indicators['avg_utilization']:.1%}")
    m3.metric("Average payment", f"{indicators['avg_payment']:,.0f}")
    m4.metric("Payment / bill", "N/A" if indicators["avg_bill"] <= 0 else f"{indicators['pay_to_bill']:.0%}")
    if indicators["avg_utilization"] > 1:
        st.warning("Average statement balance exceeds the entered credit limit. Verify the financial values before relying on this result.")
    section("4", "Key takeaways", "Plain-English reading of this customer's behavior.")
    for note in build_insights(indicators):
        st.markdown(f'<div class="insight">{note}</div>', unsafe_allow_html=True)
    with st.expander("View engineered credit indicators"):
        st.dataframe(pd.DataFrame([indicators]).T.rename(columns={0: "Value"}), width="stretch")
        st.caption("Exact model-input fields returned by the backend, in the saved artifact's feature order:")
        st.dataframe(pd.DataFrame([result["engineered_features"]]), hide_index=True, width="stretch")
    summary = {
        "default_probability": probability, "default_probability_pct": result["default_probability_percent"],
        "non_default_probability": result["non_default_probability"], "prediction": result["prediction"],
        "risk_band": level, "risk_index": result["risk_index"], "risk_level": result["risk_level"],
        "threshold": result_threshold, "model_version": result["model_version"], **indicators,
    }
    st.download_button("⬇️ Download assessment summary", pd.DataFrame([summary]).to_csv(index=False), "credit_risk_assessment.csv", "text/csv")

    trends_tab, scenario_tab, explain_tab = st.tabs(["Payment trends", "What-if scenarios", "Model explanation"])
    with trends_tab:
        chart_1, chart_2 = st.columns(2)
        with chart_1:
            st.plotly_chart(monthly_financial_chart(st.session_state.assessment_history), width="stretch", config={"displayModeBar": False})
        with chart_2:
            st.plotly_chart(repayment_status_chart(st.session_state.assessment_history), width="stretch", config={"displayModeBar": False})
        st.plotly_chart(utilization_chart(st.session_state.assessment_history, st.session_state.assessment_limit), width="stretch", config={"displayModeBar": False})
    with scenario_tab:
        st.caption("The same customer is re-scored with payments scaled up or down. Repayment status and balances are held constant.")
        if "scenarios" not in st.session_state and "scenario_error" not in st.session_state:
            try:
                with st.spinner("Comparing payment scenarios..."):
                    st.session_state.scenarios = api.scenarios(st.session_state.customer_data)
            except APIError as exc:
                st.session_state.scenario_error = str(exc)
        if "scenario_error" in st.session_state:
            st.error(st.session_state.scenario_error)
            if st.button("Retry scenario analysis"):
                st.session_state.pop("scenario_error")
                st.rerun()
        elif "scenarios" in st.session_state:
            data = st.session_state.scenarios
            scenarios = pd.DataFrame(data["scenarios"]).rename(columns={
                "label": "Scenario", "factor": "Factor", "risk_band": "Risk Level",
            })
            scenarios["Probability"] = scenarios["default_probability"] * 100
            st.plotly_chart(scenario_chart(scenarios, data["threshold"]), width="stretch", config={"displayModeBar": False})
            current = scenarios.loc[scenarios["Factor"] == 1.0, "Probability"]
            best = scenarios.loc[scenarios["Probability"].idxmin()]
            if not current.empty:
                delta = float(current.iloc[0]) - float(best["Probability"])
                msg = f"Lowest modelled risk is <b>{best['Probability']:.1f}%</b> under the <b>{best['Scenario']}</b> scenario"
                msg += f", a change of <b>{delta:.1f} points</b> from today." if delta > 0.05 else ": payment level has little effect on this profile."
                st.markdown(f'<div class="scenario-note">{msg}</div>', unsafe_allow_html=True)
            st.dataframe(scenarios[["Scenario", "Factor", "Probability", "Risk Level"]], hide_index=True, width="stretch")
            st.download_button("Download scenario comparison", scenarios.to_csv(index=False), "payment_scenarios.csv", "text/csv")
    with explain_tab:
        st.caption("SHAP values indicate which features moved this prediction up or down. They do not prove causation.")
        if "explanation" not in st.session_state and "explanation_error" not in st.session_state:
            try:
                with st.spinner("Computing explanation..."):
                    st.session_state.explanation = api.explain(st.session_state.customer_data)
            except APIError as exc:
                st.session_state.explanation_error = str(exc)
        if "explanation_error" in st.session_state:
            st.info(st.session_state.explanation_error)
            if st.button("Retry model explanation"):
                st.session_state.pop("explanation_error")
                st.rerun()
        elif "explanation" in st.session_state:
            explanation = st.session_state.explanation
            feature_data = pd.DataFrame(explanation["features"]).sort_values("shap_value")
            figure = go.Figure(go.Bar(
                x=feature_data["shap_value"], y=feature_data["feature"], orientation="h",
                marker={"color": [CORAL if value > 0 else MINT for value in feature_data["shap_value"]], "cornerradius": 6},
                hovertemplate="%{y}<br>SHAP contribution: %{x:.4f}<extra></extra>",
            ))
            figure.add_vline(x=0, line_width=1, opacity=.45)
            polish_chart(figure, "Top factors influencing this prediction", height=430)
            figure.update_layout(xaxis_title="SHAP contribution", showlegend=False)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
            st.caption(explanation.get("note", "Explanation of the model's positive/default class."))
            st.download_button("Download explanation", feature_data.to_csv(index=False), "credit_risk_explanation.csv", "text/csv")

footer()
