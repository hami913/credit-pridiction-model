"""One API-backed batch assessment view shared by the page entry point."""
from hashlib import sha256
from io import BytesIO

import pandas as pd
import streamlit as st

from frontend.components.common import footer, show_api_error
from frontend.components.ui import batch_histogram, section
from frontend.services.api import APIError, CreditAPI


def render_batch(api: CreditAPI, model_info: dict) -> None:
    st.markdown(
        """<div class="hero"><div class="hero-badge"><span class="hero-dot"></span>Batch processing</div>
        <h1 class="hero-title">Score many customers at once.</h1>
        <p class="hero-copy">Upload a CSV of customer profiles and get a default probability,
        risk level and decision for each valid row.</p></div>""", unsafe_allow_html=True,
    )
    st.page_link("app.py", label="Back to single assessment", icon="💳")
    section("1", "Upload your customer file", "All 23 raw columns are required. Customer identifiers and extra columns are retained.")
    with st.expander("View required CSV columns"):
        st.dataframe(pd.DataFrame({"Required column": model_info["raw_features"]}), hide_index=True, width="stretch")
        st.caption("Education codes 0, 5 and 6 and marriage code 0 retain the original project's category mapping in the backend.")
    left, right = st.columns([1, 2], gap="large")
    with left:
        try:
            template = api.template_csv()
            st.download_button("⬇️ Download CSV template", template, "batch_template.csv", "text/csv", width="stretch")
        except APIError as exc:
            show_api_error(exc)
        st.caption("The two template rows are synthetic examples for functionality testing.")
    with right:
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_upload")
    policy = st.radio(
        "Invalid-row handling", ["Require every row to be valid", "Score valid rows and report skipped rows"],
        help="Require every row to be valid to prevent partial results, or score valid rows and download their results while reviewing explicit reasons for each skipped row.",
    )
    skip_invalid = policy == "Score valid rows and report skipped rows"
    if uploaded is not None:
        contents = uploaded.getvalue()
        signature = sha256(contents + str(skip_invalid).encode()).hexdigest()
        if st.session_state.get("batch_upload_signature") != signature:
            st.session_state.batch_upload_signature = signature
            st.session_state.pop("batch_response", None)
        try:
            preview = pd.read_csv(BytesIO(contents), dtype=object, keep_default_na=False)
            c1, c2, c3 = st.columns(3)
            c1.metric("Uploaded customers", len(preview))
            c2.metric("Uploaded columns", len(preview.columns))
            c3.metric("Required inputs", model_info["raw_feature_count"])
            st.dataframe(preview.head(10), hide_index=True, width="stretch")
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError):
            st.error("Unable to preview this CSV. Save it as a UTF-8 comma-separated file with a header row.")
            st.stop()
        if st.button("🚀 Run batch scoring", type="primary", width="stretch"):
            try:
                with st.spinner("Scoring customers..."):
                    st.session_state.batch_response = api.batch_csv(contents, filename=uploaded.name, skip_invalid=skip_invalid)
                st.toast("Batch scoring completed.")
            except APIError as exc:
                st.session_state.pop("batch_response", None)
                show_api_error(exc)
    elif "batch_upload_signature" in st.session_state:
        st.session_state.pop("batch_upload_signature", None)
        st.session_state.pop("batch_response", None)

    if "batch_response" in st.session_state:
        response = st.session_state.batch_response
        results = pd.DataFrame(response["results"])
        section("2", "Batch assessment summary", "Each result retains its original CSV row number. The first data row is row 2.")
        flagged = int((results["prediction"] == 1).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers scored", response["scored_rows"])
        c2.metric("Higher risk", flagged)
        c3.metric("Lower risk", response["scored_rows"] - flagged)
        c4.metric("Average default risk", f"{results['default_probability'].mean():.2%}")
        if response["skipped_rows"]:
            st.warning(f"{response['skipped_rows']} row(s) were skipped because validation failed.")
        if response.get("errors"):
            failures = []
            for row in response["errors"]:
                failures.append({
                    "CSV row": row["row_number"],
                    "Reason": "; ".join(
                        f"{'.'.join(str(item) for item in error.get('loc', []))}: {error.get('msg', 'Invalid input')}"
                        for error in row["errors"]
                    ),
                })
            st.dataframe(pd.DataFrame(failures), hide_index=True, width="stretch")
        if response.get("renamed_input_columns"):
            st.info("Some input column names overlap output fields. Their original values were preserved under new names.")
            st.json(response["renamed_input_columns"])
        chart_data = results.rename(columns={"default_probability_percent": "Default Probability (%)"})
        st.plotly_chart(batch_histogram(chart_data, response["threshold"]), width="stretch", config={"displayModeBar": False})
        st.subheader("Customer risk results")
        st.dataframe(results.sort_values("default_probability", ascending=False), hide_index=True, width="stretch")
        st.caption(
            "risk_band is the original dashboard's Low/Moderate/High definition; risk_level is the original "
            "risk-index module's Low/Medium/High definition. See Model information for their different cutoffs."
        )
        st.download_button("⬇️ Download scored results", results.to_csv(index=False), "credit_risk_batch_results.csv", "text/csv", width="stretch")
    footer()
