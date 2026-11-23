"""Shared navigation and backend availability presentation."""
import streamlit as st

from frontend.services.api import APIError, CreditAPI


def navigation() -> None:
    with st.sidebar:
        st.title("Credit Risk Intelligence")
        st.page_link("app.py", label="Single assessment", icon="💳")
        st.page_link("pages/1_Batch_Assessment.py", label="Batch assessment", icon="📊")
        st.page_link("pages/2_Model_Information.py", label="Model information", icon="🔎")
        st.page_link("pages/3_Analytics.py", label="Monitoring & fairness", icon="📈")
        st.caption("Original credit-default model · FastAPI + Streamlit")


def backend() -> tuple[CreditAPI, dict]:
    try:
        api = CreditAPI()
        info = api.model_info()
        return api, info
    except APIError as exc:
        st.error(str(exc))
        st.info("Start FastAPI in one terminal, then Streamlit in another. The README includes both commands.")
        if st.button("Retry backend connection"):
            st.rerun()
        st.stop()


def show_api_error(exc: APIError) -> None:
    st.error(str(exc))
    if isinstance(exc.detail, (dict, list)):
        with st.expander("Validation details"):
            st.json(exc.detail)


def footer() -> None:
    st.markdown(
        '<div class="footer-note">Credit Risk Intelligence · Decision support only</div>',
        unsafe_allow_html=True,
    )
