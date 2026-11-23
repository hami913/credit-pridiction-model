"""Presentation helpers adapted from the original credit Streamlit application.

These helpers never load a model or engineer prediction features. Risk labels,
scenarios and explanations come from the FastAPI service.
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


STYLE = """
    <style>

    :root {
        --ink: #0E1330; --indigo: #7C3AED; --violet: #A855F7; --aqua: #19D3C5;
        --coral: #FF5C7A; --mint: #22C58B; --amber: #FFB020;
        --text: #1B2140; --muted: #6B7391; --line: #E7E1F7;
        --surface: #FFFFFF; --bg: #F5F1FF;
        --shadow-lg: 0 40px 80px -28px rgba(76,29,149,.38);
        --shadow-md: 0 20px 44px -20px rgba(76,29,149,.26);
        --shadow-sm: 0 8px 22px -12px rgba(76,29,149,.22);
        --font-head: 'Sora','Inter',sans-serif;
        --font-body: 'Inter',system-ui,sans-serif;
    }
    html {scroll-behavior: smooth;}
    html, body, .stApp, [class*="css"] {font-family: var(--font-body);}

    .stApp {
        background:
            radial-gradient(rgba(124,58,237,.10) 1px, transparent 1px) 0 0 / 26px 26px,
            radial-gradient(60rem 30rem at 0% -5%, rgba(124,58,237,.18), transparent 60%),
            radial-gradient(50rem 28rem at 100% 0%, rgba(25,211,197,.16), transparent 60%),
            radial-gradient(40rem 26rem at 50% 110%, rgba(168,85,247,.14), transparent 60%),
            var(--bg);
        color: var(--text);
    }
    .block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem;}
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden; border-radius: 32px;
        padding: 56px 52px 50px; color: #fff; isolation: isolate;
        background:
            radial-gradient(34rem 20rem at 88% 8%, rgba(168,85,247,.65), transparent 65%),
            radial-gradient(28rem 18rem at 70% 110%, rgba(25,211,197,.45), transparent 65%),
            radial-gradient(24rem 16rem at 0% 100%, rgba(124,58,237,.6), transparent 65%),
            linear-gradient(135deg, #120826 0%, #2A1257 60%, #3B1B7A 100%);
        box-shadow: var(--shadow-lg); margin-bottom: 1.6rem;
        border: 1px solid rgba(255,255,255,.12);
    }
    .hero::before {
        content: ""; position: absolute; inset: 0; z-index: -1;
        background-image:
            linear-gradient(rgba(255,255,255,.055) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.055) 1px, transparent 1px);
        background-size: 44px 44px;
        mask-image: radial-gradient(ellipse at 75% 30%, #000 0%, transparent 70%);
        -webkit-mask-image: radial-gradient(ellipse at 75% 30%, #000 0%, transparent 70%);
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: .55rem;
        padding: .45rem .95rem; border-radius: 999px;
        background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
        backdrop-filter: blur(10px); color: #EDEBFF;
        font-size: .78rem; font-weight: 600; letter-spacing: .04em;
        text-transform: uppercase; margin-bottom: 1.3rem;
    }
    .hero-dot {
        width: 8px; height: 8px; border-radius: 50%; background: var(--aqua);
        box-shadow: 0 0 0 0 rgba(25,211,197,.7); animation: pulse 2.2s infinite;
    }
    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(25,211,197,.65);}
        70% {box-shadow: 0 0 0 10px rgba(25,211,197,0);}
        100% {box-shadow: 0 0 0 0 rgba(25,211,197,0);}
    }
    @keyframes float {
        0%,100% {transform: translateY(-50%) rotate(-6deg);}
        50% {transform: translateY(-56%) rotate(-4deg);}
    }
    .hero-title {
        font-family: var(--font-head); font-size: clamp(2.3rem, 5vw, 3.9rem);
        font-weight: 800; letter-spacing: -.045em; line-height: 1.04;
        margin: 0; padding: 0; max-width: 640px;
        background: linear-gradient(100deg, #FFFFFF 30%, #E2CCFF 70%, #8CF3E8 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-copy {color: #C9CFF0; font-size: 1.05rem; margin: 1.1rem 0 0; max-width: 540px; line-height: 1.7;}

    .hero-card {
        position: absolute; right: 52px; top: 50%; width: 310px; height: 190px;
        border-radius: 24px; padding: 22px; transform: translateY(-50%) rotate(-6deg);
        background: linear-gradient(135deg, rgba(255,255,255,.26), rgba(255,255,255,.05));
        border: 1px solid rgba(255,255,255,.3); backdrop-filter: blur(18px);
        box-shadow: 0 30px 60px -20px rgba(0,0,0,.5);
        animation: float 6s ease-in-out infinite;
        display: flex; flex-direction: column; justify-content: space-between;
    }
    .hc-chip {width: 40px; height: 30px; border-radius: 7px; background: linear-gradient(135deg, #FFE08A, #E9A93B);}
    .hc-num {font-family: var(--font-head); letter-spacing: .18em; font-size: 1.05rem; color: #F3EEFF;}
    .hc-row {display: flex; justify-content: space-between; align-items: center; font-size: .72rem; color: #D7CBFF; letter-spacing: .08em;}
    .hc-pill {padding: .28rem .7rem; border-radius: 999px; background: rgba(25,211,197,.22); border: 1px solid rgba(25,211,197,.5); color: #A9F5EC; font-weight: 700;}
    @media (max-width: 980px) {.hero-card {display: none;}}

    /* ---------- Section headings ---------- */
    .section-wrap {display: flex; align-items: flex-start; gap: 1rem; margin: .2rem 0 1.1rem;}
    .section-step {
        flex: 0 0 auto; width: 40px; height: 40px; display: grid; place-items: center;
        border-radius: 13px; color: #fff; font-family: var(--font-head);
        font-weight: 700; font-size: .95rem;
        background: linear-gradient(135deg, #6D28D9, var(--violet));
        box-shadow: 0 12px 22px -8px rgba(124,58,237,.7);
    }
    .section-title {font-family: var(--font-head); color: var(--text); font-size: 1.34rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.2;}
    .section-copy {color: var(--muted); font-size: .9rem; margin-top: .2rem; line-height: 1.55;}

    /* ---------- Form / cards ---------- */
    div[data-testid="stForm"] {
        position: relative; overflow: hidden;
        background: rgba(255,255,255,.88); border: 1px solid #fff;
        outline: 1px solid var(--line); border-radius: 28px;
        padding: 1.8rem 1.9rem 1.6rem; box-shadow: var(--shadow-md);
        backdrop-filter: blur(16px);
    }
    div[data-testid="stForm"]::before {
        content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px;
        background: linear-gradient(90deg, var(--indigo), var(--violet), var(--aqua));
    }
    div[data-testid="stExpander"] {
        background: rgba(255,255,255,.82); border: 1px solid var(--line);
        border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm); margin-bottom: 1.2rem;
    }
    div[data-testid="stExpander"] details summary p {font-weight: 600; color: var(--text);}

    div[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] label, div[data-testid="stForm"] label p {
        color: var(--text) !important; font-weight: 600 !important; font-size: .88rem !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"],
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] > div > div {
        background: #FBFAFF !important; background-color: #FBFAFF !important;
        border-radius: 12px !important; border-color: var(--line) !important;
    }
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {border: 1px solid var(--line) !important; overflow: hidden;}
    div[data-testid="stNumberInput"] input {border: 0 !important; box-shadow: none !important;}
    div[data-testid="stNumberInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"] input {color: var(--text) !important; -webkit-text-fill-color: var(--text) !important;}
    div[data-testid="stNumberInput"] button {background: #EFEBFB !important; color: var(--text) !important; border: 0 !important;}
    div[data-testid="stSelectbox"] svg {fill: var(--muted) !important;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
    div[data-testid="stSelectbox"] > div > div:focus-within {
        border-color: var(--indigo) !important; box-shadow: 0 0 0 3px rgba(124,58,237,.18) !important;
    }
    div[data-testid="stDataEditor"] {border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm);}

    div[data-testid="stFormSubmitButton"] button[kind="primary"] {
        background: linear-gradient(95deg, #6D28D9 0%, var(--violet) 55%, #C084FC 100%);
        color: #fff; border: 0; border-radius: 16px; min-height: 58px;
        font-family: var(--font-head); font-weight: 700; font-size: 1.02rem;
        box-shadow: 0 22px 40px -16px rgba(124,58,237,.8);
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
        transform: translateY(-2px); filter: brightness(1.07);
        box-shadow: 0 26px 44px -16px rgba(124,58,237,.9);
    }
    div[data-testid="stFormSubmitButton"] button[kind="primary"]:focus-visible,
    .stButton > button:focus-visible {outline: 3px solid rgba(124,58,237,.4); outline-offset: 2px;}
    .stButton > button, div[data-testid="stDownloadButton"] > button {
        border-radius: 14px; border: 1px solid var(--line); background: #fff;
        font-weight: 600; color: var(--text); box-shadow: var(--shadow-sm);
        transition: border-color .15s ease, color .15s ease, transform .15s ease;
    }
    .stButton > button:hover, div[data-testid="stDownloadButton"] > button:hover {
        border-color: var(--indigo); color: var(--indigo); transform: translateY(-1px);
    }

    /* ---------- Result ---------- */
    .result-shell {
        position: relative; overflow: hidden; background: var(--surface);
        border: 1px solid var(--line); border-top: 5px solid var(--mint);
        border-radius: 28px; padding: 30px 32px; box-shadow: var(--shadow-lg);
        display: flex; flex-direction: column; justify-content: center;
    }
    .result-shell.mid {border-top-color: var(--amber);}
    .result-shell.high {border-top-color: var(--coral);}
    .result-shell::before {
        content: ""; position: absolute; width: 320px; height: 320px; right: -120px; top: -140px;
        border-radius: 50%; opacity: .2; filter: blur(6px);
    }
    .result-shell.low::before {background: radial-gradient(circle, var(--mint), transparent 70%);}
    .result-shell.mid::before {background: radial-gradient(circle, var(--amber), transparent 70%);}
    .result-shell.high::before {background: radial-gradient(circle, var(--coral), transparent 70%);}
    .result-top {display: flex; align-items: center; justify-content: space-between; gap: .8rem; flex-wrap: wrap;}
    .result-eyebrow {color: var(--muted); font-size: .86rem; font-weight: 600;}
    .result-status {display: inline-flex; align-items: center; gap: .45rem; border-radius: 999px; padding: .4rem .85rem; font-size: .8rem; font-weight: 700;}
    .result-shell.low .result-status {color: #0A7A54; background: #E6F8F0; border: 1px solid #C4EEDB;}
    .result-shell.mid .result-status {color: #9A6200; background: #FFF5DD; border: 1px solid #FFE2A3;}
    .result-shell.high .result-status {color: #C2264A; background: #FFEDF1; border: 1px solid #FFCFDA;}
    .result-number {
        font-family: var(--font-head); font-size: clamp(3.8rem, 7vw, 5.8rem);
        font-weight: 800; letter-spacing: -.06em; line-height: 1; margin: .7rem 0 1.2rem;
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }
    .result-shell.low .result-number {background-image: linear-gradient(120deg, #0E9F6E, #19D3C5);}
    .result-shell.mid .result-number {background-image: linear-gradient(120deg, #FFB020, #FF8A5C);}
    .result-shell.high .result-number {background-image: linear-gradient(120deg, #FF5C7A, #FF8A5C);}
    .meter {position: relative; height: 14px; border-radius: 999px; background: #EEEAFB; overflow: visible;}
    .meter-fill {height: 100%; border-radius: 999px;}
    .result-shell.low .meter-fill {background: linear-gradient(90deg, #22C58B, #19D3C5);}
    .result-shell.mid .meter-fill {background: linear-gradient(90deg, #FFD166, #FFB020);}
    .result-shell.high .meter-fill {background: linear-gradient(90deg, #FF8A5C, #FF5C7A);}
    .meter-mark {position: absolute; top: -5px; width: 3px; height: 24px; border-radius: 3px; background: var(--ink); transform: translateX(-50%);}
    .meter-scale {display: flex; justify-content: space-between; color: var(--muted); font-size: .76rem; font-weight: 500; margin-top: .55rem;}
    .advice {
        background: linear-gradient(120deg, #140A30, #3B1B7A); color: #F1E8FF;
        border-radius: 20px; padding: 1.05rem 1.25rem; font-size: .92rem;
        line-height: 1.65; margin-top: 1rem; box-shadow: var(--shadow-md);
        border: 1px solid rgba(255,255,255,.1);
    }
    .insight {
        background: #fff; border: 1px solid var(--line); border-left: 4px solid var(--indigo);
        border-radius: 14px; padding: .85rem 1.05rem; margin-bottom: .6rem;
        font-size: .92rem; color: var(--text); box-shadow: var(--shadow-sm);
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: 20px;
        padding: 20px 22px; box-shadow: var(--shadow-sm); position: relative; overflow: hidden;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    div[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: var(--shadow-md);}
    div[data-testid="stMetric"]::before {
        content: ""; position: absolute; left: 0; top: 16px; bottom: 16px; width: 4px;
        border-radius: 0 4px 4px 0; background: linear-gradient(180deg, var(--indigo), var(--aqua));
    }
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] *,
    div[data-testid="stMetricLabel"] p {
        color: #4A3F73 !important;
        -webkit-text-fill-color: #4A3F73 !important;
        font-weight: 700 !important;
        font-size: .82rem !important;
        letter-spacing: .02em;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"] {font-family: var(--font-head); color: var(--text); font-weight: 700; letter-spacing: -.03em;}

    /* ---------- Tabs / charts ---------- */
    div[data-baseweb="tab-list"] {gap: .4rem; background: #EAE5FA; padding: .3rem; border-radius: 16px; width: fit-content;}
    button[data-baseweb="tab"] {font-weight: 600; border-radius: 12px; padding: .5rem 1.2rem; height: auto;}
    button[data-baseweb="tab"][aria-selected="true"] {background: #fff; color: var(--indigo); box-shadow: var(--shadow-sm);}
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display: none;}
    div[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid var(--line); border-radius: 22px; padding: .5rem; box-shadow: var(--shadow-sm);}

    .footer-note {color: #8A91AD; font-size: .8rem; text-align: center; padding: 2rem 0 .2rem;}
    hr {border-color: var(--line) !important; margin: 1.4rem 0 !important;}

    @media (max-width: 700px) {
        .block-container {padding-left: .9rem; padding-right: .9rem; padding-top: .8rem;}
        .hero {padding: 32px 22px 30px; border-radius: 24px;}
        div[data-testid="stForm"] {padding: 1rem .9rem 1.1rem; border-radius: 22px;}
        .result-shell {padding: 22px; border-radius: 22px;}
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-dot, .hero-card {animation: none;}
        * {transition: none !important;}
    }
    /* ---------- Readability + motion polish ---------- */
    div[data-testid="stCaptionContainer"],
    div[data-testid="stCaptionContainer"] * {color: #5B5483 !important; font-size: .86rem !important;}
    button[data-baseweb="tab"] p {color: #4A3F73 !important; font-weight: 600 !important;}
    button[data-baseweb="tab"][aria-selected="true"] p {color: var(--indigo) !important;}
    div[data-testid="stExpander"] details summary p {color: var(--text) !important;}
    @keyframes rise {from {opacity: 0; transform: translateY(14px);} to {opacity: 1; transform: none;}}
    .result-shell, .advice, .insight, div[data-testid="stMetric"], div[data-testid="stPlotlyChart"] {
        animation: rise .6s ease backwards;
    }
    .scenario-note {
        background: linear-gradient(90deg, rgba(124,58,237,.10), rgba(25,211,197,.08));
        border: 1px solid rgba(124,58,237,.2); border-radius: 16px;
        padding: .9rem 1.1rem; color: #3B3360; font-size: .92rem; line-height: 1.6; margin-top: .8rem;
    }
    @media (prefers-reduced-motion: reduce) {
        .result-shell, .advice, .insight, div[data-testid="stMetric"], div[data-testid="stPlotlyChart"] {animation: none;}
    }

    /* ---------- Metric headings (robust) + primary buttons + hero CTA ---------- */
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] label *,
    div[data-testid="stMetric"] > div:first-child,
    div[data-testid="stMetric"] > div:first-child *,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        color: #4A3F73 !important;
        -webkit-text-fill-color: #4A3F73 !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        font-size: .82rem !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] * {
        color: #1B2140 !important;
        -webkit-text-fill-color: #1B2140 !important;
        font-size: 2rem !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(95deg, #6D28D9 0%, var(--violet) 60%, #C084FC 100%);
        color: #fff; border: 0; min-height: 52px; font-family: var(--font-head);
        box-shadow: 0 18px 34px -16px rgba(124,58,237,.8);
    }
    .stButton > button[kind="primary"]:hover {color: #fff; filter: brightness(1.07); transform: translateY(-2px);}
    .hero-cta {
        display: inline-block; margin-top: 1.5rem; padding: .7rem 1.3rem; border-radius: 14px;
        background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.32);
        color: #fff !important; font-weight: 600; font-size: .9rem; text-decoration: none !important;
        backdrop-filter: blur(10px); transition: background .2s ease, transform .2s ease;
    }
    .hero-cta:hover {background: rgba(255,255,255,.26); transform: translateY(-2px);}
    div[data-testid="stFileUploader"] section {
        background: #FBFAFF; border: 2px dashed #CDBDF5; border-radius: 18px;
    }
    div[data-testid="stFileUploader"] section * {color: #3B3360 !important;}
    div[data-testid="stFileUploader"] section button {background: #fff !important; border: 1px solid var(--line) !important;}
    div[data-testid="stDataFrame"] {border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow-sm);}

    </style>
    """


def apply_style() -> None:
    st.markdown(STYLE, unsafe_allow_html=True)


MONTHS = [
    "Most recent",
    "1 month ago",
    "2 months ago",
    "3 months ago",
    "4 months ago",
    "5 months ago",
]

STATUS_MAP = {
    "Special Status (-2)": -2,
    "Special Status (-1)": -1,
    "No Delay": 0,
    "1 Month Delay": 1,
    "2 Months Delay": 2,
    "3 Months Delay": 3,
    "4 Months Delay": 4,
    "5 Months Delay": 5,
    "6 Months Delay": 6,
    "7 Months Delay": 7,
    "8 Months Delay": 8,
}

DEFAULT_HISTORY = pd.DataFrame(
    {
        "Month": MONTHS,
        "Repayment Status": ["No Delay"] * 6,
        "Statement Balance": [0.0] * 6,
        "Payment Made": [0.0] * 6,
    }
)

DEMO_HISTORY = pd.DataFrame(
    {
        "Month": MONTHS,
        "Repayment Status": ["No Delay"] * 6,
        "Statement Balance": [
            50000.0, 45000.0, 40000.0,
            35000.0, 30000.0, 25000.0,
        ],
        "Payment Made": [5000.0] * 6,
    }
)

INDIGO = "#7C3AED"

AQUA = "#19D3C5"

CORAL = "#FF5C7A"

MINT = "#22C58B"

AMBER = "#FFB020"

INK = "#0E1330"

def section(step: str, title: str, copy: str) -> None:
    """Render a numbered section heading."""
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-step">{step}</div>
            <div>
                <div class="section-title">{title}</div>
                <div class="section-copy">{copy}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def build_customer_data(
    limit_bal: float,
    sex: int,
    education: int,
    marriage: int,
    age: int,
    history_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[int], List[float], List[float]]:
    """Convert form inputs into the single-row frame the model expects."""
    pay_status = (
        history_df["Repayment Status"]
        .map(STATUS_MAP)
        .astype(int)
        .tolist()
    )
    bills = history_df["Statement Balance"].astype(float).tolist()
    payments = history_df["Payment Made"].astype(float).tolist()

    customer_data = pd.DataFrame(
        [{
            "LIMIT_BAL": float(limit_bal),
            "SEX": int(sex),
            "EDUCATION": int(education),
            "MARRIAGE": int(marriage),
            "AGE": int(age),
            "PAY_0": pay_status[0],
            "PAY_2": pay_status[1],
            "PAY_3": pay_status[2],
            "PAY_4": pay_status[3],
            "PAY_5": pay_status[4],
            "PAY_6": pay_status[5],
            "BILL_AMT1": bills[0],
            "BILL_AMT2": bills[1],
            "BILL_AMT3": bills[2],
            "BILL_AMT4": bills[3],
            "BILL_AMT5": bills[4],
            "BILL_AMT6": bills[5],
            "PAY_AMT1": payments[0],
            "PAY_AMT2": payments[1],
            "PAY_AMT3": payments[2],
            "PAY_AMT4": payments[3],
            "PAY_AMT5": payments[4],
            "PAY_AMT6": payments[5],
        }]
    )
    return customer_data, pay_status, bills, payments

def validate_inputs(
    limit_bal: float,
    age: int,
    pay_status: List[int],
    bills: List[float],
    payments: List[float],
) -> List[str]:
    """Return a list of validation error messages (empty when valid)."""
    errors = []
    if limit_bal <= 0:
        errors.append("Credit limit must be greater than zero.")
    if not 18 <= age <= 100:
        errors.append("Age must be between 18 and 100.")
    if any(x < -2 or x > 8 for x in pay_status):
        errors.append("A repayment status is outside the model's accepted range.")
    if any(x < 0 for x in payments):
        errors.append("Payments cannot be negative.")
    if not all(np.isfinite(x) for x in bills + payments):
        errors.append("Financial values must be finite numbers.")
    return errors

def build_insights(result: Dict[str, Any]) -> List[str]:
    """Turn the engineered metrics into plain-English takeaways."""
    notes: List[str] = []

    # Repayment behaviour
    if result["max_delay"] >= 2:
        notes.append(
            f"Repayment was delayed by up to {result['max_delay']} months."
        )
    elif result["delayed_months"] == 0:
        notes.append(
            "No delayed repayments in the six-month window."
        )

    # Balance / utilization behaviour
    if result["avg_bill"] <= 0:
        notes.append(
            "No outstanding statement balance was recorded in the six-month window."
        )
    elif result["avg_utilization"] >= 0.8:
        notes.append(
            f"High average utilization ({result['avg_utilization'] * 100:.0f}% of limit)."
        )
    elif result["avg_utilization"] <= 0.3:
        notes.append(
            f"Low average utilization ({result['avg_utilization'] * 100:.0f}% of limit)."
        )

    # Payment-to-bill behaviour
    if result["avg_bill"] <= 0:
        notes.append(
            "Payment-to-bill ratio is not applicable because there is no outstanding balance."
        )
    elif result["pay_to_bill"] < 0.3:
        notes.append(
            f"Payments cover only {result['pay_to_bill'] * 100:.0f}% of balances on average."
        )
    elif result["pay_to_bill"] >= 0.8:
        notes.append(
            f"Payments cover {min(result['pay_to_bill'], 1) * 100:.0f}%+ of balances."
        )

    return notes

def scenario_chart(scenarios: pd.DataFrame, cutoff: float) -> go.Figure:
    """Bar chart of default probability under different payment levels."""
    colors = [
        {"High": CORAL, "Moderate": AMBER, "Low": MINT}[level]
        for level in scenarios["Risk Level"]
    ]
    fig = go.Figure(
        go.Bar(
            x=scenarios["Scenario"],
            y=scenarios["Probability"],
            marker=dict(color=colors, cornerradius=10),
            text=[f"{v:.1f}%" for v in scenarios["Probability"]],
            textposition="outside",
            hovertemplate="%{x}<br>Default probability: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=cutoff * 100,
        line_dash="dash",
        line_color=INK,
        opacity=.55,
        annotation_text=f"Threshold {cutoff * 100:.0f}%",
        annotation_position="top left",
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    polish_chart(fig, "Default probability vs payment level", "Probability (%)", 380)
    fig.update_yaxes(range=[0, max(100 * cutoff * 1.4, float(scenarios["Probability"].max()) * 1.25)])
    return fig

def batch_histogram(results: pd.DataFrame, cutoff: float) -> go.Figure:
    """Histogram of predicted default probabilities across the batch."""
    fig = go.Figure(
        go.Histogram(
            x=results["Default Probability (%)"],
            nbinsx=20,
            marker=dict(color=INDIGO, line=dict(color="#FFFFFF", width=1)),
            hovertemplate="%{x}%<br>Customers: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=cutoff * 100, line_dash="dash", line_color=CORAL,
        annotation_text=f"Threshold {cutoff * 100:.0f}%", annotation_position="top",
    )
    fig.update_layout(showlegend=False, xaxis_title="Default probability (%)", yaxis_title=None)
    polish_chart(fig, "Risk distribution across the batch", None, 340)
    return fig

def polish_chart(
    fig: go.Figure,
    title: str,
    y_title: str = None,
    height: int = 350,
) -> go.Figure:
    """Apply the shared chart theme."""
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=58, b=25),
        title=dict(
            text=title,
            x=0.02,
            xanchor="left",
            font=dict(size=17, color=INK, family="Sora, Inter, sans-serif"),
        ),
        font=dict(family="Inter, Arial, sans-serif", color="#5B6485", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=INK, font_color="#FFFFFF", bordercolor=INK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#7A82A1"),
        linecolor="#E6E8F4",
    )
    fig.update_yaxes(
        title=y_title,
        gridcolor="rgba(120,130,170,.14)",
        zeroline=False,
        tickfont=dict(color="#7A82A1"),
    )
    return fig

def monthly_financial_chart(history_df: pd.DataFrame) -> go.Figure:
    """Grouped bars: statement balance versus payment per month."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=history_df["Month"],
            y=history_df["Statement Balance"],
            name="Statement balance",
            marker=dict(color=INDIGO, cornerradius=8),
            hovertemplate="%{x}<br>Balance: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=history_df["Month"],
            y=history_df["Payment Made"],
            name="Payment made",
            marker=dict(color=AQUA, cornerradius=8),
            hovertemplate="%{x}<br>Payment: %{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        bargap=.28,
        bargroupgap=.08,
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
    )
    polish_chart(fig, "Balance vs payment", "Amount", 355)
    return fig

def repayment_status_chart(history_df: pd.DataFrame) -> go.Figure:
    """Line chart of the repayment status code per month."""
    status_numeric = (
        history_df["Repayment Status"]
        .map(STATUS_MAP)
        .astype(int)
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["Month"],
            y=status_numeric,
            mode="lines+markers",
            name="Repayment status",
            line={"width": 3, "color": "#A855F7", "shape": "spline"},
            marker={"size": 10, "color": "#FFFFFF", "line": {"width": 3, "color": "#A855F7"}},
            customdata=history_df["Repayment Status"],
            hovertemplate="%{x}<br>%{customdata}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        opacity=.45,
        annotation_text="No delay",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )
    polish_chart(fig, "Repayment status", "Delay status", 355)
    fig.update_yaxes(dtick=1)
    return fig

def utilization_chart(history_df: pd.DataFrame, limit_bal: float) -> go.Figure:
    """Area chart of statement balance as a percentage of the credit limit."""
    util = np.where(
        float(limit_bal) != 0,
        history_df["Statement Balance"].astype(float) / float(limit_bal) * 100,
        0,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["Month"],
            y=util,
            mode="lines+markers",
            fill="tozeroy",
            fillcolor="rgba(124,58,237,.14)",
            name="Utilization",
            line={"width": 3, "color": INDIGO, "shape": "spline"},
            marker={"size": 9, "color": "#FFFFFF", "line": {"width": 3, "color": INDIGO}},
            hovertemplate="%{x}<br>Utilization: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=100,
        line_dash="dot",
        opacity=.45,
        annotation_text="100% of entered limit",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
    )
    polish_chart(fig, "Credit utilization", "Utilization (%)", 340)
    return fig

def risk_gauge_chart(probability: float, cutoff: float, label: str) -> go.Figure:
    """Large three-band gauge (low / moderate / high) with the threshold marked."""
    pct = probability * 100
    threshold_pct = cutoff * 100
    mid_start = threshold_pct * 0.6
    bar_color = {"Low": MINT, "Moderate": AMBER, "High": CORAL}[label]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={
                "suffix": "%",
                "font": {"size": 58, "color": INK, "family": "Sora, Inter, sans-serif"},
                "valueformat": ".1f",
            },
            title={
                "text": f"Default probability · <b>{label} risk</b>",
                "font": {"size": 16, "color": "#6B7391"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickfont": {"size": 11, "color": "#8A91AD"},
                },
                "bar": {"color": bar_color, "thickness": .34},
                "bgcolor": "#EEF0FA",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, mid_start], "color": "#E6F8F0"},
                    {"range": [mid_start, threshold_pct], "color": "#FFF5DD"},
                    {"range": [threshold_pct, 100], "color": "#FFEDF1"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 4},
                    "thickness": .85,
                    "value": threshold_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=380,
        margin=dict(l=30, r=30, t=70, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif"),
    )
    return fig
