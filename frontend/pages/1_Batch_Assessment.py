"""The original batch page, integrated through the shared API client."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from frontend.components.batch import render_batch
from frontend.components.common import backend, navigation
from frontend.components.ui import apply_style

st.set_page_config(page_title="Batch Credit Risk Assessment", page_icon="📊", layout="wide")
apply_style()
navigation()
api, model_info = backend()
render_batch(api, model_info)
