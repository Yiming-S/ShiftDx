"""
ShiftDx: Drift Diagnostics for Multi-Session MI-EEG BCI
Entry point for the Streamlit app.
"""

import os
import base64

import streamlit as st

from data_loader import get_data_store

st.set_page_config(page_title="ShiftDx", page_icon="file.svg", layout="wide")

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown('''
<style>
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.08);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
        color: #4F46E5;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    h1 { color: #1E293B; font-weight: 800 !important; }
    h2 { color: #1E293B; font-weight: 700 !important; letter-spacing: -0.3px; }
    h3 { color: #334155; font-weight: 600 !important; }

    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06);
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    details summary {
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
</style>
''', unsafe_allow_html=True)

# ── Sidebar branding ──────────────────────────────────────────────────────────
_svg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "file.svg")
with open(_svg_path, "rb") as _f:
    _svg_b64 = base64.b64encode(_f.read()).decode()

st.sidebar.markdown(
    f'''
    <div style="
        display: flex; flex-direction: column; align-items: center;
        justify-content: center; margin-bottom: 20px; padding-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
    ">
        <img src="data:image/svg+xml;base64,{_svg_b64}" width="80" height="80"
             style="filter: drop-shadow(0px 2px 4px rgba(0,0,0,0.08)); margin-bottom: 8px;">
        <h1 style="
            margin: 0; font-size: 1.5rem; font-weight: 800;
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        ">ShiftDx</h1>
        <p style="
            margin: 4px 0 0 0; font-size: 0.7rem; font-weight: 600;
            color: #64748B; text-align: center; line-height: 1.3;
        ">Drift Diagnostics for MI-EEG BCI</p>
    </div>
    ''',
    unsafe_allow_html=True,
)

# ── Data store & dataset selector ─────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
store = get_data_store(_DATA_DIR)

_options = ["All"] + list(store.datasets) if len(store.datasets) > 1 else list(store.datasets) or ["(none)"]
dataset = st.sidebar.selectbox("Dataset", _options, index=0)

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_resource.clear()
    st.rerun()

# ── Import pages ──────────────────────────────────────────────────────────────
from views import (
    page_1_overview,
    page_2_drift_trajectory,
    page_3_claim1,
    page_4_claim2,
    page_5_claim3,
    page_6_claim4,
    page_7_subject,
    page_8_live_da,
    page_9_multi_metric,
    page_10_sweep,
    page_11_detection,
)


def _wrap(render_func):
    """Wrap a render(store, dataset) function for st.navigation."""
    def _run():
        render_func(store, dataset)
    return _run


pg = st.navigation({
    "Overview": [
        st.Page(_wrap(page_1_overview.render),
                title="Dataset Overview", icon=":material/dashboard:",
                url_path="overview"),
        st.Page(_wrap(page_2_drift_trajectory.render),
                title="Drift Trajectory", icon=":material/show_chart:",
                url_path="drift-trajectory"),
    ],
    "Claim Explorer": [
        st.Page(_wrap(page_3_claim1.render),
                title="Claim 1 — Drift Predicts Loss", icon=":material/trending_down:",
                url_path="claim-1"),
        st.Page(_wrap(page_4_claim2.render),
                title="Claim 2 — DA Decomposition", icon=":material/call_split:",
                url_path="claim-2"),
        st.Page(_wrap(page_5_claim3.render),
                title="Claim 3 — Retraining Gap", icon=":material/refresh:",
                url_path="claim-3"),
        st.Page(_wrap(page_6_claim4.render),
                title="Claim 4 — Feature Robustness", icon=":material/shield:",
                url_path="claim-4"),
    ],
    "Deep Dive": [
        st.Page(_wrap(page_7_subject.render),
                title="Subject Explorer", icon=":material/person_search:",
                url_path="subject-explorer"),
    ],
    "DA Lab (DA4BCI)": [
        st.Page(_wrap(page_8_live_da.render),
                title="Live DA Sandbox", icon=":material/science:",
                url_path="live-da"),
        st.Page(_wrap(page_9_multi_metric.render),
                title="Multi-Metric Drift Panel", icon=":material/tune:",
                url_path="multi-metric"),
        st.Page(_wrap(page_10_sweep.render),
                title="DA Method Sweep", icon=":material/compare_arrows:",
                url_path="da-sweep"),
        st.Page(_wrap(page_11_detection.render),
                title="Drift Detection Demo", icon=":material/sensors:",
                url_path="drift-detection"),
    ],
})

pg.run()

# ── Author info ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.75rem; color:#94A3B8; line-height:1.5;">'
    '<b>Yiming Shen</b> &amp; <b>David Degras</b><br>'
    'Department of Mathematics<br>'
    'University of Massachusetts Boston'
    '</div>',
    unsafe_allow_html=True,
)
