import streamlit as st

from shared_core import initialize_session_state, reset_session
from page_a_upload_overview import render_page_a
from page_b_cleaning_preparation import render_page_b
from page_c_visualization_builder import render_page_c
from page_d_export_report import render_page_d


st.set_page_config(
    page_title="AI-Assisted Data Wrangler & Visualizer",
    layout="wide"
)

initialize_session_state()

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to page:",
    [
        "Page A — Upload & Overview",
        "Page B — Cleaning & Preparation Studio",
        "Page C — Visualization Builder",
        "Page D — Export & Report"
    ]
)

st.sidebar.markdown("---")
if st.sidebar.button("Reset session", use_container_width=True):
    reset_session()

if page == "Page A — Upload & Overview":
    render_page_a()
elif page == "Page B — Cleaning & Preparation Studio":
    render_page_b()
elif page == "Page C — Visualization Builder":
    render_page_c()
elif page == "Page D — Export & Report":
    render_page_d()
