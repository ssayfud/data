import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import base64

try:
    from google import genai
except Exception:
    genai = None


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI-Assisted Data Wrangler & Visualizer",
    layout="wide"
)

DEFAULT_STATE = {
    "original_df": None,
    "working_df": None,
    "file_name": None,
    "transformation_log": [],
    "recipe_steps": [],
    "history_stack": [],
    "validation_violations_df": pd.DataFrame(),
    "dashboard_saved_charts": [],
    "last_generated_chart": None,
    "ai_cleaning_plan": None,
    "ai_cleaning_feedback": "",
    "ai_chart_suggestions": ""
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

def reset_session():
    for key in DEFAULT_STATE:
        st.session_state[key] = DEFAULT_STATE[key]

    # Clear dashboard + charts
    st.session_state["dashboard_saved_charts"] = []
    st.session_state["last_generated_chart"] = None

    st.rerun()
def initialize_session_state():
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value

def log_step(step_name, parameters=None, affected_columns=None):
    if parameters is None:
        parameters = {}
    if affected_columns is None:
        affected_columns = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.transformation_log.append({
        "step": step_name,
        "parameters": parameters,
        "affected_columns": affected_columns,
        "timestamp": timestamp
    })

    st.session_state.recipe_steps.append({
        "step": step_name,
        "parameters": parameters,
        "affected_columns": affected_columns
    })


def update_working_df(new_df):
    st.session_state.working_df = new_df.copy()



def push_history_state(df_snapshot):
    if df_snapshot is not None:
        st.session_state.history_stack.append(df_snapshot.copy())


def undo_last_step():
    if not st.session_state.history_stack:
        return False, "There is no step to undo."

    previous_df = st.session_state.history_stack.pop()
    st.session_state.working_df = previous_df.copy()

    if st.session_state.transformation_log:
        st.session_state.transformation_log.pop()

    if st.session_state.recipe_steps:
        st.session_state.recipe_steps.pop()

    return True, "Last transformation was undone successfully."


def render_workflow_controls(show_metrics=True):
    st.write("### Workflow Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)

    with ctrl1:
        if st.button("Undo last step", use_container_width=True, key="undo_last_step_btn"):
            ok, message = undo_last_step()
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.info(message)

    with ctrl2:
        if st.button("Reset all transformations", use_container_width=True, key="reset_all_transformations_btn"):
            if st.session_state.original_df is not None:
                st.session_state.working_df = st.session_state.original_df.copy()
                st.session_state.transformation_log = []
                st.session_state.recipe_steps = []
                st.session_state.history_stack = []
                st.success("All transformations were reset. The original dataset has been restored.")
                st.rerun()
            else:
                st.warning("No original dataset is available to restore.")

    with ctrl3:
        st.metric("Undo steps available", len(st.session_state.history_stack))

    if show_metrics:
        st.caption("Use Undo to reverse the most recent logged transformation, or Reset to restore the original uploaded dataset.")



@st.cache_data(show_spinner=False)
def _read_csv_cached(file_bytes):
    import io
    import pandas as pd

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
        "cp1251",
        "iso-8859-1"
    ]
    separators = [None, ",", ";", "\t", "|"]

    last_error = None

    for enc in encodings:
        for sep in separators:
            try:
                if sep is None:
                    return pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding=enc,
                        sep=None,
                        engine="python",
                        on_bad_lines="skip"
                    )
                else:
                    return pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding=enc,
                        sep=sep,
                        engine="python",
                        on_bad_lines="skip"
                    )
            except Exception as e:
                last_error = e
                continue

    try:
        text = file_bytes.decode("latin1", errors="replace")
        return pd.read_csv(
            io.StringIO(text),
            sep=None,
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        last_error = e

    raise ValueError(f"Could not read CSV file. Last error: {last_error}")


@st.cache_data(show_spinner=False)
def _read_excel_cached(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _read_json_cached(file_bytes):
    data = json.load(io.BytesIO(file_bytes))
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        return pd.json_normalize(data)
    return None


@st.cache_data(show_spinner=False)
def profile_dataset_cached(df):
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isna().sum().values,
            "Missing %": ((df.isna().sum() / len(df)) * 100).round(2).values
        }).sort_values(by="Missing Count", ascending=False),
        "dtype_info": pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values
        })
    }


@st.cache_data(show_spinner=False)
def load_google_sheet_cached(csv_url):
    return pd.read_csv(csv_url)


def load_file(file):
    file_name = file.name.lower()

    try:
        file_bytes = file.getvalue()

        if file_name.endswith(".csv"):
            return _read_csv_cached(file_bytes)

        elif file_name.endswith(".xlsx"):
            return _read_excel_cached(file_bytes)

        elif file_name.endswith(".json"):
            return _read_json_cached(file_bytes)

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

    return None


def convert_google_sheet_url(url):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        return None

    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def load_google_sheet(url):
    csv_url = convert_google_sheet_url(url)
    if not csv_url:
        return None

    try:
        return load_google_sheet_cached(csv_url)
    except Exception as e:
        st.error(f"Could not load Google Sheet: {e}")
        return None


def before_after_summary(before_df, after_df, affected_columns=None):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows Before", before_df.shape[0])
    col2.metric("Rows After", after_df.shape[0])

    affected_count = len(affected_columns) if affected_columns else 0
    col3.metric("Affected Columns", affected_count)

    if affected_columns:
        st.write("**Affected columns:**", ", ".join(affected_columns))


def show_small_preview(before_df, after_df):
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Before**")
        st.dataframe(before_df.head(8), use_container_width=True)
    with c2:
        st.write("**After**")
        st.dataframe(after_df.head(8), use_container_width=True)


def clean_numeric_series(series):
    cleaned = series.astype(str).str.replace(r"[,$€£% ]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def detect_outliers_iqr(series):
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return pd.Series(False, index=series.index)

    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    numeric_full = pd.to_numeric(series, errors="coerce")
    return (numeric_full < lower) | (numeric_full > upper)


def min_max_scale(series):
    s = pd.to_numeric(series, errors="coerce")
    min_val = s.min()
    max_val = s.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return s

    return (s - min_val) / (max_val - min_val)


def z_score_scale(series):
    s = pd.to_numeric(series, errors="coerce")
    mean_val = s.mean()
    std_val = s.std()

    if pd.isna(std_val) or std_val == 0:
        return s

    return (s - mean_val) / std_val


def get_download_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def get_download_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Cleaned_Data")

        if st.session_state.transformation_log:
            log_df = pd.DataFrame(st.session_state.transformation_log)
            log_df.to_excel(writer, index=False, sheet_name="Transformation_Log")

    output.seek(0)
    return output


def fig_to_png_bytes(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    buffer.seek(0)
    return buffer.getvalue()


def set_last_generated_plotly_chart(fig, title, chart_type):
    st.session_state.last_generated_chart = {
        "render_type": "plotly",
        "title": title,
        "chart_type": chart_type,
        "plotly_json": pio.to_json(fig)
    }


def set_last_generated_matplotlib_chart(fig, title, chart_type):
    png_bytes = fig_to_png_bytes(fig)
    st.session_state.last_generated_chart = {
        "render_type": "matplotlib",
        "title": title,
        "chart_type": chart_type,
        "image_bytes": png_bytes
    }


def add_last_chart_to_dashboard(custom_title=""):
    last_chart = st.session_state.get("last_generated_chart")
    if not last_chart:
        return False, "Generate a chart first on Page C before saving it to the dashboard."

    chart_copy = dict(last_chart)
    if custom_title and custom_title.strip():
        chart_copy["title"] = custom_title.strip()
    chart_copy["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.dashboard_saved_charts.append(chart_copy)
    return True, "Chart added to the dashboard successfully."


def _safe_chart_filename(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "saved_chart"


def build_saved_chart_download(saved_chart, download_title):
    final_title = (download_title or saved_chart.get("title") or "Saved Chart").strip()
    safe_name = _safe_chart_filename(final_title)

    if saved_chart.get("render_type") == "plotly":
        try:
            fig = pio.from_json(saved_chart["plotly_json"])
            fig.update_layout(title=final_title)
            try:
                image_bytes = pio.to_image(fig, format="png", scale=2)
                return image_bytes, f"{safe_name}.png", "image/png"
            except Exception:
                html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
                return html.encode("utf-8"), f"{safe_name}.html", "text/html"
        except Exception:
            fallback_html = f"<html><body><h2>{final_title}</h2><p>Could not rebuild this chart for download.</p></body></html>"
            return fallback_html.encode("utf-8"), f"{safe_name}.html", "text/html"

    image_bytes = saved_chart.get("image_bytes", b"")
    image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else ""
    html = f"""
    <html>
    <head><meta charset='utf-8'><title>{final_title}</title></head>
    <body style='font-family:Arial,sans-serif;background:#f8fafc;color:#0f172a;padding:24px;'>
        <h2 style='margin-top:0;'>{final_title}</h2>
        <p>Chart type: {saved_chart.get('chart_type', 'Chart')}</p>
        <img src='data:image/png;base64,{image_base64}' style='max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:10px;'/>
    </body>
    </html>
    """
    return html.encode("utf-8"), f"{safe_name}.html", "text/html"


def build_saved_charts_bundle_download(saved_charts, bundle_title):
    final_title = (bundle_title or "Saved Dashboard Charts").strip()
    safe_name = _safe_chart_filename(final_title)

    chart_sections = []
    include_plotlyjs = True

    for idx, saved_chart in enumerate(saved_charts, start=1):
        chart_title = saved_chart.get("title", f"Saved Chart {idx}")
        chart_type = saved_chart.get("chart_type", "Chart")
        saved_at = saved_chart.get("saved_at", "")

        if saved_chart.get("render_type") == "plotly":
            try:
                fig = pio.from_json(saved_chart["plotly_json"])
                fig.update_layout(title=chart_title)
                chart_html = pio.to_html(fig, include_plotlyjs="cdn" if include_plotlyjs else False, full_html=False)
                include_plotlyjs = False
                chart_sections.append(f"""
                <section class='chart-card'>
                    <h2>{chart_title}</h2>
                    <p class='meta'>{chart_type}{' • saved ' + saved_at if saved_at else ''}</p>
                    {chart_html}
                </section>
                """)
                continue
            except Exception:
                chart_sections.append(f"""
                <section class='chart-card'>
                    <h2>{chart_title}</h2>
                    <p class='meta'>{chart_type}{' • saved ' + saved_at if saved_at else ''}</p>
                    <p>Could not rebuild this Plotly chart for the downloaded report.</p>
                </section>
                """)
                continue

        image_bytes = saved_chart.get("image_bytes", b"")
        image_base64 = base64.b64encode(image_bytes).decode("utf-8") if image_bytes else ""
        chart_sections.append(f"""
        <section class='chart-card'>
            <h2>{chart_title}</h2>
            <p class='meta'>{chart_type}{' • saved ' + saved_at if saved_at else ''}</p>
            <img src='data:image/png;base64,{image_base64}' alt='{chart_title}' />
        </section>
        """)

    html = f"""
    <html>
    <head>
        <meta charset='utf-8' />
        <title>{final_title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
            .hero {{ padding: 20px; border-radius: 18px; background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%); color: white; margin-bottom: 20px; }}
            .chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 18px; }}
            .chart-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 16px; padding: 16px; box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06); }}
            .chart-card h2 {{ margin: 0 0 8px 0; font-size: 22px; }}
            .meta {{ margin: 0 0 12px 0; color: #475569; font-size: 14px; }}
            img {{ max-width: 100%; height: auto; border-radius: 10px; border: 1px solid #e2e8f0; }}
        </style>
    </head>
    <body>
        <div class='hero'>
            <h1 style='margin:0;'>{final_title}</h1>
            <p style='margin-top:8px;'>Downloaded collection of saved dashboard charts from Page D.</p>
        </div>
        <div class='chart-grid'>
            {''.join(chart_sections) if chart_sections else '<p>No saved charts available.</p>'}
        </div>
    </body>
    </html>
    """
    return html.encode("utf-8"), f"{safe_name}.html", "text/html"


def get_dashboard_html(df, profile_info):
    completeness_pct = 100.0
    total_cells = max(df.shape[0] * df.shape[1], 1)
    completeness_pct = round((1 - (profile_info["missing_cells"] / total_cells)) * 100, 2)
    numeric_count = int(len(df.select_dtypes(include=np.number).columns))
    categorical_count = int(len(df.select_dtypes(include=["object", "category", "bool"]).columns))
    validation_rows = 0 if st.session_state.validation_violations_df is None else len(st.session_state.validation_violations_df)

    preview_html = df.head(15).to_html(index=False, border=0)

    if st.session_state.transformation_log:
        log_df = pd.DataFrame(st.session_state.transformation_log).copy()
        if "parameters" in log_df.columns:
            log_df["parameters"] = log_df["parameters"].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x))
        if "affected_columns" in log_df.columns:
            log_df["affected_columns"] = log_df["affected_columns"].apply(lambda x: ", ".join(x) if isinstance(x, list) else str(x))
        log_html = log_df.to_html(index=False, border=0)
    else:
        log_html = "<p>No transformation steps recorded yet.</p>"

    cards = f"""
    <div class='cards'>
      <div class='card blue'><h3>Rows</h3><div class='num'>{profile_info['rows']:,}</div></div>
      <div class='card violet'><h3>Columns</h3><div class='num'>{profile_info['columns']:,}</div></div>
      <div class='card amber'><h3>Missing Cells</h3><div class='num'>{profile_info['missing_cells']:,}</div></div>
      <div class='card rose'><h3>Duplicate Rows</h3><div class='num'>{profile_info['duplicate_rows']:,}</div></div>
      <div class='card slate'><h3>Completeness %</h3><div class='num'>{completeness_pct}</div></div>
      <div class='card slate'><h3>Numeric Columns</h3><div class='num'>{numeric_count}</div></div>
      <div class='card slate'><h3>Categorical Columns</h3><div class='num'>{categorical_count}</div></div>
      <div class='card slate'><h3>Validation Rows Saved</h3><div class='num'>{validation_rows}</div></div>
    </div>
    """

    html = f"""
    <html>
    <head>
    <meta charset='utf-8' />
    <title>Dashboard Report</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
      .hero {{ padding: 20px; border-radius: 18px; background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%); color: white; margin-bottom: 20px; }}
      .cards {{ display:grid; grid-template-columns: repeat(4, minmax(180px,1fr)); gap:16px; margin-bottom:20px; }}
      .card {{ padding:18px; border-radius:16px; color:white; }}
      .blue {{ background: linear-gradient(135deg, #2563eb, #38bdf8); }}
      .violet {{ background: linear-gradient(135deg, #7c3aed, #a78bfa); }}
      .amber {{ background: linear-gradient(135deg, #d97706, #f59e0b); }}
      .rose {{ background: linear-gradient(135deg, #dc2626, #fb7185); }}
      .slate {{ background: linear-gradient(135deg, #334155, #64748b); }}
      .num {{ font-size: 30px; font-weight: 700; margin-top: 10px; }}
      h2 {{ margin-top: 28px; }}
      table {{ border-collapse: collapse; width:100%; background:white; }}
      th, td {{ border:1px solid #e2e8f0; padding:8px; text-align:left; vertical-align: top; }}
      th {{ background:#e2e8f0; }}
    </style>
    </head>
    <body>
      <div class='hero'>
        <h1 style='margin:0;'>Final Dashboard</h1>
        <p style='margin-top:8px;'>A final report view of the cleaned dataset, workflow history, and export-ready outputs.</p>
      </div>
      {cards}
      <h2>Cleaned Dataset Preview</h2>
      {preview_html}
      <h2>Transformation Log</h2>
      {log_html}
    </body>
    </html>
    """
    return html.encode("utf-8")
