# dashboard/app.py
# PayLens Streamlit dashboard.
# Input form → prediction → history. Reads all data from Supabase.

import os
import sys
import joblib
import requests
import streamlit as st
from supabase import create_client

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import config
from pipeline.predict import run_prediction

st.set_page_config(
    page_title="PayLens",
    page_icon="💰",
    layout="wide"
)

# Initialize Supabase client once
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

# @st.cache_resource loads this once per session (not on every re-render)
@st.cache_resource
def load_artifacts():
    """Load model artifacts needed for dropdowns and feature importance."""
    encoders   = joblib.load(os.path.join(ROOT_DIR, "model", "encoders.pkl"))
    thresholds = joblib.load(os.path.join(ROOT_DIR, "model", "thresholds.pkl"))
    benchmarks = joblib.load(os.path.join(ROOT_DIR, "model", "benchmarks.pkl"))
    model      = joblib.load(config.MODEL_PATH)
    return encoders, thresholds, benchmarks, model

encoders, thresholds, benchmarks, model = load_artifacts()

# Derive dropdown options from encoders (same values as training)
EXP_OPTIONS  = list(encoders["experience_level"].classes_)
EMP_OPTIONS  = list(encoders["employment_type"].classes_)
SIZE_OPTIONS = list(encoders["company_size"].classes_)
COUNTRY_OPTIONS = list(encoders["employee_residence"].classes_)
TITLE_OPTIONS   = list(encoders["job_title"].classes_)

# Human-readable labels for experience level codes
EXP_LABELS = {"EN": "EN — Entry-level", "MI": "MI — Mid-level", "SE": "SE — Senior", "EX": "EX — Executive"}
EMP_LABELS = {"FT": "FT — Full-time", "PT": "PT — Part-time", "CT": "CT — Contract", "FL": "FL — Freelance"}
SIZE_LABELS = {"S": "S — Small", "M": "M — Medium", "L": "L — Large"}


def _show_prediction_result(result):
    """Display a single prediction result: tier badge, confidence, salary range, narrative, charts."""
    tier = result.get("prediction", "Unknown")
    confidence = result.get("confidence_pct", 0)
    narrative = result.get("narrative", "No narrative available.")
    salary_range = result.get("salary_range", {})
    matched_title = result.get("matched_job_title", "")
    match_score = result.get("match_score", 0)
    benchmark = result.get("benchmark", {})

    # Tier badge with color
    tier_colors = {"Low": "🟢", "Mid": "🟡", "High": "🔴"}
    icon = tier_colors.get(tier, "⚪")

    st.markdown(f"## {icon} {tier} Salary Tier")

    # Key metrics in columns
    m1, m2, m3 = st.columns(3)
    m1.metric("Confidence", f"{confidence}%")
    m2.metric("Matched Title", matched_title)
    m3.metric("Match Score", f"{round(match_score * 100)}%")

    # Salary range
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max")
    range_str = f"${sal_min:,} – ${sal_max:,}" if sal_max else f"Above ${sal_min:,}"
    st.info(f"**Salary Range:** {range_str}")

    # Benchmark
    if benchmark:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Peer Median", f"${benchmark.get('median', 0):,}")
        b2.metric("25th Percentile", f"${benchmark.get('p25', 0):,}")
        b3.metric("75th Percentile", f"${benchmark.get('p75', 0):,}")
        b4.metric("Peer Count", benchmark.get("peer_count", 0))

    # Narrative
    st.subheader("📝 Analyst Report")
    st.write(narrative or "No narrative available.")

    # Charts
    chart_overview = result.get("chart_overview_url")
    chart_peer = result.get("chart_peer_url")

    if chart_overview or chart_peer:
        st.subheader("📊 Visualizations")
        c1, c2 = st.columns(2)
        if chart_overview:
            c1.image(chart_overview, caption="Salary by Experience Level", use_container_width=True)
        if chart_peer:
            c2.image(chart_peer, caption="Your Position vs Peers", use_container_width=True)

    # Download report button
    report_text = f"""PayLens Salary Report
========================
Job Title: {matched_title}
Predicted Tier: {tier} ({confidence}% confidence)
Salary Range: {range_str}

Peer Benchmark:
  Median: ${benchmark.get('median', 0):,}
  25th percentile: ${benchmark.get('p25', 0):,}
  75th percentile: ${benchmark.get('p75', 0):,}
  Peer count: {benchmark.get('peer_count', 0)}

Analyst Report:
{narrative}
"""
    st.download_button(
        label="⬇️ Download Report",
        data=report_text,
        file_name=f"paylens_{matched_title.replace(' ', '_').lower()}.txt",
        mime="text/plain"
    )


st.title("💰 PayLens")
st.caption("Intelligent salary prediction for data professionals")
st.divider()

tab_predict, tab_history, tab_market = st.tabs(["🔮 Predict", "📋 History", "📊 Market Insights"])

with tab_predict:
    col_form, col_result = st.columns([1, 2])

    with col_form:
        st.subheader("Your Job Details")

        experience = st.selectbox(
            "Experience Level",
            options=EXP_OPTIONS,
            format_func=lambda x: EXP_LABELS.get(x, x)
        )
        employment = st.selectbox(
            "Employment Type",
            options=EMP_OPTIONS,
            format_func=lambda x: EMP_LABELS.get(x, x)
        )
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g. Data Scientist, ML Engineer",
            help="We'll match this to the nearest known title automatically"
        )
        residence = st.selectbox("Country of Residence", options=COUNTRY_OPTIONS)
        remote = st.select_slider(
            "Remote Ratio",
            options=[0, 50, 100],
            value=100,
            format_func=lambda x: {0: "On-site", 50: "Hybrid", 100: "Full Remote"}[x]
        )
        company_loc = st.selectbox("Company Location", options=COUNTRY_OPTIONS)
        company_size = st.selectbox(
            "Company Size",
            options=SIZE_OPTIONS,
            format_func=lambda x: SIZE_LABELS.get(x, x)
        )

        predict_btn = st.button("🔮 Predict Salary", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            if not job_title.strip():
                st.warning("Please enter a job title.")
            else:
                job_input = {
                    "experience_level":   experience,
                    "employment_type":    employment,
                    "job_title":          job_title.strip(),
                    "employee_residence": residence,
                    "remote_ratio":       remote,
                    "company_location":   company_loc,
                    "company_size":       company_size
                }

                with st.spinner("Predicting... (Ollama may take up to 30s)"):
                    result = run_prediction(job_input)

                if result is None:
                    st.error("Prediction failed. Make sure the FastAPI server is running on port 8000.")
                else:
                    # Store result in session state so it persists on re-render
                    st.session_state["last_result"] = result

        # Display result from session state (persists across re-renders)
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]
            _show_prediction_result(result)
