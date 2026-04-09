# dashboard/app.py
# PayLens Streamlit dashboard.
# Input form → prediction → history. Reads all data from Supabase.

import io
import os
import sys
import joblib
import requests
import pandas as pd
import matplotlib.pyplot as plt
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


@st.cache_data(ttl=30)  # refresh every 30 seconds so new predictions appear
def load_history():
    """Fetch all prediction rows from Supabase, newest first."""
    try:
        resp = supabase.table("predictions").select("*").order("created_at", desc=True).execute()
        if resp.data:
            return pd.DataFrame(resp.data)
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load history: {e}")
        return pd.DataFrame()


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


def _show_feature_importance(model):
    """Display a horizontal bar chart of the model's feature importances."""
    try:
        feature_names = model.feature_names_in_
        importances = model.feature_importances_
    except AttributeError:
        st.caption("Feature importance not available for this model.")
        return

    # Sort by importance descending
    sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_pairs)

    # Clean up display names (remove "Unnamed: 0" label)
    display_names = []
    for n in names:
        if n == "Unnamed: 0":
            display_names.append("Row Index (artifact)")
        else:
            display_names.append(n.replace("_", " ").title())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(display_names, values, color="#3498db")
    ax.set_xlabel("Importance")
    ax.set_title("What Drives Your Salary Prediction")
    ax.spines[["top", "right"]].set_visible(False)
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _show_benchmark_yourself(result):
    """
    Visual comparison: user's predicted salary range vs. peer benchmark bars.
    Shows where the user lands relative to p25, median, p75.
    """
    benchmark = result.get("benchmark", {})
    salary_range = result.get("salary_range", {})

    median = benchmark.get("median", 0)
    p25    = benchmark.get("p25", 0)
    p75    = benchmark.get("p75", 0)
    sal_min = salary_range.get("min", 0)
    sal_max = salary_range.get("max")  # None for High tier

    # Use salary_min as the representative "your salary" point
    your_salary = sal_min

    fig, ax = plt.subplots(figsize=(7, 3))

    # Background band for p25–p75
    ax.barh(["Peer Range"], [p75 - p25], left=p25,
            color="#3498db", alpha=0.3, height=0.4, label="Peer 25–75th %ile")

    # Median line
    ax.axvline(median, color="#2980b9", linewidth=2, linestyle="--", label=f"Peer Median ${median:,}")

    # Your predicted salary floor
    ax.axvline(your_salary, color="#e74c3c", linewidth=2.5, label=f"Your Salary Min ${your_salary:,}")

    ax.set_xlabel("Annual Salary (USD)")
    ax.set_title("Your Prediction vs. Peer Benchmark")
    ax.legend(loc="upper left", fontsize=8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])

    # Format x-axis with commas
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


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

    # Feature importance
    with st.expander("🔍 Why did the model predict this? (Feature Importance)", expanded=False):
        _show_feature_importance(model)

    # Benchmark yourself visual
    with st.expander("📐 Benchmark Yourself", expanded=True):
        _show_benchmark_yourself(result)


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

with tab_history:
    st.subheader("📋 Prediction History")

    if st.button("🔄 Refresh", key="refresh_history"):
        st.cache_data.clear()

    df = load_history()

    if df.empty:
        st.info("No predictions yet. Run a prediction in the Predict tab first!")
    else:
        # Summary metrics at the top
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Predictions", len(df))
        h2.metric("Most Common Tier", df["predicted_tier"].mode()[0] if not df.empty else "—")
        h3.metric("Avg Confidence", f"{df['confidence_pct'].mean():.1f}%" if not df.empty else "—")

        # Display table with selected columns only
        display_cols = ["created_at", "matched_job_title", "experience_level",
                        "predicted_tier", "confidence_pct", "salary_min", "salary_max",
                        "benchmark_median"]

        # Rename for readability
        col_labels = {
            "created_at": "Date",
            "matched_job_title": "Job Title",
            "experience_level": "Experience",
            "predicted_tier": "Tier",
            "confidence_pct": "Confidence %",
            "salary_min": "Salary Min",
            "salary_max": "Salary Max",
            "benchmark_median": "Peer Median"
        }

        # Format the date to be shorter
        df_display = df[display_cols].copy()
        df_display["created_at"] = pd.to_datetime(df_display["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df_display = df_display.rename(columns=col_labels)

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Expandable detail view for each row
        st.subheader("🔍 Row Detail")
        selected_idx = st.number_input(
            "Enter row number to expand (0 = newest):",
            min_value=0, max_value=max(0, len(df)-1), value=0, step=1
        )

        if st.button("📂 Expand Row", key="expand_row"):
            row = df.iloc[int(selected_idx)]

            tier_colors = {"Low": "🟢", "Mid": "🟡", "High": "🔴"}
            icon = tier_colors.get(row.get("predicted_tier", ""), "⚪")

            st.markdown(f"### {icon} {row.get('predicted_tier', 'Unknown')} Tier — {row.get('matched_job_title', '')}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Confidence", f"{row.get('confidence_pct', 0)}%")
            d2.metric("Experience", row.get("experience_level", "—"))
            d3.metric("Remote", f"{row.get('remote_ratio', '—')}%")

            sal_min = row.get("salary_min", 0)
            sal_max = row.get("salary_max")
            range_str = f"${int(sal_min):,} – ${int(sal_max):,}" if sal_max else f"Above ${int(sal_min):,}"
            st.info(f"**Salary Range:** {range_str}")

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Peer Median", f"${int(row.get('benchmark_median', 0)):,}")
            b2.metric("25th %ile", f"${int(row.get('benchmark_p25', 0)):,}")
            b3.metric("75th %ile", f"${int(row.get('benchmark_p75', 0)):,}")
            b4.metric("Peer Count", int(row.get("benchmark_peers", 0)))

            st.subheader("📝 Analyst Report")
            st.write(row.get("narrative", "No narrative available."))

            # Charts from stored URLs
            chart_overview = row.get("chart_overview_url")
            chart_peer = row.get("chart_peer_url")
            if chart_overview or chart_peer:
                c1, c2 = st.columns(2)
                if chart_overview:
                    c1.image(chart_overview, caption="Salary by Experience Level", use_container_width=True)
                if chart_peer:
                    c2.image(chart_peer, caption="Your Position vs Peers", use_container_width=True)

with tab_market:
    st.subheader("📊 Market Insights")
    st.caption("Aggregated from all predictions in the database")

    df_market = load_history()

    if df_market.empty:
        st.info("No data yet. Run some predictions first to see market insights!")
    else:
        mi1, mi2 = st.columns(2)

        with mi1:
            st.markdown("**Tier Distribution**")
            tier_counts = df_market["predicted_tier"].value_counts()
            fig1, ax1 = plt.subplots(figsize=(5, 3))
            colors = {"Low": "#2ecc71", "Mid": "#f39c12", "High": "#e74c3c"}
            bar_colors = [colors.get(t, "#999") for t in tier_counts.index]
            ax1.bar(tier_counts.index, tier_counts.values, color=bar_colors)
            ax1.set_ylabel("Count")
            ax1.set_title("Salary Tier Distribution")
            ax1.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig1)
            plt.close(fig1)

        with mi2:
            st.markdown("**Avg Confidence by Tier**")
            conf_by_tier = df_market.groupby("predicted_tier")["confidence_pct"].mean().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            bar_colors2 = [colors.get(t, "#999") for t in conf_by_tier.index]
            ax2.bar(conf_by_tier.index, conf_by_tier.values, color=bar_colors2)
            ax2.set_ylabel("Avg Confidence %")
            ax2.set_title("Confidence by Tier")
            ax2.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)
            plt.close(fig2)

        mi3, mi4 = st.columns(2)

        with mi3:
            st.markdown("**Predictions by Experience Level**")
            exp_counts = df_market["experience_level"].value_counts()
            exp_labels = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
            exp_counts.index = [exp_labels.get(x, x) for x in exp_counts.index]
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.barh(exp_counts.index, exp_counts.values, color="#3498db")
            ax3.set_xlabel("Count")
            ax3.set_title("Predictions by Experience")
            ax3.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig3)
            plt.close(fig3)

        with mi4:
            st.markdown("**Remote Work Distribution**")
            remote_counts = df_market["remote_ratio"].value_counts().sort_index()
            remote_labels = {0: "On-site", 50: "Hybrid", 100: "Full Remote"}
            remote_counts.index = [remote_labels.get(x, str(x)) for x in remote_counts.index]
            fig4, ax4 = plt.subplots(figsize=(5, 3))
            ax4.pie(remote_counts.values, labels=remote_counts.index,
                    autopct="%1.0f%%", colors=["#e74c3c", "#f39c12", "#2ecc71"],
                    startangle=90)
            ax4.set_title("Remote Work Split")
            st.pyplot(fig4)
            plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# Premium Pitch Section
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## ✨ PayLens Premium — *Coming Soon*")

pr1, pr2, pr3 = st.columns(3)

with pr1:
    st.markdown("""
**🤖 Real-Time LLM Coaching**
Get a personalised salary negotiation script generated by GPT-4o, tailored to your exact title, location, and tier.
""")

with pr2:
    st.markdown("""
**📈 Salary Trajectory Planner**
See a 3-year salary growth projection based on market trends and your current experience level.
""")

with pr3:
    st.markdown("""
**🌍 Global Salary Comparison**
Compare your salary tier across 50+ countries — identify relocation opportunities and remote premium markets.
""")

st.button("💬 Join the Waitlist", type="secondary", disabled=True)
st.caption("Premium features are currently in development. Join the waitlist to be notified at launch.")
