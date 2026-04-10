# dashboard/app.py
# PayLens Streamlit dashboard.
# Input form → prediction → history. Reads all data from Supabase.

import re
import os
import sys
import joblib
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
    benchmarks = joblib.load(os.path.join(ROOT_DIR, "model", "benchmarks.pkl"))
    model      = joblib.load(config.MODEL_PATH)
    return encoders, benchmarks, model

encoders, benchmarks, model = load_artifacts()



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


# Options in logical order (derived from encoders for valid values, overridden for sort order)
EXP_OPTIONS     = ["EN", "MI", "SE", "EX"]   # Entry → Mid → Senior → Executive
EMP_OPTIONS     = ["FT", "PT", "CT", "FL"]   # Full-time first
SIZE_OPTIONS    = ["S", "M", "L"]            # Small → Medium → Large
COUNTRY_OPTIONS = sorted(encoders["employee_residence"].classes_)
TITLE_OPTIONS   = list(encoders["job_title"].classes_)

# Display labels (no codes shown to end users)
EXP_LABELS  = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
EMP_LABELS  = {"FT": "Full-time",   "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
SIZE_LABELS = {"S": "Small",        "M": "Medium",     "L": "Large"}

# Full country names for display (API still receives 2-letter codes)
COUNTRY_NAMES = {
    "AE": "United Arab Emirates", "AF": "Afghanistan", "AM": "Armenia",
    "AR": "Argentina",            "AT": "Austria",     "AU": "Australia",
    "AZ": "Azerbaijan",           "BA": "Bosnia and Herzegovina", "BD": "Bangladesh",
    "BE": "Belgium",              "BG": "Bulgaria",    "BH": "Bahrain",
    "BO": "Bolivia",              "BR": "Brazil",      "BN": "Brunei",
    "BY": "Belarus",              "CA": "Canada",      "CH": "Switzerland",
    "CI": "Côte d'Ivoire",        "CL": "Chile",       "CM": "Cameroon",
    "CN": "China",                "CO": "Colombia",    "CR": "Costa Rica",
    "CU": "Cuba",                 "CY": "Cyprus",      "CZ": "Czech Republic",
    "DE": "Germany",              "DK": "Denmark",     "DO": "Dominican Republic",
    "EC": "Ecuador",              "EE": "Estonia",     "EG": "Egypt",
    "ES": "Spain",                "FI": "Finland",     "FJ": "Fiji",
    "FR": "France",               "GB": "United Kingdom", "GE": "Georgia",
    "GH": "Ghana",                "GR": "Greece",      "GT": "Guatemala",
    "HK": "Hong Kong",            "HN": "Honduras",    "HR": "Croatia",
    "HU": "Hungary",              "ID": "Indonesia",   "IE": "Ireland",
    "IL": "Israel",               "IN": "India",       "IQ": "Iraq",
    "IR": "Iran",                 "IS": "Iceland",     "IT": "Italy",
    "JM": "Jamaica",              "JO": "Jordan",      "JP": "Japan",
    "KE": "Kenya",                "KH": "Cambodia",    "KR": "South Korea",
    "KW": "Kuwait",               "KZ": "Kazakhstan",  "LA": "Laos",
    "LB": "Lebanon",              "LI": "Liechtenstein", "LK": "Sri Lanka",
    "LT": "Lithuania",            "LU": "Luxembourg",  "LV": "Latvia",
    "MA": "Morocco",              "MD": "Moldova",     "MK": "North Macedonia",
    "MM": "Myanmar",              "MO": "Macau",       "MT": "Malta",
    "MX": "Mexico",               "MY": "Malaysia",    "NG": "Nigeria",
    "NL": "Netherlands",          "NO": "Norway",      "NP": "Nepal",
    "NZ": "New Zealand",          "OM": "Oman",        "PA": "Panama",
    "PE": "Peru",                 "PH": "Philippines", "PK": "Pakistan",
    "PL": "Poland",               "PT": "Portugal",    "PY": "Paraguay",
    "QA": "Qatar",                "RO": "Romania",     "RS": "Serbia",
    "RU": "Russia",               "SA": "Saudi Arabia", "SE": "Sweden",
    "SG": "Singapore",            "SI": "Slovenia",    "SK": "Slovakia",
    "SO": "Somalia",              "SV": "El Salvador", "SY": "Syria",
    "TH": "Thailand",             "TN": "Tunisia",     "TR": "Turkey",
    "TT": "Trinidad and Tobago",  "TW": "Taiwan",      "UA": "Ukraine",
    "UG": "Uganda",               "US": "United States", "UY": "Uruguay",
    "UZ": "Uzbekistan",           "VE": "Venezuela",   "VN": "Vietnam",
    "YE": "Yemen",                "ZA": "South Africa",
}


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

    fig, ax = plt.subplots(figsize=(5, 2.8))
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



def _draw_range_scale(salary_low, salary_avg, salary_high, offer=None, offer_label="Your Offer", offers_list=None, title=None):
    """Horizontal matplotlib salary range scale with avg marker, low/high anchors, optional offer marker."""
    fig, ax = plt.subplots(figsize=(3.2, 1))
    ax.barh([""], [salary_high - salary_low], left=salary_low, color="#3498db", alpha=0.25, height=0.4)
    ax.plot(salary_avg, 0, marker="v", color="#2980b9", markersize=12, zorder=5)
    ax.text(salary_avg, 0.28, f"Avg\n${salary_avg:,}", ha="center", va="bottom", fontsize=4, color="#2980b9", fontweight="bold")
    ax.axvline(salary_low,  color="#e67e22", linewidth=1.5, linestyle=":")
    ax.text(salary_low,  0.28, f"${salary_low:,}",  ha="center", va="bottom", fontsize=3, color="#e67e22", fontweight="bold")
    ax.axvline(salary_high, color="#27ae60", linewidth=1.5, linestyle=":")
    ax.text(salary_high, 0.28, f"${salary_high:,}", ha="center", va="bottom", fontsize=3, color="#27ae60", fontweight="bold")
    if offers_list:
        for i, (lbl, amt) in enumerate(offers_list):
            color = "#27ae60" if amt >= salary_avg else "#e67e22"
            ax.plot(amt, 0, marker="o", color=color, markersize=11, zorder=6+i)
            ax.text(amt, 0.56 + i*0.18, f"{lbl}\n${amt:,}", ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="bold")
    elif offer and offer > 0:
        color = "#27ae60" if offer >= salary_avg else "#e67e22"
        ax.plot(offer, 0, marker="o", color=color, markersize=11, zorder=6)
        ax.text(offer, 0.56, f"{offer_label}\n${offer:,}", ha="center", va="bottom", fontsize=3, color=color, fontweight="bold")
    ax.set_xlim(salary_low * 0.88, salary_high * 1.12)
    ax.set_ylim(-0.6, 1.1)  # fixed height — prevents chart resizing when offer label is added
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Annual Salary (USD)", fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _show_benchmark_yourself(result):
    """
    Visual comparison: user's predicted salary avg vs. peer benchmark bars.
    Shows where the user lands relative to p25, median, p75.
    """
    benchmark = result.get("benchmark", {})

    median = benchmark.get("median", 0)
    p25    = benchmark.get("p25", 0)
    p75    = benchmark.get("p75", 0)
    your_salary = result.get("salary_avg", 0)

    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Background band for p25–p75
    ax.barh(["Peer Range"], [p75 - p25], left=p25,
            color="#3498db", alpha=0.3, height=0.4, label="Peer 25–75th %ile")

    # Median line
    ax.axvline(median, color="#2980b9", linewidth=2, linestyle="--", label=f"Peer Median ${median:,}")

    # Your predicted salary avg
    ax.axvline(your_salary, color="#e74c3c", linewidth=2.5, label=f"Your Salary Avg ${your_salary:,}")

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
    """Display a single prediction result: matched title, salary range scale, offer input, narrative, charts."""
    salary_low  = result.get("salary_low", 0)
    salary_avg  = result.get("salary_avg", 0)
    salary_high = result.get("salary_high", 0)
    narrative   = result.get("narrative", "No narrative available.")
    matched_title = result.get("matched_job_title", "")
    match_score   = result.get("match_score", 0)
    benchmark     = result.get("benchmark", {})

    # Headline: matched job title
    st.markdown(f"## {matched_title}")
    if result.get("title_fallback"):
        original = result.get("original_job_title", "your title")
        st.caption(f"⚠️ No exact match found for **{original}** — showing the closest available prediction based on **{matched_title}**.")

    # Key metrics in columns — color coded
    m1, m2, m3 = st.columns(3)
    m1.markdown(f'<div style="background:#fef5ec;border:1px solid #e67e22;border-radius:8px;padding:14px 16px;text-align:center;"><div style="font-size:12px;color:#e67e22;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Low</div><div style="font-size:26px;font-weight:700;color:#e67e22;">${salary_low:,}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div style="background:#eaf4fd;border:1px solid #2980b9;border-radius:8px;padding:14px 16px;text-align:center;"><div style="font-size:12px;color:#2980b9;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Average</div><div style="font-size:26px;font-weight:700;color:#2980b9;">${salary_avg:,}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div style="background:#eafaf1;border:1px solid #27ae60;border-radius:8px;padding:14px 16px;text-align:center;"><div style="font-size:12px;color:#27ae60;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">High</div><div style="font-size:26px;font-weight:700;color:#27ae60;">${salary_high:,}</div></div>', unsafe_allow_html=True)
    st.write("")  # spacing

    # Range scale with offer input — constrained to half page width
    st.subheader("Salary Range Scale")
    scale_col, _ = st.columns([1, 1])
    with scale_col:
        offer_input = st.number_input(
            "Enter your offer amount (USD) to see where it lands:",
            min_value=0, value=0, step=1000,
            key="offer_input"
        )
        offer_val = int(offer_input) if int(offer_input) > 0 else None
        _draw_range_scale(salary_low, salary_avg, salary_high, offer=offer_val)

    # Offer verdict
    if offer_val and offer_val > 0:
        if offer_val >= salary_high:
            st.markdown(f'<div style="background:#d5f5e3;border-left:4px solid #27ae60;padding:10px 14px;border-radius:4px;color:#1e8449;">🎯 <strong>Excellent offer.</strong> ${offer_val:,} is at or above the high end of the market range.</div>', unsafe_allow_html=True)
        elif offer_val >= salary_avg:
            st.markdown(f'<div style="background:#d6eaf8;border-left:4px solid #2980b9;padding:10px 14px;border-radius:4px;color:#1a5276;">✅ <strong>Strong offer.</strong> ${offer_val:,} is above the average of ${salary_avg:,}.</div>', unsafe_allow_html=True)
        elif offer_val >= salary_low:
            st.markdown(f'<div style="background:#fef9e7;border-left:4px solid #e67e22;padding:10px 14px;border-radius:4px;color:#784212;">⚠️ <strong>Below average.</strong> ${offer_val:,} is below the market average of ${salary_avg:,}. There may be room to negotiate.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background:#fdedec;border-left:4px solid #e74c3c;padding:10px 14px;border-radius:4px;color:#922b21;">🚨 <strong>Low offer.</strong> ${offer_val:,} is below the low end of the market range (${salary_low:,}). Consider negotiating.</div>', unsafe_allow_html=True)

    # Benchmark — room analogy
    if benchmark:
        st.subheader("Peer Benchmark")
        peer_count = benchmark.get("peer_count", 0)
        median     = benchmark.get("median", 0)
        p25        = benchmark.get("p25", 0)
        p75        = benchmark.get("p75", 0)

        # Determine where the predicted avg sits relative to peers
        if salary_avg >= p75:
            position_line = f"Your predicted salary of <strong>${salary_avg:,}</strong> puts you in the <strong style='color:#27ae60;'>top 25%</strong> of your peers."
        elif salary_avg >= median:
            position_line = f"Your predicted salary of <strong>${salary_avg:,}</strong> puts you <strong style='color:#2980b9;'>above the median</strong> — better than half your peers."
        elif salary_avg >= p25:
            position_line = f"Your predicted salary of <strong>${salary_avg:,}</strong> puts you <strong style='color:#e67e22;'>below the median</strong> — room to grow."
        else:
            position_line = f"Your predicted salary of <strong>${salary_avg:,}</strong> puts you in the <strong style='color:#e74c3c;'>bottom 25%</strong> of your peers."

        st.markdown(
            f'<div style="background:#f8f9fa;border-left:4px solid #2980b9;padding:16px 18px;border-radius:6px;line-height:1.9;color:#2c3e50;font-size:15px;">'
            f'If you were in a room with <strong>100 people</strong> who share your job conditions:<br>'
            f'&nbsp;&nbsp;• 25 of them earn less than <strong>${p25:,}</strong><br>'
            f'&nbsp;&nbsp;• Half earn less than <strong>${median:,}</strong> (the market median)<br>'
            f'&nbsp;&nbsp;• Only 25 earn more than <strong>${p75:,}</strong><br><br>'
            f'{position_line}<br>'
            f'<span style="font-size:12px;color:#7f8c8d;">Based on {peer_count} data points.</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Benchmark yourself visual (below peer benchmark card)
    with st.expander("📊 See how you compare visually", expanded=True):
        _show_benchmark_yourself(result)

    # Analyst Report
    st.subheader("Analyst Report")
    clean_narrative = (narrative or "No narrative available.")
    clean_narrative = re.sub(r'\s*\((low|high|avg|average|median)\)', '', clean_narrative, flags=re.IGNORECASE)
    st.markdown(
        f'<div style="background:#f8f9fa;border-left:4px solid #3498db;padding:14px 16px;'
        f'border-radius:4px;font-size:15px;line-height:1.7;color:#2c3e50;">{clean_narrative}</div>',
        unsafe_allow_html=True
    )

    # Peer chart below analyst report
    chart_peer = result.get("chart_peer_url")
    if chart_peer:
        st.write("")
        _, img_col, _ = st.columns([0.5, 3, 0.5])
        img_col.image(chart_peer, caption="Salary by Experience Level", use_container_width=True)

    # Feature importance
    with st.expander("🔍 What factors influenced this prediction?", expanded=False):
        _show_feature_importance(model)

    # Download report button
    range_str = f"${salary_low:,} – ${salary_high:,}"
    report_text = f"""PayLens Salary Report
========================
Job Title: {matched_title}
Salary Range: {range_str}
  Low:  ${salary_low:,}
  Avg:  ${salary_avg:,}
  High: ${salary_high:,}

Peer Benchmark:
  Median: ${benchmark.get('median', 0):,}
  25th percentile: ${benchmark.get('p25', 0):,}
  75th percentile: ${benchmark.get('p75', 0):,}
  Peer count: {benchmark.get('peer_count', 0)}

Analyst Report:
{narrative}
"""
    st.download_button(
        label="Download Report",
        data=report_text,
        file_name=f"paylens_{matched_title.replace(' ', '_').lower()}.txt",
        mime="text/plain"
    )


st.title("💰 PayLens")
st.caption("Intelligent salary prediction for data professionals")
st.divider()

tab_predict, tab_market, tab_history = st.tabs(["🔮 Predict", "📊 Market Insights", "📋 History"])

with tab_predict:
    st.subheader("Your Job Details")
    f_left, f_right = st.columns(2)

    with f_left:
        job_title = st.text_input(
            "What's your job title?",
            placeholder="e.g. Data Scientist, ML Engineer",
            help="We'll match this to the nearest known title automatically"
        )
        experience = st.selectbox(
            "What's your experience level?",
            options=EXP_OPTIONS,
            format_func=lambda x: EXP_LABELS.get(x, x)
        )
        residence = st.selectbox(
            "Where do you live?",
            options=COUNTRY_OPTIONS,
            format_func=lambda x: COUNTRY_NAMES.get(x, x)
        )
        company_name = st.text_input(
            "Company name (optional)",
            placeholder="e.g. Google, Accenture"
        )

    with f_right:
        employment = st.selectbox(
            "What's your employment type?",
            options=EMP_OPTIONS,
            format_func=lambda x: EMP_LABELS.get(x, x)
        )
        remote = st.selectbox(
            "Where do you work from?",
            options=[100, 50, 0],
            format_func=lambda x: {0: "On-site", 50: "Hybrid", 100: "Full Remote"}[x]
        )
        company_loc = st.selectbox(
            "Where is your company based?",
            options=COUNTRY_OPTIONS,
            format_func=lambda x: COUNTRY_NAMES.get(x, x)
        )
        company_size = st.selectbox(
            "How big is your company?",
            options=["S", "M", "L"],
            index=1,
            format_func=lambda x: SIZE_LABELS.get(x, x)
        )

    predict_btn = st.button("🔮 Predict Salary", type="primary", use_container_width=True)
    st.divider()

    col_result = st.container()
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

                with st.spinner("Analyzing your profile and generating your salary report…"):
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

        avg_salary = df["salary_avg"].mean() if "salary_avg" in df.columns else 0
        h2.metric("Avg Predicted Salary", f"${avg_salary:,.0f}" if avg_salary else "—")

        if "salary_low" in df.columns and "salary_high" in df.columns:
            avg_spread = (df["salary_high"] - df["salary_low"]).mean()
            h3.metric("Avg Range Spread", f"${avg_spread:,.0f}" if avg_spread else "—")
        else:
            h3.metric("Avg Range Spread", "—")

        # Display table with selected columns only
        display_cols = [c for c in ["created_at", "matched_job_title", "experience_level",
                        "salary_low", "salary_avg", "salary_high",
                        "benchmark_median"] if c in df.columns]

        # Rename for readability
        col_labels = {
            "created_at": "Date",
            "matched_job_title": "Job Title",
            "experience_level": "Experience",
            "salary_low": "Salary Low",
            "salary_avg": "Salary Avg",
            "salary_high": "Salary High",
            "benchmark_median": "Peer Median"
        }

        # Format the date to be shorter
        df_display = df[display_cols].copy()
        df_display["created_at"] = pd.to_datetime(df_display["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df_display = df_display.rename(columns=col_labels)

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Expandable detail view for each row
        st.subheader("🔍 Row Detail")
        df["_label"] = df.apply(
            lambda r: f"{r.get('matched_job_title','?')} — {EXP_LABELS.get(r.get('experience_level',''),'?')} — {pd.to_datetime(r.get('created_at','')).strftime('%Y-%m-%d %H:%M') if r.get('created_at') else ''}",
            axis=1
        )
        selected_label = st.selectbox("Select a prediction to expand:", df["_label"].tolist(), key="history_select")
        selected_idx = df[df["_label"] == selected_label].index[0]

        if st.button("📂 Expand", key="expand_row"):
            row = df.loc[selected_idx]

            sal_low  = int(row.get("salary_low",  0) or 0)
            sal_avg  = int(row.get("salary_avg",  0) or 0)
            sal_high = int(row.get("salary_high", 0) or 0)

            st.markdown(f"### {row.get('matched_job_title', 'Unknown')} — {row.get('experience_level', '')}")

            d1, d2, d3 = st.columns(3)
            d1.metric("Salary Low",  f"${sal_low:,}")
            d2.metric("Salary Avg",  f"${sal_avg:,}")
            d3.metric("Salary High", f"${sal_high:,}")

            exp_col, rem_col = st.columns(2)
            exp_col.metric("Experience", row.get("experience_level", "—"))
            rem_col.metric("Remote", f"{row.get('remote_ratio', '—')}%")

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Peer Median", f"${int(row.get('benchmark_median', 0) or 0):,}")
            b2.metric("25th %ile", f"${int(row.get('benchmark_p25', 0) or 0):,}")
            b3.metric("75th %ile", f"${int(row.get('benchmark_p75', 0) or 0):,}")
            b4.metric("Peer Count", int(row.get("benchmark_peers", 0) or 0))

            st.subheader("📝 Analyst Report")
            hist_narrative = (row.get("narrative") or "No narrative available.")
            hist_narrative = re.sub(r'\s*\((low|high|avg|average|median)\)', '', hist_narrative, flags=re.IGNORECASE)
            st.markdown(
                f'<div style="background:#f8f9fa;border-left:4px solid #3498db;padding:14px 16px;'
                f'border-radius:4px;font-size:15px;line-height:1.7;color:#2c3e50;">{hist_narrative}</div>',
                unsafe_allow_html=True
            )

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
    df_market = load_history()

    if df_market.empty:
        st.info("No data yet. Run some predictions first to see market insights!")
    else:
        total_people = len(df_market)
        _, mid_col, _ = st.columns([1, 2, 1])
        with mid_col:
            st.markdown(f"## Out of **{total_people}** people in your field:")
            st.write("")

            # Graph 1 — Salary Distribution
            st.markdown("**Salary Distribution**")
            if "salary_avg" in df_market.columns:
                fig1, ax1 = plt.subplots(figsize=(5, 3))
                ax1.hist(df_market["salary_avg"].dropna(), bins=15, color="#3498db", edgecolor="white")
                ax1.set_xlabel("Predicted Salary Avg (USD)")
                ax1.set_ylabel("Count")
                ax1.set_title("Salary Avg Distribution")
                ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
                ax1.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.caption("salary_avg column not available.")

            st.write("")

            # Graph 2 — Predictions by Experience Level
            st.markdown("**Predictions by Experience Level**")
            exp_counts = df_market["experience_level"].value_counts()
            exp_labels_map = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
            exp_counts.index = [exp_labels_map.get(x, x) for x in exp_counts.index]
            fig3, ax3 = plt.subplots(figsize=(5, 3))
            ax3.barh(exp_counts.index, exp_counts.values, color="#3498db")
            ax3.set_xlabel("Count")
            ax3.set_title("Predictions by Experience")
            ax3.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig3)
            plt.close(fig3)

            st.write("")

            # Graph 3 — Remote Work Distribution (half size)
            st.markdown("**Remote Work Distribution**")
            remote_counts = df_market["remote_ratio"].value_counts().sort_index()
            remote_labels = {0: "On-site", 50: "Hybrid", 100: "Full Remote"}
            remote_counts.index = [remote_labels.get(x, str(x)) for x in remote_counts.index]
            pie_col, _ = st.columns([1, 1])
            with pie_col:
                fig4, ax4 = plt.subplots(figsize=(2.5, 2.5))
                ax4.pie(remote_counts.values, labels=remote_counts.index,
                        autopct="%1.0f%%", colors=["#e74c3c", "#f39c12", "#2ecc71"],
                        startangle=90)
                ax4.set_title("Remote Work Split", fontsize=9)
                st.pyplot(fig4)
                plt.close(fig4)

# ─────────────────────────────────────────────────────────────────────────────
# Premium Pitch Section
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("## ✨ PayLens Premium — *Coming Soon*")

st.markdown("""
**🤖 Real-Time LLM Coaching**

Get a personalised salary negotiation script generated by GPT-4o, tailored to your exact title, location, and experience level.
""")

st.markdown("""
**📈 Salary Trajectory Planner**

See a 3-year salary growth projection based on market trends and your current experience level.
""")

st.button("💬 Join the Waitlist", type="secondary", disabled=True)
st.caption("Premium features are currently in development. Join the waitlist to be notified at launch.")
