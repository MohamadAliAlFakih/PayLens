# pipeline/visualize.py
# Chart generation for PayLens predictions.
# Two functions produce PNG charts as bytes (no disk write).
# One function uploads bytes to Supabase Storage and returns a public URL.

import io
import os
import sys
import uuid

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from supabase import create_client

# Add the project root (paylens/) to the path so we can import config.py
# __file__ is pipeline/visualize.py
# dirname(__file__) is pipeline/
# dirname(dirname(...)) is paylens/ — the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import config

# Supabase client — initialized once at module level so every call reuses it.
# Guard against placeholder / missing credentials (e.g. during local testing
# before a real Supabase project is set up) — a bad URL would raise at import
# time and prevent predict.py from running at all.
try:
    supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
except Exception as _supabase_init_err:
    print(f"Warning: Supabase client could not be initialised ({_supabase_init_err}). "
          "Chart uploads will be skipped.")
    supabase = None

# The Supabase Storage bucket where all chart PNGs are stored
CHARTS_BUCKET = "charts"

# Fixed x-axis order: Entry → Mid → Senior → Executive
EXP_ORDER = ["EN", "MI", "SE", "EX"]

# Human-readable label for each experience code (used in axis labels)
EXP_LABELS = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}

# Colors assigned to each salary tier (green=low, orange=mid, red=high)
TIER_COLORS = {"Low": "#2ecc71", "Mid": "#f39c12", "High": "#e74c3c"}


def generate_overview_chart(benchmarks: dict) -> bytes:
    """
    Generate a bar chart of median salary by experience level.

    Reads the 'by_experience' DataFrame from the benchmarks dict,
    which is indexed on experience_level (EN/MI/SE/EX). Calls
    reset_index() so experience_level becomes a regular column
    that seaborn can find by name.

    Parameters
    ----------
    benchmarks : dict
        The benchmarks dict loaded from model/benchmarks.pkl.
        Must contain key 'by_experience' — a DataFrame indexed
        by experience_level with a 'median' column.

    Returns
    -------
    bytes
        The chart rendered as PNG bytes. No file is written to disk.
    """
    # Step 1: Get the experience-level DataFrame.
    # IMPORTANT: reset_index() converts the index (experience_level)
    # into a regular column. Without this, seaborn raises KeyError.
    df = benchmarks["by_experience"].reset_index()

    # Step 2: Force the bars to appear in EN < MI < SE < EX order
    # by converting the column to an ordered Categorical type.
    df["experience_level"] = pd.Categorical(
        df["experience_level"],
        categories=EXP_ORDER,
        ordered=True
    )
    df = df.sort_values("experience_level")

    # Step 3: Create figure and axes explicitly.
    # Using fig, ax = plt.subplots() keeps matplotlib in OO mode
    # and avoids the global state machine (safer for multiple charts).
    fig, ax = plt.subplots(figsize=(8, 5))

    # Step 4: Draw the bar chart with experience level on x, median salary on y.
    # hue="experience_level" + legend=False is the seaborn 0.14-compatible way
    # to apply a palette per bar without the FutureWarning about palette+no hue.
    sns.barplot(
        data=df,
        x="experience_level",
        y="median",
        hue="experience_level",
        palette="Blues",
        legend=False,
        ax=ax,
        order=EXP_ORDER
    )

    # Step 5: Labels and title
    ax.set_title("Salary by Experience Level", fontsize=13, fontweight="bold")
    ax.set_xlabel("Experience Level  (EN=Entry  MI=Mid  SE=Senior  EX=Exec)")
    ax.set_ylabel("Median Salary (USD)")

    # Step 6: Format y-axis as dollar amounts (e.g. $110,000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Step 7: Add value labels on top of each bar so exact amounts are readable
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1000,
                f"${height:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    plt.tight_layout()

    # Step 8: Render to bytes — no file written to disk.
    # buf.seek(0) positions the cursor at the start so buf.read() returns all bytes.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    chart_bytes = buf.read()

    # plt.close(fig) frees memory — without this, matplotlib accumulates figures
    # and will eventually exhaust memory in a long-running Streamlit session.
    plt.close(fig)

    return chart_bytes


def generate_peer_chart(benchmarks: dict, experience_level: str, salary_avg: int) -> bytes:
    """
    Generate a bar chart showing all experience levels with the
    specified experience_level highlighted in accent blue, plus a
    dashed red line at the predicted salary average.

    The highlighted bar draws attention to where the user sits relative
    to the full salary landscape; the dashed line shows their predicted
    salary against peer medians.

    Parameters
    ----------
    benchmarks : dict
        The benchmarks dict loaded from model/benchmarks.pkl.
        Must contain key 'by_experience' — a DataFrame indexed
        by experience_level with a 'median' column.
    experience_level : str
        The user's experience level code: 'EN', 'MI', 'SE', or 'EX'.
    salary_avg : int
        The predicted average salary. Used to draw a dashed reference
        line and in the chart title.

    Returns
    -------
    bytes
        The chart rendered as PNG bytes. No file is written to disk.
    """
    # Step 1: Get experience-level DataFrame.
    # reset_index() is required — experience_level is the DataFrame index,
    # not a column. Without this, df["experience_level"] raises KeyError.
    df = benchmarks["by_experience"].reset_index()

    # Step 2: Force the order EN < MI < SE < EX
    df["experience_level"] = pd.Categorical(
        df["experience_level"],
        categories=EXP_ORDER,
        ordered=True
    )
    df = df.sort_values("experience_level")

    # Step 3: Assign per-bar colors.
    # The user's bar is highlighted in accent blue (#3498db);
    # all other bars are neutral grey (#bdc3c7).
    colors = [
        "#3498db" if exp == experience_level else "#bdc3c7"
        for exp in EXP_ORDER
    ]

    # Step 4: Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 5))

    # Step 5: Use ax.bar() directly — seaborn's barplot does not support
    # per-bar color lists, so we drop down to matplotlib for this chart.
    x_positions = range(len(EXP_ORDER))
    bars = ax.bar(x_positions, df["median"].values, color=colors)

    # Step 6: Set x-axis tick labels to the experience codes
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(EXP_ORDER)

    # Step 7: Title names the user's level and predicted salary
    ax.set_title(
        f"Your Level: {experience_level} — Predicted ${salary_avg:,}",
        fontsize=13,
        fontweight="bold"
    )
    ax.set_xlabel("Experience Level  (EN=Entry  MI=Mid  SE=Senior  EX=Exec)")
    ax.set_ylabel("Median Salary (USD)")

    # Step 8: Format y-axis as dollar amounts
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Step 9: Add value labels on top of each bar
    for bar, val in zip(bars, df["median"].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"${val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    # Step 10: Add a horizontal dashed line at salary_avg to show the predicted salary
    ax.axhline(y=salary_avg, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Your prediction: ${salary_avg:,}")

    # Step 11: Add a legend showing the highlighted bar and the predicted salary line
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor="#3498db", label=f"Your level ({experience_level})"),
        Patch(facecolor="#bdc3c7", label="Other levels"),
        Line2D([0], [0], color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Your prediction: ${salary_avg:,}"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()

    # Step 11: Render to bytes — no file written to disk
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    chart_bytes = buf.read()

    # plt.close(fig) frees memory — without this, matplotlib accumulates figures
    plt.close(fig)

    return chart_bytes


def upload_chart(chart_bytes: bytes, chart_type: str) -> str:
    """
    Upload PNG bytes to the Supabase Storage 'charts' bucket and return
    the public URL of the uploaded file.

    The bucket must already exist and be set to Public in the Supabase
    dashboard. This function does NOT create the bucket.

    Parameters
    ----------
    chart_bytes : bytes
        The PNG image data as raw bytes (from generate_overview_chart
        or generate_peer_chart).
    chart_type : str
        A short label for the chart type, e.g. 'overview' or 'peer'.
        Used as part of the filename so files are identifiable in the bucket.

    Returns
    -------
    str or None
        The public URL of the uploaded file, or None if the upload failed.
        Failure is logged but does not raise an exception — the pipeline
        continues even if chart upload is unavailable.
    """
    # If Supabase client was not initialised (placeholder credentials), skip silently.
    if supabase is None:
        print(f"Warning: Supabase unavailable — skipping upload for '{chart_type}' chart.")
        return None

    # Generate a unique filename so repeated uploads never overwrite each other.
    # Format: {uuid4}_{chart_type}.png  e.g. "550e8400-..._overview.png"
    filename = f"{uuid.uuid4()}_{chart_type}.png"

    try:
        # Upload the raw bytes to the bucket.
        # CRITICAL: content-type MUST be "image/png".
        # The storage3 default is "text/plain;charset=UTF-8" which causes
        # browsers and st.image() to render the file as garbled text.
        supabase.storage.from_(CHARTS_BUCKET).upload(
            path=filename,
            file=chart_bytes,
            file_options={"content-type": "image/png"}
        )

        # get_public_url() is a pure URL computation — always call it AFTER
        # upload succeeds. It always returns a valid-looking URL regardless of
        # whether the file exists, so calling it before upload gives a 404 URL.
        url = supabase.storage.from_(CHARTS_BUCKET).get_public_url(filename)
        return url

    except Exception as e:
        # Catch any storage error (bucket not found, network failure, auth error).
        # Print the error for debugging but do not crash the pipeline.
        print(f"[ERROR] upload_chart failed for '{filename}': {e}")
        return None
