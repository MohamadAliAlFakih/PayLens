# model/train.py
# PayLens - Model Training Script
# Run with: python model/train.py

# =============================================================================
# PART 1 - IMPORTS AND SETUP
# =============================================================================

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Add the project root (paylens/) to Python's search path so we can import config.py
# __file__ is this file (model/train.py)
# dirname(__file__) is model/
# dirname(dirname(__file__)) is paylens/ - that's the root we want
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import config

# Create logs/ folder at project root if it doesn't exist
LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Logging writes to both terminal and logs/pipeline.log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE)
    ]
)
log = logging.getLogger(__name__)


# =============================================================================
# PART 2 - LOAD AND VALIDATE DATA
# =============================================================================

def load_and_validate(path):
    log.info("")
    log.info("=== PART 2: LOAD AND VALIDATE ===")

    if not os.path.exists(path):
        log.error("Dataset not found at: " + path)
        log.error("Download from: https://kaggle.com/datasets/ruchi798/data-science-job-salaries")
        sys.exit(1)

    df = pd.read_csv(path)
    log.info("Loaded " + str(len(df)) + " rows and " + str(len(df.columns)) + " columns")

    expected_columns = [
        "work_year", "experience_level", "employment_type",
        "job_title", "salary", "salary_currency", "salary_in_usd",
        "employee_residence", "remote_ratio", "company_location", "company_size"
    ]

    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        log.error("Missing columns: " + str(missing))
        sys.exit(1)

    log.info("All expected columns present")
    log.info("=== PART 2: COMPLETE ===")
    return df


# =============================================================================
# PART 3 - EXPLORATORY DATA ANALYSIS
# Understand the data before any cleaning decisions are made.
# All plots saved to paylens/logs/
# =============================================================================

def run_eda(df):
    log.info("")
    log.info("=== PART 3: EXPLORATORY DATA ANALYSIS ===")

    # Basic info
    log.info("Shape: " + str(df.shape))
    log.info("Column types:\n" + df.dtypes.to_string())

    # Missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        log.info("No missing values found")
    else:
        log.info("Missing values:\n" + missing[missing > 0].to_string())

    # Duplicates
    log.info("Duplicate rows: " + str(df.duplicated().sum()))

    # Salary stats
    log.info("Salary in USD stats:\n" + df["salary_in_usd"].describe().to_string())

    # Categorical distributions
    for col in ["experience_level", "employment_type", "company_size", "remote_ratio"]:
        log.info(col + " value counts:\n" + df[col].value_counts().to_string())

    log.info("Unique job titles: " + str(df["job_title"].nunique()))
    log.info("Top 10 job titles:\n" + df["job_title"].value_counts().head(10).to_string())

    # Build a 2x3 grid of EDA charts
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PayLens - EDA Overview", fontsize=16, fontweight="bold")

    # Chart 1: Salary distribution with density curve
    sns.histplot(df["salary_in_usd"], bins=40, kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Salary Distribution (USD)")
    axes[0, 0].set_xlabel("Salary (USD)")
    axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Chart 2: Salary spread per experience level - most important relationship
    sns.boxplot(
        data=df, x="experience_level", y="salary_in_usd",
        order=["EN", "MI", "SE", "EX"], ax=axes[0, 1], palette="Blues"
    )
    axes[0, 1].set_title("Salary by Experience Level")
    axes[0, 1].set_xlabel("EN=Entry  MI=Mid  SE=Senior  EX=Exec")
    axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Chart 3: Salary spread per company size
    sns.boxplot(
        data=df, x="company_size", y="salary_in_usd",
        order=["S", "M", "L"], ax=axes[0, 2], palette="Greens"
    )
    axes[0, 2].set_title("Salary by Company Size")
    axes[0, 2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))

    # Chart 4: Employment type breakdown as a pie
    employment_counts = df["employment_type"].value_counts()
    axes[1, 0].pie(
        employment_counts.values,
        labels=employment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=sns.color_palette("muted")
    )
    axes[1, 0].set_title("Employment Type Distribution")

    # Chart 5: How many jobs are remote vs hybrid vs onsite
    remote_counts = df["remote_ratio"].value_counts().sort_index()
    sns.barplot(
        x=remote_counts.index.astype(str),
        y=remote_counts.values,
        ax=axes[1, 1],
        palette="Oranges_r"
    )
    axes[1, 1].set_title("Remote Ratio Distribution")
    axes[1, 1].set_xlabel("0=Onsite  50=Hybrid  100=Full Remote")

    # Chart 6: Most common job titles
    top_titles = df["job_title"].value_counts().head(10)
    sns.barplot(x=top_titles.values, y=top_titles.index, ax=axes[1, 2], palette="Purples_r")
    axes[1, 2].set_title("Top 10 Job Titles")
    axes[1, 2].set_xlabel("Count")

    plt.tight_layout()
    eda_path = os.path.join(LOGS_DIR, "eda_overview.png")
    plt.savefig(eda_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("EDA chart saved to: " + eda_path)
    log.info("=== PART 3: COMPLETE ===")


# =============================================================================
# PART 4 - DATA CLEANING
# Decisions here are informed by the EDA above.
# salary_in_usd is kept here — it becomes the regression target in split_data().
# =============================================================================

def clean_data(df):
    log.info("")
    log.info("=== PART 4: DATA CLEANING ===")

    # Always work on a copy - never modify the original dataframe
    df = df.copy()

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    log.info("Removed " + str(before - len(df)) + " duplicate rows (" + str(len(df)) + " remaining)")

    # Drop columns we don't need:
    # salary + salary_currency are replaced by salary_in_usd
    # work_year has too few unique values to be meaningful
    # "Unnamed: 0" is the CSV row index accidentally included in the original training —
    # it had ~19% spurious feature importance; dropping it gives honest predictions
    cols_to_drop = ["salary", "salary_currency", "work_year"]
    # Also drop "Unnamed: 0" if the CSV was saved with its index column
    if "Unnamed: 0" in df.columns:
        cols_to_drop.append("Unnamed: 0")
        log.info("Dropping 'Unnamed: 0' (leaked CSV row index)")
    df = df.drop(columns=cols_to_drop)
    log.info("Dropped columns: " + str(cols_to_drop))

    # Group rare job titles into 'Other' to prevent overfitting on uncommon roles
    MIN_TITLE_COUNT = 10
    title_counts = df["job_title"].value_counts()
    rare_titles = title_counts[title_counts < MIN_TITLE_COUNT].index.tolist()
    df["job_title"] = df["job_title"].apply(lambda t: "Other" if t in rare_titles else t)
    log.info("Grouped " + str(len(rare_titles)) + " rare titles into 'Other'")
    log.info("Unique titles remaining: " + str(df["job_title"].nunique()))

    # Remove salary outliers above the 99th percentile
    # These are likely data entry errors or one-off contracts
    p99 = df["salary_in_usd"].quantile(0.99)
    before = len(df)
    df = df[df["salary_in_usd"] <= p99]
    log.info("Removed " + str(before - len(df)) + " outliers above $" + f"{p99:,.0f}")

    log.info("Clean dataset: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns")
    log.info("=== PART 4: COMPLETE ===")
    return df


# =============================================================================
# PART 6 - ENCODE FEATURES AND SPLIT DATA
# RandomForest needs numbers not text.
# Encoders are saved to disk so the API uses the exact same mapping.
# Note: salary_tier removed — we now predict salary_in_usd directly.
# =============================================================================

def encode_features(df):
    log.info("")
    log.info("=== PART 6: ENCODING FEATURES ===")

    # salary_tier is gone — we encode only the true input features
    categorical_cols = [
        "experience_level", "employment_type", "job_title",
        "employee_residence", "company_location", "company_size"
    ]

    encoders = {}
    df = df.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        log.info("  " + col + ": " + str(mapping))

    os.makedirs(os.path.dirname(config.ENCODERS_PATH), exist_ok=True)
    joblib.dump(encoders, config.ENCODERS_PATH)
    log.info("Encoders saved to: " + config.ENCODERS_PATH)
    log.info("=== PART 6: COMPLETE ===")
    return df, encoders


def split_data(df):
    log.info("")
    log.info("=== PART 6b: TRAIN/TEST SPLIT ===")

    # Target is salary_in_usd (continuous) — regression, not classification
    X = df.drop(columns=["salary_in_usd"])
    y = df["salary_in_usd"]

    log.info("Features: " + str(list(X.columns)))

    # 80% training, 20% testing
    # stratify is not used for regression targets (only valid for classification)
    # random_state=42 makes the split reproducible every run
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log.info("Training rows: " + str(len(X_train)))
    log.info("Test rows    : " + str(len(X_test)))
    log.info("=== PART 6b: COMPLETE ===")
    return X_train, X_test, y_train, y_test


# =============================================================================
# PART 7 - TRAIN, EVALUATE, SAVE MODEL
# Trains a RandomForestRegressor and saves the model + evaluation artifacts.
# A Random Forest trains 100 independent decision trees on random data subsets.
# The spread of those 100 predictions gives us a natural salary confidence range.
# =============================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, encoders):
    log.info("")
    log.info("=== PART 7: MODEL TRAINING ===")

    model = RandomForestRegressor(
        n_estimators=100,       # 100 trees — more trees = more stable range estimates
        max_depth=12,           # Slightly deeper than DT since the forest prevents overfitting
        min_samples_split=10,   # A node needs at least 10 samples before it can split
        min_samples_leaf=5,     # Every leaf node must have at least 5 samples
        random_state=42,        # Makes results reproducible
        n_jobs=-1               # Use all CPU cores for faster training
    )

    model.fit(X_train, y_train)
    log.info("Model trained")

    # Evaluate on the test set (data the model has never seen)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2   = r2_score(y_test, y_pred)

    log.info(f"MAE:  ${mae:,.0f}")
    log.info(f"RMSE: ${rmse:,.0f}")
    log.info(f"R²:   {r2:.4f}")

    # Chart: Feature importances — which features matter most to the forest
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("RandomForest - Feature Importances", fontsize=14, fontweight="bold")

    feature_names = X_train.columns.tolist()
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]

    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax, palette="Blues_r")
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance Score")

    plt.tight_layout()
    eval_path = os.path.join(LOGS_DIR, "model_evaluation.png")
    plt.savefig(eval_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Evaluation chart saved to: " + eval_path)

    # Save a human-readable training report to logs/
    report_path = os.path.join(LOGS_DIR, "training_report.txt")
    with open(report_path, "w") as f:
        f.write("=== PAYLENS TRAINING REPORT ===\n\n")

        f.write("--- MODEL ---\n")
        f.write("Type: RandomForestRegressor\n")
        f.write("Trees: 100  |  max_depth: 12\n\n")

        f.write("--- REGRESSION METRICS ---\n")
        f.write(f"MAE:  ${mae:,.0f}   (mean absolute error — average dollar miss)\n")
        f.write(f"RMSE: ${rmse:,.0f}   (root mean squared error — penalises large misses)\n")
        f.write(f"R²:   {r2:.4f}   (1.0 = perfect; above 0.60 is good for salary data)\n\n")

        f.write("--- FEATURE IMPORTANCES ---\n")
        for feat, imp in zip(sorted_features, sorted_importances):
            f.write("  " + f"{feat:<25}" + f"{imp:.4f}" + "\n")
        f.write("\n")

    log.info("Training report saved to: " + report_path)

    # Save the trained model
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    log.info("Model saved to: " + config.MODEL_PATH)

    # Delete stale thresholds.pkl — it belonged to the old classifier and no longer applies
    thresholds_path = os.path.join(ROOT_DIR, "model", "thresholds.pkl")
    if os.path.exists(thresholds_path):
        os.remove(thresholds_path)
        log.info("Removed stale thresholds.pkl")

    # -------------------------------------------------------------------------
    # Smoke test: verify the model returns a sensible three-value range
    # We use the individual tree predictions to derive p25 / p50 / p75.
    # Each tree independently predicts salary; the spread = confidence range.
    # -------------------------------------------------------------------------
    sample = X_test.iloc[[0]]
    tree_preds = np.array([tree.predict(sample.values)[0] for tree in model.estimators_])
    p25 = int(np.percentile(tree_preds, 25))
    p50 = int(np.percentile(tree_preds, 50))
    p75 = int(np.percentile(tree_preds, 75))
    log.info(f"Smoke test — sample prediction range: ${p25:,} | ${p50:,} | ${p75:,}")
    assert p25 <= p50 <= p75, "Smoke test FAILED: percentiles out of order"
    assert p25 > 10000, "Smoke test FAILED: p25 suspiciously low"
    log.info("Smoke test PASSED")

    log.info("=== PART 7: COMPLETE ===")
    return model


# =============================================================================
# PART 8 - CONTEXTUAL BENCHMARK TABLE
# Precomputes salary statistics by subgroup (experience, title, company size).
# Used by the LLM and dashboard to give contextual comparisons like:
# "Among Senior Data Scientists, your salary is in the top 25%"
# =============================================================================

def build_benchmark_table():
    log.info("")
    log.info("=== PART 8: BUILDING BENCHMARK TABLE ===")

    # Reload the raw CSV - we need salary_in_usd from the original data
    raw = pd.read_csv(config.DATA_PATH)
    raw = raw.drop_duplicates()
    raw = raw.drop(columns=["salary", "salary_currency", "work_year"])
    # Drop the leaked index column if present
    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])

    # Apply the same outlier filter as in clean_data
    p99 = raw["salary_in_usd"].quantile(0.99)
    raw = raw[raw["salary_in_usd"] <= p99]

    # Apply the same rare title grouping as in clean_data
    MIN_TITLE_COUNT = 10
    title_counts = raw["job_title"].value_counts()
    rare_titles = title_counts[title_counts < MIN_TITLE_COUNT].index.tolist()
    raw["job_title"] = raw["job_title"].apply(lambda t: "Other" if t in rare_titles else t)

    benchmarks = {}

    # Level 1: By experience only - always has enough data (100s of rows per level)
    level1 = raw.groupby("experience_level")["salary_in_usd"].agg(
        count="count",
        median="median",
        mean="mean",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    ).round(0)
    benchmarks["by_experience"] = level1
    log.info("Experience benchmarks:\n" + level1.to_string())

    # Level 2: By experience + company size
    level2 = raw.groupby(["experience_level", "company_size"])["salary_in_usd"].agg(
        count="count",
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75)
    ).round(0)
    level2 = level2[level2["count"] >= 5]
    benchmarks["by_experience_and_size"] = level2
    log.info("Experience+size subgroups: " + str(len(level2)))

    # Level 3: By job title only
    level3 = raw.groupby("job_title")["salary_in_usd"].agg(
        count="count",
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75),
    ).round(0)
    level3 = level3[level3["count"] >= 5]
    benchmarks["by_title"] = level3
    log.info("Title benchmarks: " + str(len(level3)) + " unique titles")

    # Level 4: By experience + job title (most specific, most likely to be sparse)
    level4 = raw.groupby(["experience_level", "job_title"])["salary_in_usd"].agg(
        count="count",
        median="median",
        p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75)
    ).round(0)
    level4 = level4[level4["count"] >= 10]
    benchmarks["by_experience_and_title"] = level4
    log.info("Experience+title subgroups: " + str(len(level4)))

    # Chart: Benchmark medians at a glance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Contextual Salary Benchmarks", fontsize=14, fontweight="bold")

    # Left: median salary per experience level
    exp_medians = level1["median"].reset_index()
    exp_order = ["EN", "MI", "SE", "EX"]
    exp_medians["experience_level"] = pd.Categorical(
        exp_medians["experience_level"], categories=exp_order, ordered=True
    )
    exp_medians = exp_medians.sort_values("experience_level")

    bars = sns.barplot(
        data=exp_medians, x="experience_level", y="median",
        ax=axes[0], palette="Blues"
    )
    axes[0].set_title("Median Salary by Experience Level")
    axes[0].set_xlabel("Experience Level")
    axes[0].set_ylabel("Median Salary (USD)")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    for bar, val in zip(bars.patches, exp_medians["median"]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1000,
            f"${val:,.0f}",
            ha="center", fontsize=9, fontweight="bold"
        )

    # Right: top 10 titles by median salary
    top_titles = level3["median"].sort_values(ascending=False).head(10).reset_index()
    sns.barplot(data=top_titles, x="median", y="job_title", ax=axes[1], palette="Purples_r")
    axes[1].set_title("Top 10 Titles by Median Salary")
    axes[1].set_xlabel("Median Salary (USD)")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    plt.tight_layout()
    bench_path = os.path.join(LOGS_DIR, "benchmarks.png")
    plt.savefig(bench_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Benchmark chart saved to: " + bench_path)

    # Save benchmark table to model/ folder
    bench_pkl_path = os.path.join(ROOT_DIR, "model", "benchmarks.pkl")
    joblib.dump(benchmarks, bench_pkl_path)
    log.info("Benchmark table saved to: " + bench_pkl_path)

    log.info("=== PART 8: COMPLETE ===")


# =============================================================================
# EXECUTION - Runs all parts in order when you call: python model/train.py
# =============================================================================

if __name__ == "__main__":
    log.info("")
    log.info("=== PAYLENS TRAINING PIPELINE STARTED ===")
    log.info("")

    df = load_and_validate(config.DATA_PATH)
    run_eda(df)
    df = clean_data(df)
    df, encoders = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_and_evaluate(X_train, X_test, y_train, y_test, encoders)
    build_benchmark_table()

    log.info("")
    log.info("=== PAYLENS TRAINING PIPELINE COMPLETE ===")
    log.info("")
    log.info("Outputs:")
    log.info("  Model     : " + config.MODEL_PATH)
    log.info("  Encoders  : " + config.ENCODERS_PATH)
    log.info("  Benchmarks: " + os.path.join(ROOT_DIR, "model", "benchmarks.pkl"))
    log.info("  Logs      : " + LOGS_DIR)
    log.info("")
    log.info("Next step: run 'uvicorn api.main:app --reload --port 8000' to start the API")
