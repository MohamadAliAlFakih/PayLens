# pipeline/predict.py
# PayLens prediction pipeline module.
# Call run_prediction(job_input) from the Streamlit dashboard to get a full result.
# This module: calls the API -> generates charts -> uploads to Supabase -> returns result dict.

import os
import sys
import joblib
import requests
from supabase import create_client

# Add the project root (paylens/) to Python's path so config.py can be imported.
# __file__ is pipeline/predict.py
# dirname(__file__) is pipeline/
# dirname(dirname(...)) is paylens/ — the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config
from pipeline.visualize import generate_overview_chart, generate_peer_chart, upload_chart
from pipeline.narrative import generate_narrative

# Load benchmarks once at module level (same pkl used for chart generation).
# Loading at import time avoids repeated disk I/O on every prediction call.
BENCHMARKS_PATH = os.path.join(ROOT_DIR, "model", "benchmarks.pkl")
benchmarks = joblib.load(BENCHMARKS_PATH)

# Supabase client — initialized once at module level
# Used by save_prediction() to persist results to the cloud database
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)


def run_prediction(job_input: dict) -> dict:
    """
    Takes job input dict, calls FastAPI /predict, generates two charts,
    uploads to Supabase Storage.

    Returns a full result dict or None if the API call fails.

    Parameters
    ----------
    job_input : dict
        Keys: experience_level, employment_type, job_title, employee_residence,
              remote_ratio, company_location, company_size

    Returns
    -------
    dict or None
        All API response fields plus 'chart_overview_url' and 'chart_peer_url'.
        Returns None if the API is unreachable or returns an HTTP error.
    """

    # ------------------------------------------------------------------
    # Step 1 — Call the FastAPI /predict endpoint
    # timeout=10 prevents the function from hanging if the server is slow
    # to respond or the connection stalls mid-request.
    # ------------------------------------------------------------------
    try:
        response = requests.post(config.API_URL, json=job_input, timeout=10)
        response.raise_for_status()  # raises HTTPError for 4xx/5xx responses
        api_result = response.json()
    except requests.exceptions.ConnectionError:
        print("Error: FastAPI server is not running. Start it with: uvicorn api.main:app --port 8000")
        return None
    except requests.exceptions.HTTPError as e:
        print("Error from API:", e)
        return None

    # ------------------------------------------------------------------
    # Step 2 — Extract values needed for chart generation
    # ------------------------------------------------------------------
    predicted_tier = api_result["prediction"]       # "Low", "Mid", or "High"
    experience_level = job_input["experience_level"]  # "EN", "MI", "SE", or "EX"

    # ------------------------------------------------------------------
    # Step 3 — Generate both charts as in-memory PNG bytes
    # Wrapped in try/except so a matplotlib failure doesn't crash the
    # whole prediction flow — the dashboard can still show text results.
    # ------------------------------------------------------------------
    try:
        overview_bytes = generate_overview_chart(benchmarks)
        peer_bytes = generate_peer_chart(benchmarks, experience_level, predicted_tier)
    except Exception as e:
        print("Warning: chart generation failed:", e)
        overview_bytes = None
        peer_bytes = None

    # ------------------------------------------------------------------
    # Step 4 — Upload charts to Supabase Storage
    # upload_chart returns None if the upload fails — dashboard handles
    # None gracefully (Phase 6 creates the bucket; None is expected until then).
    # ------------------------------------------------------------------
    overview_url = None
    peer_url = None

    if overview_bytes:
        overview_url = upload_chart(overview_bytes, "overview")

    if peer_bytes:
        peer_url = upload_chart(peer_bytes, "peer")

    # ------------------------------------------------------------------
    # Step 4.5 — Generate LLM narrative via Ollama/Mistral
    # generate_narrative always returns a string (fallback if Ollama is down)
    # so 'narrative' is always present in the result dict.
    # ------------------------------------------------------------------
    narrative = generate_narrative(api_result)

    # ------------------------------------------------------------------
    # Step 5 — Build and return the full result dict
    # ------------------------------------------------------------------
    result = {
        # Fields from API response
        "prediction":         api_result["prediction"],
        "confidence_pct":     api_result["confidence_pct"],
        "salary_range":       api_result["salary_range"],
        "matched_job_title":  api_result["matched_job_title"],
        "match_score":        api_result["match_score"],
        "benchmark":          api_result["benchmark"],
        "inputs_received":    api_result["inputs_received"],
        "narrative":          narrative,           # LLM text or fallback string
        # Chart URLs from Supabase Storage (None if upload failed or bucket not yet created)
        "chart_overview_url": overview_url,
        "chart_peer_url":     peer_url,
    }

    # Save result to Supabase (failure is non-fatal — dashboard still works from prior rows)
    save_prediction(result)

    return result


def save_prediction(result: dict) -> bool:
    """
    Inserts a prediction result into the Supabase predictions table.

    Maps the result dict keys to the table column names.
    Returns True if the insert succeeded, False if it failed.
    Never raises an exception — a save failure should not crash the pipeline.
    """
    # Flatten the nested dicts into individual column values
    row = {
        "experience_level":    result["inputs_received"]["experience_level"],
        "employment_type":     result["inputs_received"]["employment_type"],
        "job_title":           result["inputs_received"]["job_title"],
        "matched_job_title":   result["matched_job_title"],
        "match_score":         result["match_score"],
        "employee_residence":  result["inputs_received"]["employee_residence"],
        "remote_ratio":        result["inputs_received"]["remote_ratio"],
        "company_location":    result["inputs_received"]["company_location"],
        "company_size":        result["inputs_received"]["company_size"],
        "predicted_tier":      result["prediction"],
        "confidence_pct":      result["confidence_pct"],
        "salary_min":          result["salary_range"]["min"],
        "salary_max":          result["salary_range"].get("max"),   # None for High tier
        "benchmark_median":    result["benchmark"]["median"],
        "benchmark_p25":       result["benchmark"]["p25"],
        "benchmark_p75":       result["benchmark"]["p75"],
        "benchmark_peers":     result["benchmark"]["peer_count"],
        "narrative":           result["narrative"],
        "chart_overview_url":  result.get("chart_overview_url"),
        "chart_peer_url":      result.get("chart_peer_url"),
    }
    try:
        supabase.table("predictions").insert(row).execute()
        return True
    except Exception as e:
        print("Warning: failed to save prediction to Supabase:", e)
        return False


# if __name__ == "__main__" means this block only runs when you call the file
# directly (python pipeline/predict.py), not when it's imported by Streamlit or
# another module.
if __name__ == "__main__":
    # Sample input for manual testing.
    # Run: python pipeline/predict.py  (with FastAPI running on port 8000)
    sample = {
        "experience_level":   "SE",
        "employment_type":    "FT",
        "job_title":          "Data Scientist",
        "employee_residence": "US",
        "remote_ratio":       100,
        "company_location":   "US",
        "company_size":       "M"
    }
    result = run_prediction(sample)
    if result:
        print("Prediction:", result["prediction"])
        print("Confidence:", result["confidence_pct"], "%")
        print("Narrative:\n", result["narrative"])
        print("Overview chart URL:", result["chart_overview_url"])
        print("Peer chart URL:", result["chart_peer_url"])
