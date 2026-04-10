"""
api/main.py
PayLens FastAPI prediction server.

What this does:
  - Loads the trained RandomForestRegressor model and encoders from disk at startup (once only).
  - Exposes 5 endpoints:
      GET  /                  - basic health check
      GET  /health            - model info and output format
      GET  /supported-inputs  - valid dropdown values for the UI
      GET  /supported-countries - valid employee residence / company location values
      POST /predict           - takes job details, returns salary_low / salary_avg / salary_high

Run from the project root (paylens/) with:
  uvicorn api.main:app --reload --port 8000
"""

# =============================================================================
# PART 1 - IMPORTS AND PATH SETUP
# =============================================================================

import os
import sys
import difflib

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, validator

# Add the project root (paylens/) to Python's path so we can import config.py.
# __file__  = paylens/api/main.py
# dirname   = paylens/api/
# dirname x2 = paylens/   <-- that's the root we need
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config  # noqa: E402  (must come after sys.path.append)


# =============================================================================
# PART 2 - LOAD ARTIFACTS AT MODULE LEVEL
# These lines run once when uvicorn imports this file.
# Putting them here (not inside a function) means every request shares
# the same objects in memory — no disk I/O per request.
# =============================================================================

model = joblib.load(config.MODEL_PATH)
encoders = joblib.load(config.ENCODERS_PATH)

# benchmarks.pkl stores a dict of DataFrames, keyed by grouping level.
# We use "by_experience" which is indexed on raw experience strings: EN / MI / SE / EX
benchmarks = joblib.load(os.path.join(ROOT_DIR, "model", "benchmarks.pkl"))


# =============================================================================
# PART 3 - DERIVE SUPPORTED VALUES FROM ENCODERS
# We read the valid options directly from the fitted LabelEncoders instead of
# hardcoding them.  This way, if the training data changes, the API updates
# automatically on the next server restart.
# =============================================================================

SUPPORTED_EXPERIENCE   = list(encoders["experience_level"].classes_)
SUPPORTED_EMPLOYMENT   = list(encoders["employment_type"].classes_)
SUPPORTED_COMPANY_SIZE = list(encoders["company_size"].classes_)
SUPPORTED_JOB_TITLES   = list(encoders["job_title"].classes_)

# Countries are used both for employee_residence and company_location.
SUPPORTED_COUNTRIES = list(encoders["employee_residence"].classes_)


# =============================================================================
# PART 4 - CREATE THE FASTAPI APP
# =============================================================================

app = FastAPI(
    title="PayLens API",
    version="2.0",
    description="Salary range prediction for data professionals (RF Regressor)."
)


# =============================================================================
# PART 5 - REQUEST SCHEMA (Pydantic model)
# Pydantic validates the incoming JSON body before our code even runs.
# If a field is missing or fails a validator, FastAPI returns HTTP 422
# automatically with a clear error message.
# =============================================================================

class PredictionInput(BaseModel):
    experience_level:   str  # EN / MI / SE / EX
    employment_type:    str  # FT / PT / CT / FL
    job_title:          str  # "Data Scientist", "ML Engineer", etc.
    employee_residence: str  # ISO country code, e.g. "US"
    remote_ratio:       int  # 0 = onsite, 50 = hybrid, 100 = full remote
    company_location:   str  # ISO country code, e.g. "US"
    company_size:       str  # S / M / L

    @validator("experience_level")
    def validate_experience(cls, v):
        if v not in SUPPORTED_EXPERIENCE:
            raise ValueError(
                f"experience_level '{v}' is not valid. "
                f"Choose from: {SUPPORTED_EXPERIENCE}"
            )
        return v

    @validator("employment_type")
    def validate_employment(cls, v):
        if v not in SUPPORTED_EMPLOYMENT:
            raise ValueError(
                f"employment_type '{v}' is not valid. "
                f"Choose from: {SUPPORTED_EMPLOYMENT}"
            )
        return v

    @validator("employee_residence")
    def validate_residence(cls, v):
        if v not in SUPPORTED_COUNTRIES:
            raise ValueError(
                f"employee_residence '{v}' is not a supported country. "
                f"Use GET /supported-countries for the full list."
            )
        return v

    @validator("company_location")
    def validate_company_location(cls, v):
        # company_location uses the same encoder as employee_residence
        company_location_options = list(encoders["company_location"].classes_)
        if v not in company_location_options:
            raise ValueError(
                f"company_location '{v}' is not a supported country. "
                f"Use GET /supported-countries for the full list."
            )
        return v

    @validator("company_size")
    def validate_company_size(cls, v):
        if v not in SUPPORTED_COMPANY_SIZE:
            raise ValueError(
                f"company_size '{v}' is not valid. "
                f"Choose from: {SUPPORTED_COMPANY_SIZE}"
            )
        return v

    @validator("remote_ratio")
    def validate_remote_ratio(cls, v):
        if v not in [0, 50, 100]:
            raise ValueError(
                f"remote_ratio must be 0 (onsite), 50 (hybrid), or 100 (full remote). Got: {v}"
            )
        return v


# =============================================================================
# PART 6 - ENDPOINTS
# =============================================================================

@app.get("/")
def root():
    """Basic health check — confirms the server is up."""
    return {"status": "ok", "message": "PayLens API is running"}


@app.get("/health")
def health():
    """Returns model metadata and output format."""
    return {
        "status": "ok",
        "model": "RandomForestRegressor",
        "output": "salary_low / salary_avg / salary_high (USD)"
    }


@app.get("/supported-inputs")
def supported_inputs():
    """Returns all valid dropdown values for the prediction form."""
    return {
        "experience_level": SUPPORTED_EXPERIENCE,
        "employment_type":  SUPPORTED_EMPLOYMENT,
        "company_size":     SUPPORTED_COMPANY_SIZE,
        "remote_ratio":     [0, 50, 100],
        "job_titles":       SUPPORTED_JOB_TITLES
    }


@app.get("/supported-countries")
def supported_countries():
    """Returns the list of supported country codes for employee_residence / company_location."""
    return {"countries": SUPPORTED_COUNTRIES}


@app.post("/predict")
def predict(input: PredictionInput):
    """
    Main prediction endpoint.

    Steps:
      1. Fuzzy-match the job title to the closest known title.
      2. Encode all categorical inputs using the saved LabelEncoders.
      3. Build the feature array in the same column order used during training.
      4. Extract p25/p50/p75 salary estimates from individual RF tree predictions.
      5. Look up the benchmark stats for this experience level.
      6. Return the full response.
    """

    # -----------------------------------------------------------------------
    # Step 1: Fuzzy job title matching
    # The user might type "data scientist" or "ML Eng" — we map it to the
    # nearest title the model was trained on.
    # difflib.get_close_matches returns a list; take the first element if any.
    # -----------------------------------------------------------------------
    matches = difflib.get_close_matches(
        input.job_title,
        SUPPORTED_JOB_TITLES,
        n=1,
        cutoff=0.6  # require at least 60% similarity to accept a match
    )

    title_fallback = False  # True when we couldn't find a real match

    if matches:
        matched_title = matches[0]
        # Compute how similar the input title is to the matched title (0.0 to 1.0)
        match_score = difflib.SequenceMatcher(
            None, input.job_title.lower(), matched_title.lower()
        ).ratio()
    else:
        # No match at 60% cutoff — try a relaxed search (any similarity) to find
        # the closest real title for display purposes.
        known_titles = [t for t in SUPPORTED_JOB_TITLES if t != "Other"]
        best = difflib.get_close_matches(input.job_title, known_titles, n=1, cutoff=0.0)
        display_title = best[0] if best else "Data Professional"
        match_score = difflib.SequenceMatcher(
            None, input.job_title.lower(), display_title.lower()
        ).ratio() if best else 0.0

        # The model still predicts using "Other" — only display changes
        matched_title = "Other"
        title_fallback = True

    # -----------------------------------------------------------------------
    # Step 2: Encode categorical inputs
    # We use the exact same LabelEncoder objects saved during training.
    # .transform() maps the string label to its integer code.
    # Note: remote_ratio is a plain integer — it was numeric in training data
    #       and does NOT go through a LabelEncoder.
    # -----------------------------------------------------------------------
    enc_experience  = encoders["experience_level"].transform([input.experience_level])[0]
    enc_employment  = encoders["employment_type"].transform([input.employment_type])[0]
    enc_job_title   = encoders["job_title"].transform([matched_title])[0]
    enc_residence   = encoders["employee_residence"].transform([input.employee_residence])[0]
    enc_location    = encoders["company_location"].transform([input.company_location])[0]
    enc_size        = encoders["company_size"].transform([input.company_size])[0]

    # -----------------------------------------------------------------------
    # Step 3: Build feature DataFrame
    # Column order MUST match model.feature_names_in_ (what the model was trained on):
    #   experience_level, employment_type, job_title,
    #   employee_residence, remote_ratio, company_location, company_size
    #
    # We use a DataFrame (not a bare numpy array) so sklearn does not warn
    # about missing feature names — the column names must match exactly.
    # -----------------------------------------------------------------------
    features = pd.DataFrame([{
        "experience_level":   enc_experience,
        "employment_type":    enc_employment,
        "job_title":          enc_job_title,
        "employee_residence": enc_residence,
        "remote_ratio":       input.remote_ratio,  # numeric — no encoding needed
        "company_location":   enc_location,
        "company_size":       enc_size,
    }])

    # -----------------------------------------------------------------------
    # Step 4: Get salary range from individual tree predictions
    # Each decision tree in the RandomForestRegressor produces its own salary
    # estimate. Taking percentiles across all 100 trees gives us a natural
    # confidence interval: p25=low, p50=median, p75=high.
    # -----------------------------------------------------------------------
    tree_preds = np.array([tree.predict(features.values)[0] for tree in model.estimators_])
    salary_low  = int(np.percentile(tree_preds, 25))
    salary_avg  = int(np.percentile(tree_preds, 50))
    salary_high = int(np.percentile(tree_preds, 75))

    # -----------------------------------------------------------------------
    # Step 5: Look up benchmark for this experience level
    # benchmarks["by_experience"] is a DataFrame indexed on raw experience
    # strings (EN, MI, SE, EX) — NOT the encoded integer.
    # We use input.experience_level (the original string) as the index key.
    # -----------------------------------------------------------------------
    bench_row = benchmarks["by_experience"].loc[input.experience_level]

    # -----------------------------------------------------------------------
    # Step 6: Return the full response
    # -----------------------------------------------------------------------
    return {
        "salary_low":        salary_low,
        "salary_avg":        salary_avg,
        "salary_high":       salary_high,
        "matched_job_title": display_title if title_fallback else matched_title,
        "match_score":       round(match_score, 2),
        "title_fallback":    title_fallback,
        "original_job_title": input.job_title if title_fallback else None,
        "benchmark": {
            "median":     int(bench_row["median"]),
            "p25":        int(bench_row["p25"]),
            "p75":        int(bench_row["p75"]),
            "peer_count": int(bench_row["count"])
        },
        "inputs_received": input.dict()
    }


# =============================================================================
# PART 7 - ENTRY POINT
# Allows running the server directly: python api/main.py
# (The normal way is: uvicorn api.main:app --reload --port 8000)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
