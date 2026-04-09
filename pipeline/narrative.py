# pipeline/narrative.py
# Generates a written salary analyst narrative using Ollama/Mistral.
# Called by pipeline/predict.py after a prediction is made.

import os
import sys
import ollama

# Add the project root (paylens/) to Python's path so config.py can be imported.
# __file__ is pipeline/narrative.py
# dirname(__file__) is pipeline/
# dirname(dirname(...)) is paylens/ — the project root
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import config


def generate_narrative(api_result: dict) -> str:
    """
    Build a structured prompt from the prediction result and benchmark data,
    call Ollama/Mistral locally, and return a written analyst narrative.

    Parameters
    ----------
    api_result : dict
        The full result dict returned by pipeline/predict.py — contains keys:
        'prediction', 'confidence_pct', 'matched_job_title', 'salary_range',
        'benchmark', and 'inputs_received'.

    Returns
    -------
    str
        A 3–5 sentence analyst narrative referencing real benchmark numbers.
        If Ollama is unavailable, returns a plain-text fallback summary instead.
        Always returns a string — never None.
    """

    # ------------------------------------------------------------------
    # Step 1 — Extract values from api_result
    # Each line pulls one piece of data we will use in the prompt below.
    # ------------------------------------------------------------------
    tier        = api_result["prediction"]                           # "Low", "Mid", or "High"
    confidence  = api_result["confidence_pct"]                      # e.g. 47.1
    job_title   = api_result["matched_job_title"]                   # e.g. "Data Scientist"
    experience  = api_result["inputs_received"]["experience_level"] # "EN"/"MI"/"SE"/"EX"
    salary_min  = api_result["salary_range"]["min"]
    # salary_range["max"] is None for the High tier — convert to the string "above"
    # so all downstream formatting can treat salary_max as a plain value with no None checks.
    raw_max     = api_result["salary_range"].get("max")
    salary_max  = "above" if raw_max is None else raw_max           # str or int, never None
    median      = api_result["benchmark"]["median"]
    p25         = api_result["benchmark"]["p25"]
    p75         = api_result["benchmark"]["p75"]
    peers       = api_result["benchmark"]["peer_count"]

    # Map experience codes to readable labels for the prompt
    exp_labels = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
    exp_label  = exp_labels.get(experience, experience)

    # ------------------------------------------------------------------
    # Step 2 — Build the prompt
    # The prompt is structured to force Mistral to use the actual numbers —
    # generic prompts produce generic output.
    # ------------------------------------------------------------------
    prompt = f"""You are a data analyst writing a salary insight report for a job applicant.

Job details:
- Role: {job_title}
- Experience: {exp_label}

Prediction result:
- Salary tier: {tier} (model confidence: {confidence}%)
- Salary range for this tier: ${salary_min:,} to {salary_max if isinstance(salary_max, str) else f"${salary_max:,}"}

Peer benchmark ({peers} similar professionals in the dataset):
- Median salary: ${median:,}
- 25th percentile: ${p25:,}
- 75th percentile: ${p75:,}

Write a 3-4 sentence analyst commentary that:
1. States the predicted tier and what it means in dollar terms
2. Compares this to the peer group using the actual numbers above
3. Gives one practical insight for the applicant (negotiation, market position, etc.)

Be specific. Use the numbers. Do not give generic salary advice."""

    # ------------------------------------------------------------------
    # Step 3 — Call Ollama and return the generated narrative
    # The fallback is a plain-text summary using the same data —
    # dashboard always gets a string, never None.
    # ------------------------------------------------------------------
    try:
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        # response["message"]["content"] is the text Mistral generated
        narrative = response["message"]["content"].strip()
        return narrative
    except Exception as e:
        # Ollama not running, model not pulled, or any other failure
        print("Warning: Ollama unavailable:", e)
        fallback = (
            f"Predicted salary tier: {tier} ({confidence}% confidence). "
            f"Peer median for {exp_label} {job_title} roles: ${median:,}. "
            f"Salary range for this tier: ${salary_min:,}–"
            f"{'above' if isinstance(salary_max, str) else f'${salary_max:,}'}."
        )
        return fallback


# ── Manual test ─────────────────────────────────────────────────────────────
# Run: python pipeline/narrative.py  (from paylens/ project root, venv active)
# Confirms Mistral returns a paragraph mentioning real numbers, or prints the
# fallback cleanly if Ollama is not running.
if __name__ == "__main__":
    mock_result = {
        "prediction":        "High",
        "confidence_pct":    72.3,
        "matched_job_title": "Data Scientist",
        "salary_range":      {"min": 120000, "max": None, "currency": "USD"},
        "benchmark":         {"median": 135000, "p25": 105000, "p75": 160000, "peer_count": 87},
        "inputs_received":   {"experience_level": "SE"}
    }
    narrative = generate_narrative(mock_result)
    print(narrative)
