# pipeline/narrative.py
# Generates a written salary analyst narrative using Gemini (cloud) or Ollama (local).
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
    Generate a salary analyst narrative for the prediction result.

    Uses OpenAI GPT-3.5 if OPENAI_API_KEY is set in config (cloud deployments).
    Falls back to Ollama/Mistral for local development.
    Always returns a string — never raises an exception.
    """
    # --- Extract values ---
    salary_low  = api_result.get("salary_low", 0)
    salary_avg  = api_result.get("salary_avg", 0)
    salary_high = api_result.get("salary_high", 0)
    job_title   = api_result.get("matched_job_title", "this role")
    exp_code    = api_result.get("inputs_received", {}).get("experience_level", "SE")
    exp_label   = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}.get(exp_code, exp_code)

    benchmark   = api_result.get("benchmark", {})
    median      = benchmark.get("median", 0)
    p25         = benchmark.get("p25", 0)
    p75         = benchmark.get("p75", 0)
    peer_count  = benchmark.get("peer_count", 0)

    # --- Fallback string (used if both LLM paths fail) ---
    fallback = (
        f"For a {exp_label} {job_title}, predicted salary range is "
        f"${salary_low:,}–${salary_high:,} with an average of ${salary_avg:,}. "
        f"Peer median ({peer_count} peers): ${median:,}."
    )

    # --- Build prompt ---
    prompt = f"""You are a compensation analyst. Write a 3-sentence salary insight in plain prose.

Profile: {exp_label} {job_title}
Predicted salary range: low ${salary_low:,}, average ${salary_avg:,}, high ${salary_high:,}
Peer benchmark based on {peer_count} peers: median ${median:,}, 25th percentile ${p25:,}, 75th percentile ${p75:,}

Rules:
- Do NOT add labels like (low), (high), (avg), (median) after dollar amounts
- Do NOT use parentheses around salary labels
- Use plain dollar amounts inline, e.g. "a range of $57,000 to $103,000"
- Focus on: (1) what this range means for their profile, (2) how their average compares to the peer median, (3) one actionable tip
- No generic advice. Be specific."""

    # --- Gemini path (cloud) ---
    # Read at call time so st.secrets is available (not frozen at import time)
    try:
        import streamlit as st
        gemini_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        gemini_key = os.getenv("GEMINI_API_KEY")

    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            print(f"Warning: Gemini unavailable: {e}")
            return fallback

    # --- Ollama path (local) ---
    try:
        response = ollama.chat(model=config.OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Warning: Ollama unavailable: {e}")
        return fallback


# ── Manual test ─────────────────────────────────────────────────────────────
# Run: python pipeline/narrative.py  (from paylens/ project root, venv active)
# Confirms Mistral returns a paragraph mentioning real numbers, or prints the
# fallback cleanly if Ollama is not running.
if __name__ == "__main__":
    mock_result = {
        "salary_low":        95000,
        "salary_avg":        130000,
        "salary_high":       170000,
        "matched_job_title": "Data Scientist",
        "benchmark":         {"median": 135000, "p25": 105000, "p75": 160000, "peer_count": 87},
        "inputs_received":   {"experience_level": "SE"}
    }
    narrative = generate_narrative(mock_result)
    print(narrative)
