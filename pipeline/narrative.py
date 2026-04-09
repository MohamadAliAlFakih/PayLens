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
    Generate a salary analyst narrative for the prediction result.

    Uses OpenAI GPT-3.5 if OPENAI_API_KEY is set in config (cloud deployments).
    Falls back to Ollama/Mistral for local development.
    Always returns a string — never raises an exception.
    """
    # --- Extract values ---
    tier        = api_result.get("prediction", "Unknown")
    confidence  = api_result.get("confidence_pct", 0)
    job_title   = api_result.get("matched_job_title", "this role")
    exp_code    = api_result.get("inputs_received", {}).get("experience_level", "SE")
    exp_labels  = {"EN": "Entry-level", "MI": "Mid-level", "SE": "Senior", "EX": "Executive"}
    exp_label   = exp_labels.get(exp_code, exp_code)

    salary_range = api_result.get("salary_range", {})
    salary_min   = salary_range.get("min", 0)
    raw_max      = salary_range.get("max")
    salary_max   = "above" if raw_max is None else raw_max

    benchmark   = api_result.get("benchmark", {})
    median      = benchmark.get("median", 0)
    p25         = benchmark.get("p25", 0)
    p75         = benchmark.get("p75", 0)
    peer_count  = benchmark.get("peer_count", 0)

    # --- Fallback string (used if both LLM paths fail) ---
    fallback = (
        f"Predicted salary tier: {tier} ({confidence}% confidence). "
        f"Peer median for {exp_label} {job_title} roles: ${median:,}. "
        f"Salary range for this tier: ${salary_min:,}–{salary_max if salary_max == 'above' else f'${salary_max:,}'}."
    )

    # --- Build prompt ---
    sal_max_str = "above" if salary_max == "above" else f"${salary_max:,}"
    prompt = f"""You are a compensation analyst. Write a 3-sentence salary insight for this professional.

Profile: {exp_label} {job_title}
Predicted tier: {tier} ({confidence}% confidence)
Salary range: ${salary_min:,} – {sal_max_str}
Peer benchmark ({peer_count} peers): median ${median:,}, 25th %ile ${p25:,}, 75th %ile ${p75:,}

Be specific with the dollar amounts above. Focus on: (1) what this tier means for their career stage, (2) how they compare to peers, (3) one actionable tip. No generic advice."""

    # --- OpenAI path (cloud) ---
    if config.OPENAI_API_KEY:
        try:
            import openai
            client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: OpenAI unavailable: {e}")
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
        "prediction":        "High",
        "confidence_pct":    72.3,
        "matched_job_title": "Data Scientist",
        "salary_range":      {"min": 120000, "max": None, "currency": "USD"},
        "benchmark":         {"median": 135000, "p25": 105000, "p75": 160000, "peer_count": 87},
        "inputs_received":   {"experience_level": "SE"}
    }
    narrative = generate_narrative(mock_result)
    print(narrative)
