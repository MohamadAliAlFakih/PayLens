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
    prompt = f"""You are a compensation analyst. Write a 3-sentence salary insight.

Profile: {exp_label} {job_title}
Predicted salary range: ${salary_low:,} (low) – ${salary_avg:,} (avg) – ${salary_high:,} (high)
Peer benchmark ({peer_count} peers): median ${median:,}, 25th %ile ${p25:,}, 75th %ile ${p75:,}

Be specific with the dollar amounts. Focus on: (1) what this range means for their profile,
(2) how the predicted average compares to peer median, (3) one actionable tip. No generic advice."""

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
        "salary_low":        95000,
        "salary_avg":        130000,
        "salary_high":       170000,
        "matched_job_title": "Data Scientist",
        "benchmark":         {"median": 135000, "p25": 105000, "p75": 160000, "peer_count": 87},
        "inputs_received":   {"experience_level": "SE"}
    }
    narrative = generate_narrative(mock_result)
    print(narrative)
