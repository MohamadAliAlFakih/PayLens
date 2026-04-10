# config.py
# Central configuration for the entire PayLens project.
# Every file that needs a path, URL, or setting imports from here.

import os
from dotenv import load_dotenv

# Load the .env file so os.getenv() can read our secrets
load_dotenv()


def _get_secret(key, default=None):
    """
    Read a secret from st.secrets (Streamlit Cloud) or os.getenv (local .env).
    Streamlit Cloud does not inject secrets as environment variables, so we
    must check st.secrets first. Falls back to os.getenv for local runs and
    non-Streamlit contexts like the FastAPI server.
    """
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return val
    except Exception:
        pass
    return os.getenv(key, default)


# ── Paths ──────────────────────────────────────────────────────────────────
# The base directory is wherever config.py lives (the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the trained model and encoders will be saved
MODEL_PATH = os.path.join(BASE_DIR, "model", "salary_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "model", "encoders.pkl")

# Where the raw dataset lives
DATA_PATH = os.path.join(BASE_DIR, "data", "salaries.csv")

# Where generated charts get saved before uploading to Supabase
CHART_PATH = os.path.join(BASE_DIR, "logs", "chart.png")

# ── API ─────────────────────────────────────────────────────────────────────
# The FastAPI server address — reads from env on cloud (Render URL), defaults to local
API_URL = _get_secret("API_URL", "http://127.0.0.1:8000/predict")

# ── Supabase ────────────────────────────────────────────────────────────────
# These are read from .env locally and from st.secrets on Streamlit Cloud
SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_KEY")

# ── Ollama ──────────────────────────────────────────────────────────────────
# The local LLM model name - change this if you pull a different model
OLLAMA_MODEL = "mistral"

# OpenAI API key — set on cloud deployments to enable LLM narrative via GPT-3.5
# Leave unset locally to use Ollama instead
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(BASE_DIR, "logs", "pipeline.log")