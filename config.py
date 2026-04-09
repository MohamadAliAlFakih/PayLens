# config.py
# Central configuration for the entire PayLens project.
# Every file that needs a path, URL, or setting imports from here.

import os
from dotenv import load_dotenv

# Load the .env file so os.getenv() can read our secrets
load_dotenv()

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
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

# ── Supabase ────────────────────────────────────────────────────────────────
# These are read from .env - never written directly here
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# ── Ollama ──────────────────────────────────────────────────────────────────
# The local LLM model name - change this if you pull a different model
OLLAMA_MODEL = "mistral"

# OpenAI API key — set on cloud deployments to enable LLM narrative via GPT-3.5
# Leave unset locally to use Ollama instead
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = os.path.join(BASE_DIR, "logs", "pipeline.log")