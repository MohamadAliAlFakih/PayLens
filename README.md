# PayLens

**Intelligent salary prediction for data professionals.**

Enter your job details and get a data-driven salary range (low / average / high), peer benchmark comparison, an LLM-generated analyst report, and a downloadable summary — all in one dashboard.

---

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit Dashboard | *(add your Streamlit Cloud URL here)* |
| FastAPI Docs | https://paylens.onrender.com/docs |

---

## Architecture

```
User (Browser)
      │
      ▼
Streamlit Dashboard          — dashboard/app.py
      │
      ├── POST /predict  ──► FastAPI on Render     — api/main.py
      │                           │
      │                           └── RandomForestRegressor → salary_low / avg / high
      │                                + fuzzy job title match + peer benchmarks
      │
      ├── generate_narrative() ──► OpenAI GPT-3.5 (cloud) / Ollama Mistral (local)
      │                                └── returns analyst report text
      │
      ├── upload_chart() ──► Supabase Storage      — public PNG URL
      │
      └── save_prediction() ──► Supabase PostgreSQL — predictions table
```

---

## Request Flow

1. User fills the form and clicks **Predict Salary**
2. Streamlit calls `run_prediction()`, which sends a `POST /predict` to the FastAPI server on Render
3. FastAPI runs the trained model and returns the salary range, benchmark stats, and matched job title
4. Streamlit calls `generate_narrative()` — sends a `POST` to OpenAI (`/v1/chat/completions`) or Ollama locally — and receives the analyst report
5. Charts are generated in-memory by matplotlib on the Streamlit server and rendered directly on the page
6. The experience-level chart is uploaded to Supabase Storage via a `PUT` request and displayed via its public URL
7. The full result is saved to Supabase PostgreSQL via an `INSERT` for history and market insights

---

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "experience_level":   "SE",
  "employment_type":    "FT",
  "job_title":          "Data Scientist",
  "employee_residence": "US",
  "remote_ratio":       100,
  "company_location":   "US",
  "company_size":       "M"
}
```

**Response:**
```json
{
  "salary_low":         160000,
  "salary_avg":         165000,
  "salary_high":        172000,
  "matched_job_title":  "Data Scientist",
  "match_score":        1.0,
  "title_fallback":     false,
  "benchmark": {
    "median": 135000,
    "p25": 105000,
    "p75": 160000,
    "peer_count": 278
  }
}
```

### Field Codes

| Field | Code | Meaning |
|-------|------|---------|
| `experience_level` | EN | Entry-level |
| | MI | Mid-level |
| | SE | Senior |
| | EX | Executive |
| `employment_type` | FT | Full-time |
| | PT | Part-time |
| | CT | Contract |
| | FL | Freelance |
| `company_size` | S | Small (<50 employees) |
| | M | Medium (50–250 employees) |
| | L | Large (>250 employees) |
| `remote_ratio` | 0 | On-site |
| | 50 | Hybrid |
| | 100 | Fully remote |

---

## Model

| Property | Detail |
|----------|--------|
| Algorithm | Random Forest Regressor (100 trees, scikit-learn) |
| Target | `salary_in_usd` — continuous dollar value |
| Salary range | p25 / p50 / p75 across all 100 tree predictions |
| Job title matching | Fuzzy match via `difflib.get_close_matches` (cutoff 0.6) |
| Features | experience level, employment type, job title, residence, remote ratio, company location, company size |

---

## Project Structure

```
paylens/
├── api/
│   └── main.py              # FastAPI server — POST /predict
├── pipeline/
│   ├── predict.py           # run_prediction() — orchestrates the full pipeline
│   ├── visualize.py         # Chart generation + Supabase Storage upload
│   └── narrative.py         # LLM narrative (OpenAI GPT-3.5 / Ollama Mistral)
├── dashboard/
│   └── app.py               # Streamlit UI — Predict / Market Insights / History tabs
├── model/
│   ├── train.py             # Model training script
│   ├── salary_model.pkl     # Trained RandomForestRegressor
│   ├── encoders.pkl         # LabelEncoders for categorical fields
│   └── benchmarks.pkl       # Peer salary benchmarks by experience and title
├── data/
│   └── salaries.csv         # AI/ML jobs salary dataset (~3,000 rows)
├── config.py                # Central config — reads from .env / st.secrets
├── render.yaml              # Render deployment config for FastAPI
└── requirements.txt
```

---

## Deployment

| Component | Platform | Notes |
|-----------|----------|-------|
| FastAPI backend | Render (free tier) | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Streamlit dashboard | Streamlit Cloud | Secrets: `SUPABASE_URL`, `SUPABASE_KEY`, `API_URL`, `OPENAI_API_KEY` |
| Database | Supabase PostgreSQL | `predictions` table — stores every result |
| File storage | Supabase Storage | `charts` public bucket — stores chart PNGs |

---

## Local Setup

**Prerequisites:** Python 3.12, Supabase project, Ollama with `mistral` pulled (for local LLM)

```bash
git clone https://github.com/MohamadAliAlFakih/PayLens.git
cd paylens
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` in the project root:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
OPENAI_API_KEY=sk-...         # optional — falls back to Ollama locally
```

Train the model (first time only):
```bash
python model/train.py
```

Run locally:
```bash
# Terminal 1 — FastAPI backend
python -m uvicorn api.main:app --port 8000

# Terminal 2 — Streamlit dashboard
streamlit run dashboard/app.py
```

Open `http://localhost:8501`

---

## Assignment Context

Built as the Week 1 project for the AI Engineering (AIE39) program.  
Demonstrates: data preprocessing, ML model training, REST API design, LLM integration, cloud persistence, and full-stack deployment across Render, Streamlit Cloud, and Supabase.
