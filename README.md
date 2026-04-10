# PayLens

Intelligent salary prediction for data professionals.  
Enter your job details → get a salary range (low / avg / high), peer benchmark, LLM-generated analyst report, and downloadable summary.

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit Dashboard | *coming soon* |
| FastAPI Docs | *coming soon* |

---

## Architecture

```
User
 │
 ▼
Streamlit Dashboard (dashboard/app.py)
 │  ├── calls run_prediction(job_input)  →  FastAPI /predict  →  RandomForestRegressor
 │  ├── generates charts (seaborn/matplotlib)  →  uploads to Supabase Storage
 │  ├── generates narrative (OpenAI GPT-3.5 / Ollama Mistral)
 │  └── saves full result to Supabase predictions table
 │
 ▼
Supabase PostgreSQL (predictions table)
Supabase Storage   (charts bucket — public PNG URLs)
```

---

## Local Development

### Prerequisites
- Python 3.12
- [Ollama](https://ollama.com/) with `mistral` model pulled (`ollama pull mistral`)
- Supabase project (free tier) with `predictions` table and `charts` bucket

### Setup

```bash
git clone <your-repo-url>
cd paylens
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` in the project root:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-publishable-key
OPENAI_API_KEY=sk-...        
```

Train the model (first time only):
```bash
python model/train.py
```

### Run

**Terminal 1 — FastAPI backend:**
```bash
python -m uvicorn api.main:app --port 8000
```

**Terminal 2 — Streamlit dashboard:**
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501`

---

## API Reference

### POST `/predict`

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
  "original_job_title": null,
  "benchmark": {
    "median": 135000, "p25": 105000, "p75": 160000, "peer_count": 278
  },
  "inputs_received": { "...": "..." }
}
```

### Field Codes

**experience_level**
| Code | Meaning |
|------|---------|
| EN | Entry-level |
| MI | Mid-level |
| SE | Senior |
| EX | Executive |

**employment_type**
| Code | Meaning |
|------|---------|
| FT | Full-time |
| PT | Part-time |
| CT | Contract |
| FL | Freelance |

**company_size**
| Code | Meaning |
|------|---------|
| S | Small (<50 employees) |
| M | Medium (50–250 employees) |
| L | Large (>250 employees) |

**remote_ratio**
| Value | Meaning |
|-------|---------|
| 0 | On-site |
| 50 | Hybrid |
| 100 | Fully remote |

---

## Project Structure

```
paylens/
├── api/
│   └── main.py              # FastAPI server (5 endpoints)
├── pipeline/
│   ├── predict.py           # run_prediction() — orchestrates full pipeline
│   ├── visualize.py         # Chart generation + Supabase Storage upload
│   └── narrative.py         # LLM narrative (OpenAI / Ollama)
├── dashboard/
│   └── app.py               # Streamlit dashboard (Predict / History / Market Insights)
├── model/
│   ├── train.py             # Model training script
│   ├── salary_model.pkl     # Trained RandomForestRegressor
│   ├── encoders.pkl         # LabelEncoders for categorical fields
│   └── benchmarks.pkl       # Peer salary benchmarks by experience level
├── data/
│   └── salaries.csv         # AI/ML jobs salary dataset (~3,000 rows)
├── config.py                # Central configuration (reads from .env)
└── requirements.txt
```

---

## Model

- **Algorithm:** Random Forest Regressor (100 trees, scikit-learn)
- **Features:** experience_level, employment_type, job_title, employee_residence, remote_ratio, company_location, company_size
- **Target:** salary_in_usd (continuous — predicts a dollar amount directly)
- **Salary range:** p25/p50/p75 across individual tree predictions gives low/avg/high
- **Job title matching:** fuzzy match via `difflib.get_close_matches` (cutoff 0.6)

---

## Deployment

| Component | Platform | Notes |
|-----------|----------|-------|
| FastAPI backend | Render (free tier) | Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |
| Streamlit dashboard | Streamlit Cloud | Secrets: `SUPABASE_URL`, `SUPABASE_KEY`, `API_URL`, `OPENAI_API_KEY` |
| Database + Storage | Supabase (free tier) | `predictions` table + `charts` public bucket |

---

## Assignment Context

Built as the Week 1 project for the AIE Program (AI Engineering cohort).  
Demonstrates: data preprocessing, ML model training, REST API design, cloud persistence, LLM integration, and full-stack deployment.
