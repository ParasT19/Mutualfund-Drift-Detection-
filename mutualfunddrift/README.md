# MutualFundDrift — Indian Mutual Fund Style Drift Detector

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)

## Overview

MutualFundDrift is a production-grade platform that automatically detects and predicts **style drift** in Indian mutual funds. Style drift occurs when a fund manager gradually shifts the portfolio away from the fund's SEBI-mandated investment mandate — for example, a Mid Cap Fund that quietly accumulates large cap stocks. SEBI's October 2017 categorisation circular mandated that Large Cap Funds must hold ≥80% in the top 100 NSE stocks by market cap, Mid Cap Funds ≥65% in ranks 101–250, and Small Cap Funds ≥65% in ranks 251+. MutualFundDrift enforces these mandates continuously.

The platform ingests AMFI monthly portfolio disclosures, computes Morningstar-style box coordinates for each fund, measures Euclidean drift from the SEBI mandate coordinate, predicts imminent drift using XGBoost, explains predictions using SHAP values, and generates plain-English investor alerts using the Anthropic Claude API — all visualised through an interactive Streamlit dashboard.

## Features

- 📊 **Morningstar Style Box Journey Chart** — plots each fund's quarterly (size, style) coordinate trajectory on a 3×3 grid showing drift from mandate
- 🔬 **Multi-Dimensional Drift Detection** — tracks 6 drift signals: Euclidean style score, sector HHI, rolling NAV correlation, active share, top-10 concentration, and turnover
- 🤖 **XGBoost Early Warning** — predicts style drift 1 quarter in advance with 19 engineered features using time-series cross-validation
- 🧠 **SHAP Explainability** — identifies the top 5 features driving each drift prediction in plain English
- 🤝 **Claude AI Investor Alerts** — generates cited, factual investor warnings via the Anthropic Claude API with deterministic fallback
- 📅 **Automated Ingestion** — APScheduler runs weekly to pull AMFI disclosures, compute metrics, and escalate alerts
- 🌐 **FastAPI REST Backend** — 20+ production endpoints with Pydantic validation, SQLAlchemy ORM, and OpenAPI docs
- 🎛️ **4-Page Streamlit Dashboard** — Dashboard, Fund Analyser, Compare Funds, and Alerts Centre with Plotly, Seaborn, and NetworkX charts

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Ingestion | mftool, requests | AMFI portfolio disclosures |
| Database | SQLite + SQLAlchemy + SQLModel | Persistent storage |
| Core Engine | pandas, numpy | Portfolio analytics |
| ML Model | XGBoost, scikit-learn, SHAP | Drift prediction & explainability |
| Alert Generation | Rule-based engine | Plain-English investor alerts |
| Backend API | FastAPI, uvicorn, Pydantic | REST API layer |
| Visualisations | Matplotlib, Seaborn, Plotly, NetworkX | Charts & graphs |
| Frontend | Streamlit | Interactive dashboard |

## Project Structure

```
mutualfunddrift/
├── backend/               # FastAPI application
│   ├── config.py          # pydantic-settings configuration
│   ├── database.py        # SQLAlchemy engine & sessions
│   ├── models.py          # SQLModel ORM tables
│   ├── schemas.py         # Pydantic v2 request/response schemas
│   ├── main.py            # FastAPI app entry point
│   └── routers/           # API route handlers
│       ├── funds.py       # Fund CRUD + chart endpoints
│       ├── drift.py       # Drift scores, predictions, comparison
│       └── alerts.py      # Alert management
├── engine/                # Core analytics pipeline
│   ├── data_ingestion.py  # AMFI data fetching
│   ├── classification.py  # SEBI market cap classification
│   ├── style_box.py       # Style box coordinate computation
│   ├── drift_scorer.py    # Euclidean drift scoring
│   ├── feature_engineer.py # XGBoost feature engineering
│   ├── predictor.py       # XGBoost training & inference
│   └── alert_generator.py # Anthropic Claude alert generation
├── visualisations/        # Chart rendering modules
│   ├── style_box_chart.py
│   ├── composition_chart.py
│   ├── correlation_heatmap.py
│   ├── sector_drift_chart.py
│   ├── drift_timeline.py
│   └── network_graph.py
├── scheduler/
│   └── jobs.py            # APScheduler weekly jobs
├── frontend/
│   └── app.py             # Streamlit 4-page dashboard
├── data/                  # Seed data files
│   ├── fund_mandates.csv
│   ├── nse_classification.csv
│   └── sample_portfolio.csv
├── tests/                 # pytest test suites
├── models_saved/          # XGBoost model artifacts
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Setup and Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerised setup)
- An Anthropic API key (optional — fallback alerts work without it)

### Clone and Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd mutualfunddrift

# Copy environment file and fill in your API key
cp .env.example .env
```

Edit `.env` and set your `ANTHROPIC_API_KEY`.

### Docker Setup (Recommended)

```bash
# Build and start all services (postgres, backend, frontend)
docker-compose up --build

# Verify backend health
curl http://localhost:8000/health

# Open the Streamlit dashboard
# http://localhost:8501

# Explore the API
# http://localhost:8000/docs
```

### Verify at http://localhost:8501

The Streamlit dashboard will open automatically. Select any fund from the sidebar to see its style box journey, drift score timeline, and sector heatmap.

## Running Without Docker

### Virtual Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### PostgreSQL Setup

```bash
# Install PostgreSQL locally, then:
createuser -P mfdrift_user          # Password: mfdrift_pass
createdb -O mfdrift_user mfdrift_db
```

### Start Services

```bash
# Terminal 1: Start FastAPI backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run frontend/app.py
```

## Loading Seed Data

To register the 10 funds from `fund_mandates.csv` into the database:

```bash
python -c "
from backend.database import get_db_context, create_db_and_tables
from backend.models import Fund
from engine.data_ingestion import load_fund_mandates
from datetime import datetime

create_db_and_tables()
mandates = load_fund_mandates()

with get_db_context() as db:
    for _, row in mandates.iterrows():
        fund = Fund(
            scheme_code=str(row['scheme_code']),
            scheme_name=row['scheme_name'],
            amc_name='Various AMC',
            category=row['category'],
            sub_category='',
            benchmark_index=row['benchmark_index'],
            mandate_size_score=float(row['mandate_size_score']),
            mandate_style_score=float(row['mandate_style_score']),
            active=True,
            created_at=datetime.utcnow()
        )
        db.merge(fund)
    print(f'Loaded {len(mandates)} funds.')
"
```

## How Drift Detection Works

Each monthly portfolio snapshot is transformed into a two-dimensional coordinate **(size_score, style_score)** on a 3×3 Morningstar-style grid.

**Size Score** (x-axis, 0.0 to 1.0): Each holding is assigned 1.0 (large cap), 0.5 (mid cap), or 0.0 (small cap) based on its NSE rank. The portfolio-weighted average gives the size score. A fund with 80% large caps scores ~0.90.

**Style Score** (y-axis, 0.0 to 1.0): Each holding's P/B ratio is divided by the median P/B for its market cap category to get a relative P/B. These are min-max normalised to [0,1] across the portfolio. Low P/B = value (0.0), high P/B = growth (1.0).

**Drift Score** = √((size_score − mandate_size)² + (style_score − mandate_style)²)

A Mid Cap Blend fund has mandate coordinate (0.5, 0.5). If its current coordinate is (0.74, 0.60), the drift score is √(0.0576 + 0.01) = 0.26 — triggering an **Amber** alert.

## Running Tests

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test module
pytest tests/test_style_box.py -v
pytest tests/test_drift_scorer.py -v
pytest tests/test_schemas.py -v

# Run with coverage report
pytest tests/ --cov=engine --cov=backend --cov-report=term-missing
```

## API Documentation

After starting the backend, interactive documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Key endpoints:
- `GET /api/funds` — list all tracked funds
- `GET /api/drift/{code}/score` — current drift score and severity
- `GET /api/drift/{code}/predict` — XGBoost prediction for a fund
- `GET /api/drift/leaderboard` — all funds ranked by drift score
- `POST /api/drift/compare` — compare 2–6 funds with network graph
- `POST /api/alerts/{code}` — trigger a fresh LLM alert for a fund

## Data Sources

| Source | Description | URL |
|---|---|---|
| AMFI India | Monthly portfolio disclosures and NAV data | https://www.amfiindia.com |
| NSE India | Market cap rankings updated semi-annually by AMFI | https://www.amfiindia.com/research-information/other-data/categorization |

## Extending the Project

### Adding More Funds

1. Add rows to `data/fund_mandates.csv` with scheme_code, mandate coordinates, and benchmark.
2. Run the seed loading script above.
3. Call `POST /api/funds/{new_scheme_code}` to register via API.
4. The weekly scheduler will automatically ingest portfolio data on the next run.

### Retraining the Model

After accumulating 12+ monthly snapshots per fund:

```python
from backend.database import get_db_context
from backend.models import PortfolioSnapshot
from engine.feature_engineer import prepare_training_data
from engine.predictor import train_model
import pandas as pd
from sqlmodel import select

with get_db_context() as db:
    all_snapshots = pd.read_sql(select(PortfolioSnapshot), db.bind)

X, y = prepare_training_data(all_snapshots)
model, mean_auc = train_model(X, y)
print(f"Model trained — mean AUC: {mean_auc:.4f}")
```

The model is saved to `models_saved/xgb_drift_model.joblib` and automatically loaded by subsequent `predict_drift()` calls.
