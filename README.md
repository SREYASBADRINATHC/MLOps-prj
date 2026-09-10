# CartSense — Drift-Resilient Hybrid Recommendation Engine for Electronics

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![PySpark](https://img.shields.io/badge/PySpark-3.5-E25A1C.svg)](https://spark.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2.svg)](https://mlflow.org)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.37-purple.svg)](https://evidentlyai.com)

---

## 1. Project Overview

CartSense is a production-grade electronics recommendation system that demonstrates:

1. **Item Cold-Start Resolution** — New products with zero interaction history receive content-based recommendations via TF-IDF and cosine similarity, bypassing ALS completely.
2. **Concept Drift Resilience** — Evidently AI monitors the distribution of product features interacted with in production. When drift exceeds a configurable threshold, a full retraining pipeline is triggered, evaluated, and promoted through MLflow.

Both models are **trained from scratch on our own electronics dataset** — no pretrained LLMs, no external embeddings.

---

## 2. Problem Statement

| Problem | Description |
|---|---|
| **Item Cold Start** | A brand-new laptop or smartphone has zero user interactions. Matrix factorization (ALS) cannot generate predictions for it. |
| **Concept Drift** | Consumer hardware preferences shift over time (e.g., market moves from IPS/60Hz to OLED/120Hz+). A model trained on historical data degrades in quality as preferences shift. |

---

## 3. Architecture

```
USER
  │
  ▼
STREAMLIT (port 8501)
  │ HTTP REST
  ▼
FASTAPI (port 8000)
  │
  ▼
HYBRID ROUTER
  ├── interaction_count == 0  → COLD-START PATH
  │       TF-IDF + Cosine Similarity + Spec Match
  │
  └── interaction_count > 0 + known user  → HYBRID PATH
          α·ALS + (1-α)·TF-IDF + β·SpecBoost
  
  FinalScore = α × NormalizedALS + (1-α) × TF-IDF + β × SpecificationMatch
  Default: α=0.65, β=0.20

PostgreSQL → Evidently AI → Drift Detection
  │
  └── Drift Detected?
       YES → Retraining Pipeline
              ├── generate data
              ├── train TF-IDF
              ├── train ALS (PySpark)
              ├── evaluate (temporal split)
              ├── quality gate (RMSE ≤ prod × 1.05)
              ├── MLflow registration
              └── FastAPI hot-reload
```

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.38 |
| API | FastAPI 0.115 + Uvicorn |
| Validation | Pydantic v2 |
| Content Filtering | scikit-learn TfidfVectorizer |
| Collaborative Filtering | PySpark MLlib ALS |
| Database | PostgreSQL 15 + SQLAlchemy 2.0 |
| MLOps | MLflow 2.16 |
| Drift Detection | Evidently AI 0.4.37 |
| Containerization | Docker + Docker Compose |

---

## 5. Quick Start

### Prerequisites
- Docker Desktop (with Linux containers)
- 8 GB RAM minimum (PySpark ALS needs memory)

### Run with Docker Compose

```bash
git clone <repo>
cd MLOps-prj

# Start everything (init service runs data gen + training automatically)
docker compose up --build

# Access:
#   Streamlit:  http://localhost:8501
#   FastAPI:    http://localhost:8000/docs
#   MLflow:     http://localhost:5000
```

The `cartsense_init` service will automatically:
1. Generate 1000 products, 500 users, 12,000+ interactions
2. Train TF-IDF on the catalog
3. Train PySpark ALS on interaction data
4. Save artifacts to the shared volume

### Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL (e.g. via docker)
docker run -d -e POSTGRES_USER=cartsense_user -e POSTGRES_PASSWORD=cartsense_pass \
  -e POSTGRES_DB=cartsense -p 5432:5432 postgres:15-alpine

# Set environment
export DATABASE_URL=postgresql+psycopg2://cartsense_user:cartsense_pass@localhost:5432/cartsense
export MLFLOW_TRACKING_URI=./mlruns
export ARTIFACTS_DIR=./artifacts

# Generate data + train
python training/generate_data.py
python training/train_tfidf.py
python training/train_als.py

# Start API
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
streamlit run frontend/app.py
```

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | postgres://... | PostgreSQL connection string |
| `MLFLOW_TRACKING_URI` | http://mlflow_server:5000 | MLflow server URL |
| `ARTIFACTS_DIR` | /app/artifacts | Model artifact storage |
| `ALPHA` | 0.65 | ALS weight in hybrid blend |
| `BETA` | 0.20 | Specification match boost |
| `TOP_K` | 5 | Default recommendation count |
| `ALS_RANK` | 20 | ALS latent factor rank |
| `ALS_MAX_ITER` | 15 | ALS training iterations |
| `ALS_REG_PARAM` | 0.1 | ALS regularization |
| `DRIFT_THRESHOLD` | 0.5 | Fraction of drifted features to trigger retraining |

---

## 7. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System health check |
| GET | `/api/products` | Product catalog (filterable) |
| GET | `/api/products/{id}` | Full product details |
| GET | `/api/users/{id}` | User profile |
| POST | `/api/recommend` | Get hybrid recommendations |
| GET | `/api/drift` | Latest drift check result |
| POST | `/api/drift/check` | Run Evidently drift analysis |
| POST | `/api/retrain` | Trigger retraining pipeline |
| POST | `/api/model/reload` | Hot-reload model artifacts |
| GET | `/api/model/status` | Current model information |
| GET | `/api/training/history` | Historical training runs |
| GET | `/api/stats` | System statistics |

### Recommendation Request

```json
POST /api/recommend
{
    "user_id": "U0001",
    "product_id": "L0042",
    "required_specs": "32GB RAM OLED 144Hz",
    "top_k": 5
}
```

### Recommendation Response

```json
{
    "request_id": "a8f3b...",
    "user_id": "U0001",
    "product_id": "L0042",
    "recommendation_mode": "hybrid",
    "model_version": "v_2024-01-15",
    "latency_ms": 38.4,
    "als_used": true,
    "tfidf_used": true,
    "spec_matching_used": true,
    "recommendations": [
        {
            "product_id": "L0089",
            "name": "ASUS ROG Zephyrus G16 Gen 3",
            "als_score": 0.81,
            "tfidf_score": 0.76,
            "spec_match_score": 0.92,
            "final_score": 0.84,
            "reason": "Recommended because: similar users have interacted with this product; matches required: ram, display, refresh_rate"
        }
    ]
}
```

---

## 8. TF-IDF Methodology

**Training:**
- Fit `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)` on all 1000 product `spec_text` documents
- Vocabulary and IDF statistics come entirely from our own dataset
- `sublinear_tf=True`: applies log(1+tf) to prevent dominance of repeated terms
- `ngram_range=(1,2)`: captures "32GB", "OLED display", "144Hz gaming", etc.

**Inference:**
- Transform anchor product spec_text into TF-IDF vector
- Compute cosine similarity against all same-category products
- Apply specification matching boost
- Return sorted candidates

---

## 9. ALS Methodology

**Training:**
- Real PySpark MLlib ALS: `ALS(rank=20, maxIter=15, regParam=0.1, nonnegative=True)`
- Interaction weights aggregated: view=1, click=2, wishlist=3, cart=4, purchase=5
- Temporal split: oldest 80% interactions → training, newest 20% → evaluation
- User and item latent factors extracted via `model.userFactors` / `model.itemFactors`

**Inference (no Spark at request time):**
- Pre-computed item factor matrix loaded as numpy array at startup
- Recommendation: user_vector × item_matrix dot product → sort → top-K
- Unknown users → fallback to content-based (labeled `new_user_fallback`)

---

## 10. Hybrid Scoring

```
FinalScore = α × NormalizedALS + (1-α) × TF-IDF + β × SpecificationMatch

Where:
  α = 0.65  (collaborative weight, configurable via ALPHA env var)
  β = 0.20  (specification boost, configurable via BETA env var)

All components normalized to [0, 1] via min-max scaling before blending.
```

---

## 11. Cold-Start Workflow

```
New Product Added (interaction_count = 0)
           ↓
POST /api/recommend
           ↓
Hybrid Router: interaction_count == 0?
           ↓ YES
recommendation_mode = "cold_start"
           ↓
ALS: SKIPPED (no behavioral data)
           ↓
TF-IDF: ACTIVE
  - Transform product spec_text
  - Cosine similarity vs all same-category products
           ↓
Specification Matching: ACTIVE (if required_specs provided)
           ↓
FinalScore = (1-β) × TF-IDF + β × SpecMatch
           ↓
Top-K returned with explanation
```

---

## 12. Drift Detection

**Monitored features:**
- `ram_gb` — RAM distribution
- `price_usd` — Price distribution  
- `refresh_rate_hz` — Refresh rate distribution
- `storage_gb` — Storage distribution
- `display_encoded` — Display type (OLED=2, IPS=1, other=0)
- `battery_mah` — Battery capacity

**Method:** Evidently `DataDriftPreset` using Kolmogorov-Smirnov test per feature

**Threshold:** `DRIFT_THRESHOLD=0.5` means: if ≥50% of monitored features show statistically significant shift (KS p-value < 0.05), declare dataset drift.

**Simulation:** The Drift Simulation page generates a synthetic "drifted" distribution reflecting the documented market shift (16GB/IPS/60Hz → 32GB/OLED/120Hz) to demonstrate the pipeline.

---

## 13. Continuous Training Pipeline

```
POST /api/retrain
       ↓
training/retrain_pipeline.py
       ↓
1. Load data from PostgreSQL
2. Validate (min rows, no nulls)
3. Train TF-IDF (from scratch)
4. Train PySpark ALS (from scratch)
5. Evaluate: RMSE, P@5, R@5, NDCG@5
6. MLflow: log params + metrics
7. Quality Gate: new_rmse ≤ prod_rmse × 1.05?
       ├── PASS → save artifacts → register MLflow → notify API
       └── FAIL → reject → keep production model
8. POST /api/model/reload → hot-swap artifacts
```

---

## 14. Testing

```bash
# Run all unit tests
pytest tests/ -v

# Run specific test files
pytest tests/test_tfidf.py -v     # Spec matching tests
pytest tests/test_hybrid.py -v    # Hybrid routing tests
pytest tests/test_als.py -v       # ALS inference tests
pytest tests/test_drift.py -v     # Drift detection tests
pytest tests/test_api.py -v       # FastAPI integration tests
```

---

## 15. Project Structure

```
CartSense/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
│
├── backend/
│   ├── Dockerfile
│   ├── main.py           ← FastAPI app with all endpoints
│   ├── config.py         ← Centralized settings (env vars)
│   ├── database.py       ← SQLAlchemy engine + session
│   ├── models.py         ← ORM models (7 tables)
│   ├── schemas.py        ← Pydantic v2 request/response
│   ├── hybrid_engine.py  ← HybridRecommendationEngine
│   ├── model_manager.py  ← Thread-safe artifact manager
│   ├── als_inference.py  ← Numpy-based ALS scoring
│   ├── spec_matcher.py   ← Specification parsing + scoring
│   └── drift_monitor.py  ← Evidently drift detection
│
├── frontend/
│   ├── Dockerfile
│   └── app.py            ← 7-page Streamlit dashboard
│
├── training/
│   ├── generate_data.py  ← 1000 products, 500 users, 12k+ interactions
│   ├── train_tfidf.py    ← TF-IDF training + MLflow
│   ├── train_als.py      ← PySpark ALS + factor export
│   ├── evaluate.py       ← Metrics + baseline comparison
│   └── retrain_pipeline.py ← Full CT pipeline
│
├── tests/
│   ├── test_api.py
│   ├── test_tfidf.py     ← Spec matcher tests
│   ├── test_hybrid.py    ← Hybrid routing tests
│   ├── test_als.py       ← ALS inference tests
│   └── test_drift.py     ← Drift detection tests
│
└── artifacts/            ← Model artifacts (auto-generated)
    ├── tfidf_vectorizer.joblib
    ├── tfidf_matrix.npz
    ├── product_id_map.json
    ├── tfidf_metadata.json
    ├── als_user_factors.parquet
    ├── als_item_factors.parquet
    ├── als_id_maps.json
    └── als_metadata.json
```

---

## 16. Evaluation Methodology

**Temporal split** (time-aware evaluation):
- Training: oldest 80% of interactions (earlier market behavior)
- Test: newest 20% of interactions (later market behavior)

This answers: *"Can a model trained on earlier market behavior generalize to future user preferences?"*

**Metrics:**
- **RMSE**: rating prediction error
- **Precision@K**: fraction of top-K that user actually interacted with in test set
- **Recall@K**: fraction of test interactions captured in top-K
- **NDCG@K**: normalized discounted cumulative gain (ranking quality)

---

## 17. Known Limitations

1. **Spark startup time**: ALS training requires PySpark JVM startup (~30-60s). Inference is numpy-only (no Spark).
2. **Cold-start precision**: Without behavioral signals, TF-IDF precision depends entirely on spec text quality.
3. **ALS implicit feedback**: We use explicit aggregated weights, not true implicit feedback. Pure implicit ALS (Hu et al. 2008) could improve cold-implicit scenarios.
4. **Single-node deployment**: PySpark runs in local mode — not distributed. Production would use a cluster.
5. **Drift threshold**: The 0.5 share-of-features threshold is a configurable heuristic, not analytically derived.

---

## 18. Future Improvements

- Two-tower neural retrieval (replace TF-IDF)
- Real-time ALS incremental updates (without full retraining)
- User cold-start via demographic embedding
- A/B testing framework for model comparison
- Kubernetes deployment with HPA
- Streaming interactions via Kafka → Flink
