"""
CartSense — FastAPI Backend
Drift-Resilient Hybrid Recommendation Engine for Electronics
"""
from __future__ import annotations

import logging
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from backend.config import settings
from backend.database import check_database_health, engine, scalar_sql_safe
from backend.drift_monitor import DriftMonitor
from backend.hybrid_engine import HybridRecommendationEngine
from backend.model_manager import model_manager
from backend.schemas import (
    DriftCheckRequest,
    DriftMetricsResponse,
    HealthCheck,
    ModelStatus,
    ProductSpec,
    ProductSummary,
    RecommendationRequest,
    RecommendationResponse,
    RetrainRequest,
    RetrainingStatus,
    TrainingRunSummary,
    UserProfile,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("cartsense.api")

# ── Application-level singletons ──────────────────────────────────────────────
_products_cache: pd.DataFrame = pd.DataFrame()
_hybrid_engine: HybridRecommendationEngine | None = None
_drift_monitor: DriftMonitor | None = None
_mlflow_ok: bool = False


def _build_hybrid_engine() -> HybridRecommendationEngine | None:
    """Construct the HybridRecommendationEngine from loaded artifacts."""
    arts = model_manager.artifacts
    if not arts.tfidf_loaded:
        logger.warning("TF-IDF not loaded — hybrid engine unavailable.")
        return None
    if _products_cache.empty:
        logger.warning("Products cache empty — hybrid engine unavailable.")
        return None
    return HybridRecommendationEngine(
        tfidf_vectorizer=arts.tfidf_vectorizer,
        tfidf_matrix=arts.tfidf_matrix,
        product_ids=arts.product_ids,
        als_engine=arts.als_engine,
        products_df=_products_cache,
        alpha=settings.alpha,
        beta=settings.beta,
        model_version=arts.model_version,
    )


def _check_mlflow() -> bool:
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.search_experiments()
        return True
    except Exception:
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load models on startup, release on shutdown."""
    global _products_cache, _hybrid_engine, _drift_monitor, _mlflow_ok

    logger.info("CartSense API starting up...")

    # ── Load products into memory ─────────────────────────────────────────────
    try:
        with engine.begin() as conn:
            _products_cache = pd.read_sql(text("SELECT * FROM products ORDER BY product_id"), conn)
        logger.info("Products cache loaded: %d records.", len(_products_cache))
    except Exception as exc:
        logger.error("Failed to load products: %s — check database and run generate_data.py", exc)
        _products_cache = pd.DataFrame()

    # ── Load model artifacts ──────────────────────────────────────────────────
    model_manager.load(settings.artifacts_dir)

    # ── Build hybrid engine ───────────────────────────────────────────────────
    _hybrid_engine = _build_hybrid_engine()
    if _hybrid_engine:
        logger.info("Hybrid recommendation engine ready.")
    else:
        logger.warning("Hybrid engine not available — run training scripts first.")

    # ── Drift monitor ─────────────────────────────────────────────────────────
    _drift_monitor = DriftMonitor(engine, drift_threshold=settings.drift_threshold)

    # ── MLflow connectivity ───────────────────────────────────────────────────
    _mlflow_ok = _check_mlflow()
    logger.info("MLflow connected: %s", _mlflow_ok)

    logger.info("CartSense API startup complete.")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("CartSense API shutting down...")
    engine.dispose()


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CartSense Recommendation API",
    description="Drift-Resilient Hybrid Recommendation Engine for Electronics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_engine() -> HybridRecommendationEngine:
    if _hybrid_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation engine not loaded. Run generate_data.py → train_tfidf.py → train_als.py first.",
        )
    return _hybrid_engine


def _log_prediction(
    request_id: str,
    user_id: str,
    product_id: str,
    mode: str,
    model_version: str,
    top_k: int,
    latency_ms: float,
) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO prediction_logs
                      (request_id, user_id, product_id, recommendation_mode,
                       model_version, top_k, latency_ms, timestamp)
                    VALUES
                      (:rid, :uid, :pid, :mode, :mv, :k, :lat, :ts)
                """),
                {
                    "rid": request_id,
                    "uid": user_id,
                    "pid": product_id,
                    "mode": mode,
                    "mv": model_version,
                    "k": top_k,
                    "lat": latency_ms,
                    "ts": datetime.utcnow(),
                },
            )
    except Exception as exc:
        logger.warning("Prediction log failed (non-fatal): %s", exc)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthCheck, tags=["System"])
def health_check() -> HealthCheck:
    """Check health of all system components."""
    db_ok = check_database_health()
    arts = model_manager.artifacts
    return HealthCheck(
        status="healthy" if db_ok else "degraded",
        database=db_ok,
        models_loaded=arts.tfidf_loaded,
        tfidf_loaded=arts.tfidf_loaded,
        als_loaded=arts.als_loaded,
        mlflow=_mlflow_ok,
        products_count=len(_products_cache),
        version="1.0.0",
    )


@app.get("/api/products", response_model=list[ProductSummary], tags=["Catalog"])
def list_products(
    category: str | None = Query(None, description="Filter: laptop | smartphone"),
    search: str | None = Query(None, description="Search in product name/spec"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[ProductSummary]:
    """List products from the catalog with optional filtering."""
    if _products_cache.empty:
        raise HTTPException(status_code=503, detail="Product catalog not loaded.")

    df = _products_cache.copy()

    if category:
        df = df[df["category"].str.lower() == category.lower()]
    if search:
        mask = (
            df["name"].str.contains(search, case=False, na=False)
            | df["spec_text"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    df = df.iloc[offset : offset + limit]

    result = []
    for _, row in df.iterrows():
        result.append(
            ProductSummary(
                product_id=str(row["product_id"]),
                category=str(row["category"]),
                brand=str(row["brand"]),
                name=str(row["name"]),
                price_usd=float(row["price_usd"]),
                ram_gb=row.get("ram_gb"),
                storage_gb=row.get("storage_gb"),
                display_type=row.get("display_type"),
                refresh_rate_hz=row.get("refresh_rate_hz"),
                interaction_count=int(row.get("interaction_count", 0)),
            )
        )
    return result


@app.get("/api/products/{product_id}", response_model=ProductSpec, tags=["Catalog"])
def get_product(product_id: str) -> ProductSpec:
    """Get full details for a single product."""
    if _products_cache.empty:
        raise HTTPException(status_code=503, detail="Product catalog not loaded.")
    rows = _products_cache[_products_cache["product_id"] == product_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")
    row = rows.iloc[0].to_dict()
    return ProductSpec(
        product_id=str(row["product_id"]),
        category=str(row["category"]),
        brand=str(row["brand"]),
        name=str(row["name"]),
        description=row.get("description"),
        spec_text=str(row.get("spec_text", "")),
        processor=row.get("processor"),
        ram_gb=row.get("ram_gb"),
        storage_gb=row.get("storage_gb"),
        display_type=row.get("display_type"),
        refresh_rate_hz=row.get("refresh_rate_hz"),
        gpu=row.get("gpu"),
        battery_mah=row.get("battery_mah"),
        camera_mp=row.get("camera_mp"),
        chipset=row.get("chipset"),
        os=row.get("os"),
        price_usd=float(row["price_usd"]),
        interaction_count=int(row.get("interaction_count", 0)),
    )


@app.get("/api/users/{user_id}", response_model=UserProfile, tags=["Users"])
def get_user(user_id: str) -> UserProfile:
    """Get user profile and interaction count."""
    try:
        with engine.begin() as conn:
            user_df = pd.read_sql(
                text("SELECT * FROM users WHERE user_id = :uid LIMIT 1"),
                conn,
                params={"uid": user_id},
            )
            count = conn.execute(
                text("SELECT COUNT(*) FROM interactions WHERE user_id = :uid"),
                {"uid": user_id},
            ).scalar_one()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if user_df.empty:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")

    row = user_df.iloc[0].to_dict()
    return UserProfile(
        user_id=str(row["user_id"]),
        segment=row.get("segment"),
        interaction_count=int(count),
        created_at=row.get("created_at"),
    )


@app.get("/api/users", tags=["Users"])
def list_users(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    """List all users."""
    try:
        with engine.begin() as conn:
            df = pd.read_sql(
                text("SELECT user_id, segment, created_at FROM users ORDER BY user_id LIMIT :lim OFFSET :off"),
                conn,
                params={"lim": limit, "off": offset},
            )
        return df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
def recommend(req: RecommendationRequest) -> RecommendationResponse:
    """
    Get hybrid recommendations.

    Routing logic:
    - item interaction_count == 0  → cold_start (TF-IDF + Spec only)
    - user interaction_count == 0  → new_user_fallback (TF-IDF + Spec)
    - otherwise                    → hybrid (ALS + TF-IDF + Spec)
    """
    engine_instance = _require_engine()
    request_id = str(uuid.uuid4())

    # ── Validate product exists ───────────────────────────────────────────────
    known_pids = set(_products_cache["product_id"].astype(str).tolist())
    if req.product_id not in known_pids:
        raise HTTPException(status_code=404, detail=f"Product '{req.product_id}' not found.")

    # ── Get interaction counts ────────────────────────────────────────────────
    try:
        item_interaction_count = int(scalar_sql_safe(
            "SELECT COUNT(*) FROM interactions WHERE product_id = :pid",
            {"pid": req.product_id},
        ) or 0)
        user_interaction_count = int(scalar_sql_safe(
            "SELECT COUNT(*) FROM interactions WHERE user_id = :uid",
            {"uid": req.user_id},
        ) or 0)
    except Exception as exc:
        logger.error("DB query failed for interaction counts: %s", exc)
        item_interaction_count = -1
        user_interaction_count = -1

    # ── Run recommendation ────────────────────────────────────────────────────
    try:
        result = engine_instance.recommend(
            user_id=req.user_id,
            product_id=req.product_id,
            required_specs=req.required_specs or "",
            top_k=req.top_k,
            interaction_count=item_interaction_count,
            user_interaction_count=user_interaction_count,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Recommendation engine error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Recommendation engine error: {exc}") from exc

    # ── Log prediction ────────────────────────────────────────────────────────
    _log_prediction(
        request_id=request_id,
        user_id=req.user_id,
        product_id=req.product_id,
        mode=result.recommendation_mode,
        model_version=result.model_version,
        top_k=req.top_k,
        latency_ms=result.latency_ms,
    )

    # ── Build response ────────────────────────────────────────────────────────
    from backend.schemas import RecommendationItemResponse

    recommendations = [
        RecommendationItemResponse(
            product_id=item.product_id,
            name=item.name,
            category=item.category,
            brand=item.brand,
            price_usd=item.price_usd,
            spec_text=item.spec_text,
            als_score=item.als_score,
            tfidf_score=item.tfidf_score,
            spec_match_score=item.spec_match_score,
            final_score=item.final_score,
            reason=item.reason,
            attribute_scores=item.attribute_scores,
        )
        for item in result.recommendations
    ]

    return RecommendationResponse(
        request_id=request_id,
        user_id=result.user_id,
        product_id=result.product_id,
        recommendation_mode=result.recommendation_mode,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
        als_used=result.als_used,
        tfidf_used=result.tfidf_used,
        spec_matching_used=result.spec_matching_used,
        recommendations=recommendations,
        explanation=result.explanation,
    )


@app.get("/api/drift", tags=["MLOps"])
def get_drift_status():
    """Get the latest drift check result from the database."""
    if _drift_monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not initialized.")
    latest = _drift_monitor.get_latest_drift()
    if latest is None:
        return {
            "message": "No drift checks run yet. POST /api/drift/check to run one.",
            "dataset_drift_detected": False,
            "share_drifted_columns": 0.0,
        }
    return latest


@app.post("/api/drift/check", response_model=DriftMetricsResponse, tags=["MLOps"])
def check_drift(req: DriftCheckRequest) -> DriftMetricsResponse:
    """Run a drift analysis comparing training vs production distributions."""
    if _drift_monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor not initialized.")

    try:
        result = _drift_monitor.check_drift(
            simulate=req.simulate_drift,
            n_reference_days=req.n_reference_days,
            n_current_days=req.n_current_days,
        )
    except Exception as exc:
        logger.exception("Drift check failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Drift check failed: {exc}") from exc

    from backend.schemas import ColumnDriftResult

    col_results = [ColumnDriftResult(**c) for c in result.get("column_results", [])]

    return DriftMetricsResponse(
        run_id=result["run_id"],
        dataset_drift_detected=result["dataset_drift_detected"],
        share_drifted_columns=result["share_drifted_columns"],
        n_drifted_columns=result["n_drifted_columns"],
        n_total_columns=result["n_total_columns"],
        drift_threshold=result["drift_threshold"],
        retraining_recommended=result["retraining_recommended"],
        column_results=col_results,
        timestamp=result["timestamp"],
        simulation_mode=result.get("simulation_mode", False),
    )


@app.post("/api/retrain", response_model=RetrainingStatus, tags=["MLOps"])
def trigger_retrain(req: RetrainRequest) -> RetrainingStatus:
    """Trigger the full retraining pipeline."""
    try:
        process = subprocess.Popen(
            [sys.executable, "training/retrain_pipeline.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info("Retraining process started (PID %d).", process.pid)
        return RetrainingStatus(
            status="started",
            message=f"Retraining pipeline started (reason: {req.reason}). Check /api/model/status for updates.",
            timestamp=datetime.utcnow(),
        )
    except Exception as exc:
        logger.exception("Failed to start retraining: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {exc}") from exc


@app.post("/api/model/reload", tags=["MLOps"])
def reload_model():
    """Hot-reload model artifacts after retraining (called by retrain_pipeline)."""
    global _hybrid_engine
    success = model_manager.reload(settings.artifacts_dir)
    if success:
        _hybrid_engine = _build_hybrid_engine()
        logger.info("Model hot-reloaded successfully.")
        return {"status": "reloaded", "model_version": model_manager.artifacts.model_version}
    raise HTTPException(status_code=500, detail="Model reload failed — artifacts may be incomplete.")


@app.get("/api/model/status", response_model=ModelStatus, tags=["MLOps"])
def get_model_status() -> ModelStatus:
    """Return current loaded model information."""
    status = model_manager.get_status()
    return ModelStatus(**status)


@app.get("/api/training/history", response_model=list[TrainingRunSummary], tags=["MLOps"])
def get_training_history(limit: int = Query(20, ge=1, le=100)):
    """Return historical training run records."""
    try:
        with engine.begin() as conn:
            df = pd.read_sql(
                text("""
                    SELECT id, run_id, model_type, model_version, rmse,
                           precision_at_k, recall_at_k, ndcg_at_k, status, promoted, timestamp
                    FROM training_runs
                    ORDER BY timestamp DESC LIMIT :lim
                """),
                conn,
                params={"lim": limit},
            )
        return [
            TrainingRunSummary(
                id=int(row["id"]),
                run_id=row.get("run_id"),
                model_type=str(row["model_type"]),
                model_version=row.get("model_version"),
                rmse=row.get("rmse"),
                precision_at_k=row.get("precision_at_k"),
                recall_at_k=row.get("recall_at_k"),
                ndcg_at_k=row.get("ndcg_at_k"),
                status=str(row.get("status", "unknown")),
                promoted=bool(row.get("promoted", False)),
                timestamp=row["timestamp"],
            )
            for _, row in df.iterrows()
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/stats", tags=["System"])
def get_system_stats():
    """Return summary statistics for the dashboard."""
    try:
        with engine.begin() as conn:
            n_products = conn.execute(text("SELECT COUNT(*) FROM products")).scalar_one()
            n_users = conn.execute(text("SELECT COUNT(*) FROM users")).scalar_one()
            n_interactions = conn.execute(text("SELECT COUNT(*) FROM interactions")).scalar_one()
            n_cold_start = conn.execute(
                text("SELECT COUNT(*) FROM products WHERE interaction_count = 0")
            ).scalar_one()
            n_laptops = conn.execute(
                text("SELECT COUNT(*) FROM products WHERE category='laptop'")
            ).scalar_one()
            n_phones = conn.execute(
                text("SELECT COUNT(*) FROM products WHERE category='smartphone'")
            ).scalar_one()
        return {
            "n_products": int(n_products),
            "n_users": int(n_users),
            "n_interactions": int(n_interactions),
            "n_cold_start_products": int(n_cold_start),
            "n_laptops": int(n_laptops),
            "n_smartphones": int(n_phones),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/mlflow/experiments", tags=["MLOps"])
def get_mlflow_experiments():
    """List MLflow experiments."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        experiments = mlflow.search_experiments()
        return [
            {
                "experiment_id": e.experiment_id,
                "name": e.name,
                "lifecycle_stage": e.lifecycle_stage,
            }
            for e in experiments
        ]
    except Exception as exc:
        return {"error": str(exc), "experiments": []}


@app.get("/api/mlflow/runs", tags=["MLOps"])
def get_mlflow_runs(experiment_name: str = "CartSense_Hybrid_Recommendation", limit: int = 10):
    """Get recent MLflow runs for an experiment."""
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return {"runs": [], "message": "Experiment not found"}
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"],
        )
        return runs[["run_id", "status", "start_time", "metrics.tfidf_rmse", "metrics.als_rmse"]].to_dict(orient="records") if not runs.empty else {"runs": []}
    except Exception as exc:
        return {"error": str(exc), "runs": []}
