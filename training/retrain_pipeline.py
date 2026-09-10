"""
CartSense — Continuous Training / Retraining Pipeline

Workflow:
  1. Load latest data from PostgreSQL
  2. Validate data quality (minimum rows, no null anchors)
  3. Train TF-IDF from scratch
  4. Train PySpark ALS from scratch
  5. Evaluate all models (temporal split)
  6. Log to MLflow
  7. Compare against production model (quality gate)
     - RMSE must be ≤ production RMSE × 1.05 (5% tolerance)
     - OR no production model exists yet
  8. If passes: save artifacts + register in MLflow + notify FastAPI reload
  9. If fails: reject, keep production model
 10. Record training run in PostgreSQL training_runs table

Run directly:
  python training/retrain_pipeline.py

Or triggered via API:
  POST /api/retrain
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine, text

# Ensure project root is on path when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain_pipeline")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@postgres_db:5432/cartsense",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
API_BASE_URL = os.getenv("BACKEND_URL", "http://fastapi_backend:8000")
TFIDF_MODEL_NAME = os.getenv("TFIDF_MODEL_NAME", "CartSense_TFIDF")
ALS_MODEL_NAME = os.getenv("ALS_MODEL_NAME", "CartSense_ALS")
ALS_RANK = int(os.getenv("ALS_RANK", "20"))
ALS_MAX_ITER = int(os.getenv("ALS_MAX_ITER", "15"))
ALS_REG_PARAM = float(os.getenv("ALS_REG_PARAM", "0.1"))
ALS_SEED = int(os.getenv("ALS_SEED", "42"))


def get_engine():
    is_sqlite = DATABASE_URL.startswith("sqlite")
    kwargs = {"pool_pre_ping": True, "pool_recycle": 1800}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 3
        kwargs["max_overflow"] = 5
    return create_engine(DATABASE_URL, **kwargs)


# ── Step 1: Load Data ─────────────────────────────────────────────────────────

def load_data(eng) -> tuple[pd.DataFrame, pd.DataFrame]:
    with eng.begin() as conn:
        products_df = pd.read_sql(text("SELECT * FROM products ORDER BY product_id"), conn)
        interactions_df = pd.read_sql(
            text("SELECT * FROM interactions ORDER BY timestamp"), conn
        )
    logger.info("Loaded %d products, %d interactions.", len(products_df), len(interactions_df))
    return products_df, interactions_df


# ── Step 2: Validate ──────────────────────────────────────────────────────────

def validate_data(products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> None:
    if len(products_df) < 10:
        raise ValueError(f"Too few products ({len(products_df)}). Need at least 10.")
    if len(interactions_df) < 50:
        raise ValueError(f"Too few interactions ({len(interactions_df)}). Need at least 50.")
    null_products = products_df["product_id"].isna().sum()
    if null_products > 0:
        raise ValueError(f"{null_products} products with null product_id.")
    logger.info("Data validation passed.")


# ── Step 3 & 4: Train TF-IDF ─────────────────────────────────────────────────

def train_tfidf_artifacts(products_df: pd.DataFrame, interactions_df: pd.DataFrame) -> tuple:
    """Train TF-IDF and return (vectorizer, tfidf_matrix, product_ids, metrics)."""
    import math
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics import mean_squared_error
    import joblib
    from scipy import sparse

    spec_texts = products_df["spec_text"].fillna("").tolist()
    product_ids = products_df["product_id"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True,
        strip_accents="unicode", analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(spec_texts)
    logger.info("TF-IDF fitted: vocab=%d, docs=%d", len(vectorizer.vocabulary_), len(product_ids))

    # Quick evaluation
    from training.evaluate import evaluate_tfidf
    
    interactions_df = interactions_df.copy()
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    train_cut = int(len(interactions_df) * 0.80)
    train_df = interactions_df.iloc[:train_cut]
    test_df = interactions_df.iloc[train_cut:]

    metrics = evaluate_tfidf(
        train_df, test_df, products_df,
        vectorizer, tfidf_matrix, product_ids, k=5
    )

    return vectorizer, tfidf_matrix, product_ids, metrics


# ── Step 4: Train ALS ─────────────────────────────────────────────────────────

def train_als_artifacts(interactions_df: pd.DataFrame, products_df: pd.DataFrame) -> tuple:
    """Run PySpark ALS and return (user_factors, item_factors, maps, metrics)."""
    from training.train_als import (
        build_integer_maps,
        temporal_split,
        train_als_spark,
        evaluate_als,
        save_als_artifacts,
    )

    df, user_to_int, int_to_user, item_to_int, int_to_item = build_integer_maps(
        interactions_df, products_df
    )
    train_df, test_df = temporal_split(df, 0.80)
    model, user_factors, item_factors = train_als_spark(
        train_df, rank=ALS_RANK, max_iter=ALS_MAX_ITER, reg_param=ALS_REG_PARAM, seed=ALS_SEED
    )
    metrics = evaluate_als(test_df, user_factors, item_factors, int_to_item, K=5)
    return model, user_factors, item_factors, user_to_int, int_to_user, item_to_int, int_to_item, metrics


# ── Step 5: Quality Gate ──────────────────────────────────────────────────────

def quality_gate(new_tfidf_metrics: dict, new_als_metrics: dict) -> tuple[bool, str]:
    """
    Compare new models against MLflow @champion models.
    Pass condition:
    - Both TF-IDF and ALS Precision@5 >= 0.01
    - Both TF-IDF and ALS Precision@5 >= max(production Precision@5 * 0.95, 0.01)
    """
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1. Evaluate TF-IDF
    new_tfidf_prec = new_tfidf_metrics.get("precision_at_5", new_tfidf_metrics.get("precision_at_k", 0.0))
    if new_tfidf_prec < 0.01:
        return False, f"New TF-IDF Precision@5 is too low ({new_tfidf_prec:.4f} < 0.01) — REJECTED."

    try:
        tfidf_champ = client.get_model_version_by_alias(TFIDF_MODEL_NAME, "champion")
        tfidf_run = client.get_run(tfidf_champ.run_id)
        prod_tfidf_prec = float(tfidf_run.data.metrics.get("precision_at_5", tfidf_run.data.metrics.get("precision_at_k", 0.0)))
        if prod_tfidf_prec > 0.0 and new_tfidf_prec < prod_tfidf_prec * 0.95:
            return False, f"New TF-IDF Precision@5 {new_tfidf_prec:.4f} degraded vs prod {prod_tfidf_prec:.4f} — REJECTED."
    except Exception as e:
        logger.info("No champion TF-IDF found in MLflow or error fetching: %s", e)

    # 2. Evaluate ALS
    if new_als_metrics:
        new_als_prec = new_als_metrics.get("precision_at_5", new_als_metrics.get("precision_at_k", 0.0))
        if new_als_prec < 0.01:
            return False, f"New ALS Precision@5 is too low ({new_als_prec:.4f} < 0.01) — REJECTED."

        try:
            als_champ = client.get_model_version_by_alias(ALS_MODEL_NAME, "champion")
            als_run = client.get_run(als_champ.run_id)
            prod_als_prec = float(als_run.data.metrics.get("precision_at_5", als_run.data.metrics.get("precision_at_k", 0.0)))
            if prod_als_prec > 0.0 and new_als_prec < prod_als_prec * 0.95:
                return False, f"New ALS Precision@5 {new_als_prec:.4f} degraded vs prod {prod_als_prec:.4f} — REJECTED."
        except Exception as e:
            logger.info("No champion ALS found in MLflow or error fetching: %s", e)

    return True, "All models passed the quality gate — PROMOTED."


# ── Step 6: Save & Log ────────────────────────────────────────────────────────

def save_tfidf_artifacts(vectorizer, tfidf_matrix, product_ids, metadata, artifacts_dir):
    import joblib
    from scipy import sparse
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, f"{artifacts_dir}/tfidf_vectorizer.joblib")
    sparse.save_npz(f"{artifacts_dir}/tfidf_matrix.npz", tfidf_matrix)
    with open(f"{artifacts_dir}/product_id_map.json", "w") as f:
        json.dump({"index_to_product_id": product_ids}, f)
    with open(f"{artifacts_dir}/tfidf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def log_run_to_db(eng, run_id, model_type, model_version, metrics, status, promoted):
    try:
        with eng.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO training_runs
                      (run_id, model_type, model_version, rmse, precision_at_k,
                       recall_at_k, ndcg_at_k, status, promoted, timestamp)
                    VALUES
                      (:run_id, :mt, :mv, :rmse, :p5, :r5, :n5, :status, :promoted, :ts)
                """),
                {
                    "run_id": run_id,
                    "mt": model_type,
                    "mv": model_version,
                    "rmse": metrics.get("rmse"),
                    "p5": metrics.get("precision_at_5"),
                    "r5": metrics.get("recall_at_5"),
                    "n5": metrics.get("ndcg_at_5"),
                    "status": status,
                    "promoted": promoted,
                    "ts": datetime.utcnow(),
                },
            )
    except Exception as exc:
        logger.warning("Could not save training run to DB: %s", exc)


def notify_api_reload():
    """Tell FastAPI to hot-reload model artifacts."""
    try:
        resp = requests.post(f"{API_BASE_URL}/api/model/reload", timeout=30)
        if resp.status_code == 200:
            logger.info("FastAPI model reload successful.")
        else:
            logger.warning("FastAPI reload returned HTTP %d.", resp.status_code)
    except Exception as exc:
        logger.warning("Could not notify FastAPI for reload: %s", exc)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    run_id = str(uuid.uuid4())[:12]
    model_version = f"v_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    logger.info("=== CartSense Retraining Pipeline | Run %s ===", run_id)

    eng = get_engine()
    t0 = time.time()

    # 1. Load data
    products_df, interactions_df = load_data(eng)

    # 2. Validate
    try:
        validate_data(products_df, interactions_df)
    except ValueError as exc:
        logger.error("Data validation failed: %s", exc)
        log_run_to_db(eng, run_id, "hybrid", model_version, {}, "failed", False)
        sys.exit(1)

    # 3. Train TF-IDF
    logger.info("Training TF-IDF...")
    vectorizer, tfidf_matrix, product_ids, tfidf_metrics = train_tfidf_artifacts(
        products_df, interactions_df
    )

    # 4. Train ALS (best-effort; skip if Spark unavailable)
    als_metrics = {}
    user_factors = item_factors = als_model = None
    user_to_int = int_to_user = item_to_int = int_to_item = {}
    try:
        logger.info("Training PySpark ALS...")
        als_model, user_factors, item_factors, user_to_int, int_to_user, item_to_int, int_to_item, als_metrics = (
            train_als_artifacts(interactions_df, products_df)
        )
    except Exception as exc:
        logger.warning("ALS training failed (will use TF-IDF only): %s", exc)

    # 5. Quality gate
    combined_metrics = {**tfidf_metrics, **als_metrics}
    passed, gate_msg = quality_gate(tfidf_metrics, als_metrics)
    logger.info("Quality gate: %s — %s", "PASS" if passed else "FAIL", gate_msg)

    # 6. MLflow logging
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.search_experiments()
    except Exception as e:
        logger.error("MLflow URI %s unreachable. Failing the run. %s", MLFLOW_TRACKING_URI, e)
        raise RuntimeError(f"Cannot connect to MLflow at {MLFLOW_TRACKING_URI}") from e

    mlflow.set_experiment("CartSense_Hybrid_Recommendation")
    with mlflow.start_run(run_name=f"retrain_{model_version}") as run:
        mlflow.log_params({
            "run_id": run_id,
            "model_version": model_version,
            "als_rank": ALS_RANK,
            "als_max_iter": ALS_MAX_ITER,
            "als_reg_param": ALS_REG_PARAM,
            "tfidf_vocab": len(vectorizer.vocabulary_),
            "quality_gate": "pass" if passed else "fail",
            "n_products": len(products_df),
            "n_interactions": len(interactions_df),
        })
        mlflow.log_metrics({k: v for k, v in combined_metrics.items() if isinstance(v, float) and not (v != v)})
        mlflow_run_id = run.info.run_id
        
        # Only register models if quality gate passed
        if passed:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            
            # Register TF-IDF
            tfidf_model_info = mlflow.sklearn.log_model(
                sk_model=vectorizer,
                artifact_path="tfidf_vectorizer",
                registered_model_name=TFIDF_MODEL_NAME
            )
            client.set_registered_model_alias(TFIDF_MODEL_NAME, "champion", tfidf_model_info.registered_model_version)
            logger.info("TFIDF model registered and set as @champion version %s", tfidf_model_info.registered_model_version)
            
            # Register ALS
            if als_model is not None:
                als_model_info = mlflow.spark.log_model(
                    spark_model=als_model,
                    artifact_path="als_model",
                    registered_model_name=ALS_MODEL_NAME
                )
                client.set_registered_model_alias(ALS_MODEL_NAME, "champion", als_model_info.registered_model_version)
                logger.info("ALS model registered and set as @champion version %s", als_model_info.registered_model_version)

    if not passed:
        logger.warning("New model REJECTED. Production model unchanged.")
        log_run_to_db(eng, run_id, "hybrid", model_version, combined_metrics, "rejected", False)
        print(f"❌ Retraining REJECTED: {gate_msg}")
        sys.exit(0)

    # 7. Save artifacts
    logger.info("Saving artifacts for model version %s...", model_version)
    tfidf_metadata = {
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_documents": len(product_ids),
        "training_timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "mlflow_run_id": mlflow_run_id,
        "metrics": tfidf_metrics,
    }
    save_tfidf_artifacts(vectorizer, tfidf_matrix, product_ids, tfidf_metadata, ARTIFACTS_DIR)

    if user_factors is not None and item_factors is not None:
        from training.train_als import save_als_artifacts
        als_metadata = {
            "rank": ALS_RANK, "max_iter": ALS_MAX_ITER, "reg_param": ALS_REG_PARAM,
            "seed": ALS_SEED, "training_timestamp": datetime.utcnow().isoformat(),
            "model_version": model_version, "metrics": als_metrics,
            "n_users": len(user_to_int), "n_items": len(item_to_int),
        }
        save_als_artifacts(
            user_factors, item_factors,
            user_to_int, int_to_user, item_to_int, int_to_item, als_metadata,
        )

    # 8. Record in DB
    log_run_to_db(eng, run_id, "hybrid", model_version, combined_metrics, "success", True)

    # 9. Notify FastAPI
    notify_api_reload()

    elapsed = time.time() - t0
    print(f"✅ Retraining complete in {elapsed:.1f}s")
    print(f"   Model version : {model_version}")
    print(f"   TF-IDF RMSE  : {tfidf_metrics.get('rmse', 'N/A')}")
    print(f"   ALS RMSE     : {als_metrics.get('rmse', 'N/A')}")
    print(f"   Quality gate : PASS — {gate_msg}")
    print(f"   MLflow run   : {mlflow_run_id}")


if __name__ == "__main__":
    main()
