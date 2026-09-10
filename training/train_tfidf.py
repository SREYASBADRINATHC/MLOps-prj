"""
CartSense — TF-IDF Content-Based Model Training

Fits a TfidfVectorizer EXCLUSIVELY on our own electronics product corpus.
No pretrained embeddings or external models.

Outputs (saved to artifacts/):
  - tfidf_vectorizer.joblib   : fitted TfidfVectorizer
  - tfidf_matrix.npz          : sparse TF-IDF matrix (scipy CSR format)
  - product_id_map.json       : {index -> product_id} mapping
  - tfidf_metadata.json       : training metadata

Also registers the vectorizer in the MLflow model registry.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_tfidf")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@postgres_db:5432/cartsense",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
TFIDF_MODEL_NAME = os.getenv("TFIDF_MODEL_NAME", "CartSense_TFIDF")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")


def get_engine():
    is_sqlite = DATABASE_URL.startswith("sqlite")
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 1800}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 3
        kwargs["max_overflow"] = 5
    return create_engine(DATABASE_URL, **kwargs)


def load_products(eng) -> pd.DataFrame:
    with eng.begin() as conn:
        df = pd.read_sql(text("SELECT product_id, category, spec_text FROM products ORDER BY product_id"), conn)
    if df.empty:
        raise ValueError("Products table is empty. Run generate_data.py first.")
    logger.info("Loaded %d products for TF-IDF training.", len(df))
    return df


def load_interactions(eng) -> pd.DataFrame:
    with eng.begin() as conn:
        df = pd.read_sql(
            text("SELECT user_id, product_id, weight, rating, timestamp FROM interactions ORDER BY timestamp"),
            conn,
        )
    logger.info("Loaded %d interactions.", len(df))
    return df


def train_tfidf(products_df: pd.DataFrame) -> tuple[TfidfVectorizer, Any, list[str]]:
    """
    Fit TfidfVectorizer on the product spec_text corpus.

    Parameters chosen:
    - ngram_range=(1,2): capture both single words and bigrams (e.g. "OLED display", "32GB RAM")
    - min_df=2: ignore terms appearing in only 1 product (avoids overfitting to unique model numbers)
    - max_df=0.95: ignore near-universal terms
    - sublinear_tf=True: apply log(1+tf) scaling to reduce dominance of common terms
    - analyzer='word': standard word tokenization
    """
    spec_texts = products_df["spec_text"].fillna("").tolist()
    product_ids = products_df["product_id"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-zA-Z0-9]+\b",
    )

    logger.info("Fitting TF-IDF on %d documents...", len(spec_texts))
    tfidf_matrix = vectorizer.fit_transform(spec_texts)
    logger.info(
        "Vocabulary size: %d | Matrix shape: %s",
        len(vectorizer.vocabulary_),
        tfidf_matrix.shape,
    )
    return vectorizer, tfidf_matrix, product_ids


def evaluate_tfidf(
    products_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: Any,
    product_ids: list[str],
) -> dict[str, float]:
    """
    Evaluate TF-IDF recommendation quality using a temporal split.

    Method:
    - Training set  : interactions in the earliest 80% of the time range
    - Test set      : interactions in the latest 20%
    - For each user-product pair in test set, rank all same-category products
      by cosine similarity to the target product. Check if top-K contain
      products the user also interacted with.
    """
    if interactions_df.empty:
        return {"precision_at_5": 0.0, "recall_at_5": 0.0, "ndcg_at_5": 0.0}

    K = 5
    interactions_df = interactions_df.copy()
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    interactions_df = interactions_df.sort_values("timestamp")

    cutoff_idx = int(len(interactions_df) * 0.80)
    train_ints = interactions_df.iloc[:cutoff_idx]
    test_ints = interactions_df.iloc[cutoff_idx:]

    product_id_to_idx: dict[str, int] = {pid: idx for idx, pid in enumerate(product_ids)}
    category_by_pid: dict[str, str] = dict(zip(products_df["product_id"], products_df["category"]))


    # Precision/Recall/NDCG @K
    precision_scores: list[float] = []
    recall_scores: list[float] = []
    ndcg_scores: list[float] = []

    test_users = test_ints["user_id"].unique()
    for uid in test_users[:200]:  # cap for speed
        test_products = set(test_ints[test_ints["user_id"] == uid]["product_id"].tolist())
        train_products = set(train_ints[train_ints["user_id"] == uid]["product_id"].tolist())
        if not train_products or not test_products:
            continue

        # Pick first train product as anchor
        anchor_pid = next(iter(train_products))
        if anchor_pid not in product_id_to_idx:
            continue
        anchor_idx = product_id_to_idx[anchor_pid]
        anchor_cat = category_by_pid.get(anchor_pid, "")

        # Get same-category products
        same_cat = [
            pid for pid in product_ids
            if category_by_pid.get(pid, "") == anchor_cat and pid != anchor_pid
        ]
        if not same_cat:
            continue
        same_cat_indices = [product_id_to_idx[pid] for pid in same_cat]
        sims = cosine_similarity(tfidf_matrix[anchor_idx], tfidf_matrix[same_cat_indices]).flatten()
        top_k_indices = np.argsort(sims)[::-1][:K]
        top_k_pids = [same_cat[i] for i in top_k_indices]

        hits = sum(1 for p in top_k_pids if p in test_products)
        precision_scores.append(hits / K)
        recall_scores.append(hits / max(len(test_products), 1))

        # NDCG
        dcg = sum(
            (1 if top_k_pids[i] in test_products else 0) / math.log2(i + 2)
            for i in range(len(top_k_pids))
        )
        ideal_hits = min(K, len(test_products))
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {
        "precision_at_5": round(float(np.mean(precision_scores)), 4) if precision_scores else 0.0,
        "recall_at_5": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
        "ndcg_at_5": round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else 0.0,
    }
    logger.info("TF-IDF evaluation: %s", metrics)
    return metrics


def save_artifacts(
    vectorizer: TfidfVectorizer,
    tfidf_matrix: Any,
    product_ids: list[str],
    metadata: dict,
) -> None:
    """Save all TF-IDF artifacts to the artifacts directory."""
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    vectorizer_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vectorizer_path)
    logger.info("Saved vectorizer → %s", vectorizer_path)

    matrix_path = os.path.join(ARTIFACTS_DIR, "tfidf_matrix.npz")
    sparse.save_npz(matrix_path, tfidf_matrix)
    logger.info("Saved TF-IDF matrix → %s", matrix_path)

    id_map_path = os.path.join(ARTIFACTS_DIR, "product_id_map.json")
    with open(id_map_path, "w") as f:
        json.dump({"index_to_product_id": product_ids}, f)
    logger.info("Saved product ID map → %s", id_map_path)

    meta_path = os.path.join(ARTIFACTS_DIR, "tfidf_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved metadata → %s", meta_path)


def log_to_mlflow(
    vectorizer: TfidfVectorizer,
    metrics: dict,
    metadata: dict,
) -> str:
    """Log TF-IDF model and metrics to MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.search_experiments()
        logger.info("MLflow connected at %s", MLFLOW_TRACKING_URI)
    except Exception as e:
        logger.error("MLflow URI %s unreachable. Failing the run. %s", MLFLOW_TRACKING_URI, e)
        raise RuntimeError(f"Cannot connect to MLflow at {MLFLOW_TRACKING_URI}") from e

    mlflow.set_experiment("CartSense_Hybrid_Recommendation")

    with mlflow.start_run(run_name="tfidf_training") as run:
        mlflow.log_params({
            "ngram_range": "(1,2)",
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
            "vocabulary_size": metadata["vocabulary_size"],
            "n_documents": metadata["n_documents"],
            "training_timestamp": metadata["training_timestamp"],
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="tfidf_vectorizer",
            registered_model_name=TFIDF_MODEL_NAME,
        )
        # Also log the matrix and map as artifacts
        mlflow.log_artifacts(ARTIFACTS_DIR, artifact_path="tfidf_artifacts")

        run_id = run.info.run_id
        logger.info("MLflow run ID: %s", run_id)
        return run_id


def main() -> None:
    t0 = time.time()
    eng = get_engine()

    products_df = load_products(eng)
    interactions_df = load_interactions(eng)

    vectorizer, tfidf_matrix, product_ids = train_tfidf(products_df)

    metrics = evaluate_tfidf(products_df, interactions_df, vectorizer, tfidf_matrix, product_ids)

    metadata = {
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_documents": len(product_ids),
        "training_timestamp": datetime.utcnow().isoformat(),
        "model_type": "TF-IDF",
        "ngram_range": "(1,2)",
        "min_df": 2,
        "metrics": metrics,
    }

    save_artifacts(vectorizer, tfidf_matrix, product_ids, metadata)
    run_id = log_to_mlflow(vectorizer, metrics, metadata)

    elapsed = time.time() - t0
    print(f"✅ TF-IDF training complete in {elapsed:.1f}s")
    print(f"   Vocabulary: {metadata['vocabulary_size']} terms")
    print(f"   Products  : {metadata['n_documents']}")
    print(f"   P@5       : {metrics['precision_at_5']:.4f}")
    print(f"   R@5       : {metrics['recall_at_5']:.4f}")
    print(f"   NDCG@5    : {metrics['ndcg_at_5']:.4f}")
    print(f"   MLflow Run: {run_id}")


if __name__ == "__main__":
    main()
