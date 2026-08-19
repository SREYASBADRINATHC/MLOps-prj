import logging
import os
import re
import tempfile
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.spark
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
from pyspark.sql import functions as spark_functions
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

logger = logging.getLogger("cartsense-api")
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@localhost:5432/cartsense",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ALS_MODEL_NAME = os.getenv("ALS_MODEL_NAME", "CartSense_ALS_Model")
TFIDF_MODEL_NAME = os.getenv("TFIDF_MODEL_NAME", "CartSense_TFIDF_Model")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=5,
    max_overflow=10,
)

app = FastAPI(title="CartSense Recommendation API", version="1.0.0")

spark_session: SparkSession | None = None
als_model = None
tfidf_vectorizer = None
products_df_cache = pd.DataFrame()


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., ge=1)
    product_id: int = Field(..., ge=1)
    required_specs: str = ""


def configure_windows_spark_runtime() -> None:
    """Set Windows-safe Spark/Hadoop temp settings to avoid Spark startup hangs."""
    os.environ.setdefault("PYSPARK_PYTHON", os.environ.get("PYTHON", "python"))
    os.environ.setdefault("HADOOP_HOME", tempfile.gettempdir())
    os.environ.setdefault("SPARK_LOCAL_DIRS", os.path.join(tempfile.gettempdir(), "spark-local"))
    os.makedirs(os.environ["SPARK_LOCAL_DIRS"], exist_ok=True)


def read_sql_with_retry(sql_query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    """Read SQL with one retry after invalid/stale connection errors."""
    try:
        with engine.begin() as connection:
            return pd.read_sql(text(sql_query), con=connection, params=params or {})
    except OperationalError:
        engine.dispose()
        try:
            with engine.begin() as connection:
                return pd.read_sql(text(sql_query), con=connection, params=params or {})
        except SQLAlchemyError as retry_exc:
            raise RuntimeError(f"Retry failed for query '{sql_query}': {retry_exc}") from retry_exc
    except SQLAlchemyError as exc:
        raise RuntimeError(f"SQL read failed for query '{sql_query}': {exc}") from exc


def scalar_sql_with_retry(sql_query: str, params: dict[str, Any] | None = None) -> int:
    """Fetch scalar value with retry to handle dropped DB connections gracefully."""
    statement = text(sql_query)
    try:
        with engine.begin() as connection:
            value = connection.execute(statement, params or {}).scalar_one()
            return int(value)
    except OperationalError:
        engine.dispose()
        try:
            with engine.begin() as connection:
                value = connection.execute(statement, params or {}).scalar_one()
                return int(value)
        except SQLAlchemyError as retry_exc:
            raise RuntimeError(f"Retry failed for scalar query '{sql_query}': {retry_exc}") from retry_exc
    except SQLAlchemyError as exc:
        raise RuntimeError(f"Scalar query failed for '{sql_query}': {exc}") from exc


def load_products_cache() -> pd.DataFrame:
    products_df = read_sql_with_retry("SELECT * FROM products ORDER BY product_id")
    if products_df.empty:
        raise RuntimeError("Products table is empty. Ingest data first.")
    return products_df


def load_registry_models() -> None:
    """Load ALS and TF-IDF artifacts from MLflow Model Registry."""
    global als_model, tfidf_vectorizer
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        als_model = mlflow.spark.load_model(f"models:/{ALS_MODEL_NAME}/latest")
        logger.info("ALS model loaded from MLflow registry.")
    except Exception as exc:
        als_model = None
        logger.warning("ALS model unavailable, hybrid mode will skip ALS path: %s", exc)

    try:
        tfidf_vectorizer = mlflow.sklearn.load_model(f"models:/{TFIDF_MODEL_NAME}/latest")
        logger.info("TF-IDF model loaded from MLflow registry.")
    except Exception as exc:
        raise RuntimeError(f"TF-IDF model must be available. Loading failed: {exc}") from exc


def _required_specs_boost(scores: np.ndarray, specs_series: pd.Series, required_specs: str) -> np.ndarray:
    """Boost base cosine similarity when required specs terms occur in candidate specs."""
    terms = [token for token in re.split(r"[,\s]+", required_specs.lower().strip()) if token]
    if not terms:
        return scores

    boosts = np.zeros_like(scores)
    lowered_specs = specs_series.fillna("").str.lower().tolist()
    for idx, value in enumerate(lowered_specs):
        match_count = sum(1 for term in terms if term in value)
        boosts[idx] = min(match_count * 0.12, 0.30)
    return np.clip(scores + boosts, 0.0, 1.0)


def compute_tfidf_scores(anchor_product_id: int, required_specs: str) -> dict[int, float]:
    if tfidf_vectorizer is None:
        raise RuntimeError("TF-IDF vectorizer is not loaded.")

    local_df = products_df_cache.copy().reset_index(drop=True)
    product_ids = local_df["product_id"].astype(int).tolist()
    if anchor_product_id not in product_ids:
        raise ValueError(f"Unknown product_id={anchor_product_id}.")

    specs_text = local_df["specs_text"].fillna("")
    tfidf_matrix = tfidf_vectorizer.transform(specs_text)
    product_index = {pid: idx for idx, pid in enumerate(product_ids)}
    anchor_index = product_index[anchor_product_id]

    base_scores = cosine_similarity(tfidf_matrix[anchor_index], tfidf_matrix).flatten()
    base_scores[anchor_index] = 0.0

    final_scores = _required_specs_boost(base_scores, specs_text, required_specs)
    return {product_ids[idx]: float(final_scores[idx]) for idx in range(len(product_ids))}


def compute_als_scores(user_id: int) -> dict[int, float]:
    if als_model is None:
        return {}
    if spark_session is None:
        raise RuntimeError("Spark session is unavailable.")

    product_ids = products_df_cache["product_id"].astype(int).tolist()
    scoring_df = spark_session.createDataFrame([(int(user_id), int(pid)) for pid in product_ids], ["user_id", "product_id"])
    prediction_df = als_model.transform(scoring_df).select("product_id", "prediction")
    prediction_df = prediction_df.withColumn("prediction", spark_functions.coalesce("prediction", spark_functions.lit(0.0)))
    prediction_pdf = prediction_df.toPandas()
    if prediction_pdf.empty:
        return {}

    min_prediction = float(prediction_pdf["prediction"].min())
    max_prediction = float(prediction_pdf["prediction"].max())
    if max_prediction == min_prediction:
        prediction_pdf["normalized_prediction"] = 0.5
    else:
        prediction_pdf["normalized_prediction"] = (
            (prediction_pdf["prediction"] - min_prediction) / (max_prediction - min_prediction)
        )
    return {
        int(row["product_id"]): float(row["normalized_prediction"])
        for _, row in prediction_pdf.iterrows()
    }


@app.on_event("startup")
def startup_event() -> None:
    global spark_session, products_df_cache
    configure_windows_spark_runtime()
    spark_session = (
        SparkSession.builder.appName("CartSense-FastAPI")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    products_df_cache = load_products_cache()
    load_registry_models()


@app.on_event("shutdown")
def shutdown_event() -> None:
    if spark_session is not None:
        spark_session.stop()
    engine.dispose()


@app.get("/api/products")
def get_products() -> list[dict[str, Any]]:
    subset = products_df_cache[["product_id", "product_name", "category", "brand", "price_usd"]]
    return subset.to_dict(orient="records")


@app.post("/api/recommend")
def recommend_products(request: RecommendationRequest) -> list[dict[str, Any]]:
    known_product_ids = set(products_df_cache["product_id"].astype(int).tolist())
    if request.product_id not in known_product_ids:
        raise HTTPException(status_code=404, detail=f"product_id={request.product_id} not found.")

    try:
        interaction_count = scalar_sql_with_retry(
            "SELECT COUNT(*) FROM interactions WHERE product_id = :product_id",
            {"product_id": request.product_id},
        )
        tfidf_scores = compute_tfidf_scores(request.product_id, request.required_specs)
        route_label = "Routed via: TF-IDF Cold Start"
        final_scores = tfidf_scores

        # Routing rule:
        # - 0 interactions => hard TF-IDF route
        # - >0 interactions => blended Hybrid route
        if interaction_count > 0 and als_model is not None:
            als_scores = compute_als_scores(request.user_id)
            if als_scores:
                route_label = "Routed via: Hybrid"
                final_scores = {}
                for product_id, tfidf_score in tfidf_scores.items():
                    final_scores[product_id] = (0.65 * als_scores.get(product_id, 0.0)) + (0.35 * tfidf_score)

        final_scores.pop(request.product_id, None)
        top_product_ids = sorted(final_scores.keys(), key=lambda pid: final_scores[pid], reverse=True)[:3]

        lookup_df = products_df_cache.set_index("product_id")
        response_payload: list[dict[str, Any]] = []
        for product_id in top_product_ids:
            response_payload.append(
                {
                    "product_id": int(product_id),
                    "product_name": str(lookup_df.loc[product_id, "product_name"]),
                    "match_confidence": round(float(final_scores[product_id]) * 100.0, 2),
                    "algorithm_route": route_label,
                }
            )
        return response_payload
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/drift")
def get_mock_drift_metrics() -> dict[str, Any]:
    """Generate mock drift metrics using Evidently so frontend can visualize MLOps health."""
    try:
        reference_df = read_sql_with_retry("SELECT price_usd, ram_gb, storage_gb FROM products ORDER BY product_id")
        current_df = reference_df.copy()
        current_df["price_usd"] = current_df["price_usd"] * 1.05
        current_df["ram_gb"] = current_df["ram_gb"] + np.where(current_df.index % 2 == 0, 0, 2)

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report_payload = report.as_dict()
        result = report_payload["metrics"][0]["result"]
        return {
            "dataset_drift_detected": bool(result["dataset_drift"]),
            "share_of_drifted_columns": round(float(result["share_of_drifted_columns"]) * 100.0, 2),
            "drifted_columns_count": int(result["number_of_drifted_columns"]),
            "total_columns_count": int(result["number_of_columns"]),
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
