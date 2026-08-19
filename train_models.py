import math
import os
import tempfile
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import mlflow.spark
import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@localhost:5432/cartsense",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
ALS_MODEL_NAME = os.getenv("ALS_MODEL_NAME", "CartSense_ALS_Model")
TFIDF_MODEL_NAME = os.getenv("TFIDF_MODEL_NAME", "CartSense_TFIDF_Model")


def _configure_windows_spark_runtime() -> None:
    """Set Windows-safe Spark/Hadoop temp settings to avoid Spark startup hangs."""
    jdk_candidates = [
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot",
        r"C:\Program Files\Eclipse Adoptium\jdk-17.0.20-hotspot",
    ]
    for jdk_path in jdk_candidates:
        if os.path.isdir(jdk_path):
            os.environ.setdefault("JAVA_HOME", jdk_path)
            os.environ["PATH"] = f"{os.path.join(jdk_path, 'bin')};{os.environ.get('PATH', '')}"
            break

    os.environ.setdefault("PYSPARK_PYTHON", os.environ.get("PYTHON", "python"))
    os.environ.setdefault("HADOOP_HOME", tempfile.gettempdir())
    os.environ.setdefault("SPARK_LOCAL_DIRS", os.path.join(tempfile.gettempdir(), "spark-local"))
    os.makedirs(os.environ["SPARK_LOCAL_DIRS"], exist_ok=True)


@dataclass
class TrainArtifacts:
    products: pd.DataFrame
    interactions: pd.DataFrame


def read_table_with_retry(sql_query: str) -> pd.DataFrame:
    """Read a SQL query with one retry after a connection-related failure."""
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=1800)
    try:
        with engine.begin() as connection:
            return pd.read_sql(sql_query, connection)
    except OperationalError:
        engine.dispose()
        try:
            with engine.begin() as connection:
                return pd.read_sql(sql_query, connection)
        except SQLAlchemyError as retry_exc:
            raise RuntimeError(f"Retry failed while reading '{sql_query}': {retry_exc}") from retry_exc
    except SQLAlchemyError as exc:
        raise RuntimeError(f"SQL read failed for '{sql_query}': {exc}") from exc
    finally:
        engine.dispose()


def load_training_data() -> TrainArtifacts:
    products_df = read_table_with_retry("SELECT * FROM products ORDER BY product_id")
    interactions_df = read_table_with_retry("SELECT * FROM interactions")
    if products_df.empty or interactions_df.empty:
        raise ValueError("Training data is empty. Run data_ingestion.py first.")
    return TrainArtifacts(products=products_df, interactions=interactions_df)


def train_and_log_als(artifacts: TrainArtifacts) -> float:
    _configure_windows_spark_runtime()
    spark = (
        SparkSession.builder.appName("CartSense-ALS-Training")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    try:
        ratings_df = spark.createDataFrame(
            artifacts.interactions[["user_id", "product_id", "rating"]].astype(
                {"user_id": int, "product_id": int, "rating": float}
            )
        )
        train_df, test_df = ratings_df.randomSplit([0.8, 0.2], seed=42)
        if test_df.count() == 0:
            train_df, test_df = ratings_df.randomSplit([0.7, 0.3], seed=7)

        als = ALS(
            userCol="user_id",
            itemCol="product_id",
            ratingCol="rating",
            rank=10,
            maxIter=12,
            regParam=0.08,
            nonnegative=True,
            coldStartStrategy="drop",
        )
        als_model = als.fit(train_df)
        predictions = als_model.transform(test_df)
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction",
        )
        rmse = float(evaluator.evaluate(predictions))

        with mlflow.start_run(run_name="als_training"):
            mlflow.log_params(
                {
                    "rank": 10,
                    "max_iter": 12,
                    "reg_param": 0.08,
                    "cold_start_strategy": "drop",
                }
            )
            mlflow.log_metric("als_rmse", rmse)
            mlflow.spark.log_model(
                spark_model=als_model,
                artifact_path="als_model",
                registered_model_name=ALS_MODEL_NAME,
            )
        return rmse
    finally:
        spark.stop()


def train_and_log_tfidf(artifacts: TrainArtifacts) -> float:
    products_df = artifacts.products.copy()
    interactions_df = artifacts.interactions.copy()
    products_df["specs_text"] = products_df["specs_text"].fillna("")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(products_df["specs_text"])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    product_to_idx = {pid: idx for idx, pid in enumerate(products_df["product_id"].tolist())}
    y_true: list[float] = []
    y_pred: list[float] = []

    for _, interaction in interactions_df.iterrows():
        user_id = int(interaction["user_id"])
        anchor_pid = int(interaction["product_id"])
        true_rating = float(interaction["rating"])
        if anchor_pid not in product_to_idx:
            continue

        peer_pids = interactions_df.loc[
            (interactions_df["user_id"] == user_id) & (interactions_df["product_id"] != anchor_pid),
            "product_id",
        ].astype(int).tolist()

        idx = product_to_idx[anchor_pid]
        valid_peer_scores = [
            similarity_matrix[idx, product_to_idx[peer_pid]]
            for peer_pid in peer_pids
            if peer_pid in product_to_idx
        ]
        predicted_rating = 1.0 + (4.0 * float(np.mean(valid_peer_scores))) if valid_peer_scores else 3.0
        y_true.append(true_rating)
        y_pred.append(predicted_rating)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if y_true else 0.0

    with mlflow.start_run(run_name="tfidf_training"):
        mlflow.log_params({"ngram_range": "(1,2)", "min_df": 1})
        mlflow.log_metric("tfidf_rmse", rmse)
        mlflow.sklearn.log_model(
            sk_model=vectorizer,
            artifact_path="tfidf_vectorizer",
            registered_model_name=TFIDF_MODEL_NAME,
        )
    return rmse


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("CartSense_Hybrid_Recommendation")
    artifacts = load_training_data()
    als_rmse = train_and_log_als(artifacts)
    tfidf_rmse = train_and_log_tfidf(artifacts)
    print(f"ALS RMSE: {als_rmse:.4f}")
    print(f"TF-IDF RMSE: {tfidf_rmse:.4f}")


if __name__ == "__main__":
    main()
