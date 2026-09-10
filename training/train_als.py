"""
CartSense — PySpark ALS Collaborative Filtering Training

Trains PySpark MLlib ALS from scratch on our user-item interaction data.
No pretrained embeddings or external models.

Interaction weights:
  view=1, click=2, wishlist=3, cart=4, purchase=5

Outputs (saved to artifacts/):
  - als_item_factors.parquet  : item latent factors (item_id, features)
  - als_user_factors.parquet  : user latent factors (user_id, features)
  - als_metadata.json         : model metadata + integer→string ID maps

Also logs metrics to MLflow.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_als")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://cartsense_user:cartsense_pass@postgres_db:5432/cartsense",
)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
ALS_MODEL_NAME = os.getenv("ALS_MODEL_NAME", "CartSense_ALS")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")
ALS_RANK = int(os.getenv("ALS_RANK", "20"))
ALS_MAX_ITER = int(os.getenv("ALS_MAX_ITER", "15"))
ALS_REG_PARAM = float(os.getenv("ALS_REG_PARAM", "0.1"))
ALS_SEED = int(os.getenv("ALS_SEED", "42"))


def get_engine():
    is_sqlite = DATABASE_URL.startswith("sqlite")
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "pool_recycle": 1800}
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = 3
        kwargs["max_overflow"] = 5
    return create_engine(DATABASE_URL, **kwargs)


def load_data(eng) -> tuple[pd.DataFrame, pd.DataFrame]:
    with eng.begin() as conn:
        interactions_df = pd.read_sql(
            text("""
                SELECT user_id, product_id, weight, rating, timestamp
                FROM interactions ORDER BY timestamp
            """),
            conn,
        )
        products_df = pd.read_sql(
            text("SELECT product_id, name, category FROM products ORDER BY product_id"),
            conn,
        )
    logger.info("Loaded %d interactions, %d products.", len(interactions_df), len(products_df))
    return interactions_df, products_df


def build_integer_maps(
    interactions_df: pd.DataFrame, products_df: pd.DataFrame
) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    """
    ALS requires integer user/item IDs.
    We build a bijective mapping:
      user_id (string) ↔ user_int (int)
      product_id (string) ↔ item_int (int)
    """
    all_user_ids = sorted(interactions_df["user_id"].unique())
    all_product_ids = sorted(products_df["product_id"].unique())

    user_to_int: dict[str, int] = {uid: i for i, uid in enumerate(all_user_ids)}
    int_to_user: dict[int, str] = {i: uid for uid, i in user_to_int.items()}
    item_to_int: dict[str, int] = {pid: i for i, pid in enumerate(all_product_ids)}
    int_to_item: dict[int, str] = {i: pid for pid, i in item_to_int.items()}

    df = interactions_df.copy()
    df["user_int"] = df["user_id"].map(user_to_int)
    df["item_int"] = df["product_id"].map(item_to_int)
    df = df.dropna(subset=["user_int", "item_int"])
    df["user_int"] = df["user_int"].astype(int)
    df["item_int"] = df["item_int"].astype(int)

    logger.info(
        "Mapped %d users, %d items to integers.", len(user_to_int), len(item_to_int)
    )
    return df, user_to_int, int_to_user, item_to_int, int_to_item


def temporal_split(df: pd.DataFrame, train_frac: float = 0.80):
    """Split interactions by time: older → train, newer → test."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def train_als_spark(
    train_df: pd.DataFrame,
    rank: int,
    max_iter: int,
    reg_param: float,
    seed: int,
) -> tuple[Any, Any, Any]:
    """
    Train PySpark ALS on the integer-mapped interaction matrix.
    Returns (als_model, user_factors_df, item_factors_df)
    """
    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    # Suppress PySpark INFO logging noise
    import pyspark
    from pyspark.sql import SparkSession
    from pyspark.ml.recommendation import ALS

    spark = (
        SparkSession.builder.appName("CartSense_ALS")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    logger.info("Spark session started (version %s)", spark.version)

    # Aggregate duplicate user-item pairs (sum weights → implicit feedback)
    agg = (
        train_df.groupby(["user_int", "item_int"])["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "rating"})
    )

    spark_df = spark.createDataFrame(agg[["user_int", "item_int", "rating"]])
    spark_df = spark_df.withColumnRenamed("user_int", "user").withColumnRenamed("item_int", "item")

    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="user",
        itemCol="item",
        ratingCol="rating",
        implicitPrefs=False,       # we use explicit aggregated weights
        coldStartStrategy="drop",
        seed=seed,
        nonnegative=True,
    )

    logger.info("Training ALS (rank=%d, maxIter=%d, regParam=%.3f)...", rank, max_iter, reg_param)
    model = als.fit(spark_df)
    logger.info("ALS training complete.")

    # Extract latent factors as pandas DataFrames
    user_factors = model.userFactors.toPandas()
    item_factors = model.itemFactors.toPandas()

    spark.stop()
    return model, user_factors, item_factors


def evaluate_als(
    test_df: pd.DataFrame,
    user_factors_df: pd.DataFrame,
    item_factors_df: pd.DataFrame,
    int_to_item: dict[int, str],
    K: int = 5,
) -> dict[str, float]:
    """
    Evaluate ALS by computing dot-product scores for test users.
    Returns RMSE, Precision@K, Recall@K, NDCG@K.
    """
    # Build numpy factor matrices
    user_factors_df = user_factors_df.set_index("id")
    item_factors_df = item_factors_df.set_index("id")

    # Sort item factors by index for consistent ordering
    item_ids_sorted = sorted(item_factors_df.index.tolist())
    item_matrix = np.vstack([np.array(item_factors_df.loc[i, "features"]) for i in item_ids_sorted])

    # RMSE on weighted ratings
    y_true, y_pred = [], []
    for _, row in test_df.iterrows():
        uid_int = int(row["user_int"])
        iid_int = int(row["item_int"])
        if uid_int not in user_factors_df.index or iid_int not in item_factors_df.index:
            continue
        u_vec = np.array(user_factors_df.loc[uid_int, "features"])
        i_vec = np.array(item_factors_df.loc[iid_int, "features"])
        pred_rating = float(np.dot(u_vec, i_vec))
        y_true.append(float(row["weight"]))
        y_pred.append(pred_rating)

    rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if y_true else float("nan")

    # Precision@K / Recall@K / NDCG@K
    test_users = test_df["user_int"].unique()
    prec_scores, rec_scores, ndcg_list = [], [], []

    train_user_items: dict[int, set[int]] = {}
    for uid in test_users[:200]:
        uid_int = int(uid)
        if uid_int not in user_factors_df.index:
            continue
        u_vec = np.array(user_factors_df.loc[uid_int, "features"])
        scores = item_matrix @ u_vec  # shape: (n_items,)
        top_k = set(np.argsort(scores)[::-1][:K].tolist())

        relevant = set(test_df[test_df["user_int"] == uid_int]["item_int"].tolist())
        if not relevant:
            continue

        hits = top_k & relevant
        prec_scores.append(len(hits) / K)
        rec_scores.append(len(hits) / len(relevant))

        # NDCG
        top_k_list = np.argsort(scores)[::-1][:K].tolist()
        dcg = sum((1 if top_k_list[i] in relevant else 0) / math.log2(i + 2) for i in range(len(top_k_list)))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(K, len(relevant))))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

    metrics = {
        "rmse": round(rmse, 4) if not math.isnan(rmse) else 0.0,
        "precision_at_5": round(float(np.mean(prec_scores)), 4) if prec_scores else 0.0,
        "recall_at_5": round(float(np.mean(rec_scores)), 4) if rec_scores else 0.0,
        "ndcg_at_5": round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
    }
    logger.info("ALS evaluation: %s", metrics)
    return metrics


def save_als_artifacts(
    user_factors: pd.DataFrame,
    item_factors: pd.DataFrame,
    user_to_int: dict,
    int_to_user: dict,
    item_to_int: dict,
    int_to_item: dict,
    metadata: dict,
) -> None:
    """Save ALS factors and mappings to artifacts directory."""
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    user_path = os.path.join(ARTIFACTS_DIR, "als_user_factors.parquet")
    item_path = os.path.join(ARTIFACTS_DIR, "als_item_factors.parquet")

    user_factors.to_parquet(user_path, index=False)
    item_factors.to_parquet(item_path, index=False)
    logger.info("Saved user factors → %s", user_path)
    logger.info("Saved item factors → %s", item_path)

    maps_path = os.path.join(ARTIFACTS_DIR, "als_id_maps.json")
    with open(maps_path, "w") as f:
        json.dump(
            {
                "user_to_int": user_to_int,
                "int_to_item": {str(k): v for k, v in int_to_item.items()},
                "item_to_int": item_to_int,
            },
            f,
            indent=2,
        )
    logger.info("Saved ID maps → %s", maps_path)

    meta_path = os.path.join(ARTIFACTS_DIR, "als_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info("Saved ALS metadata → %s", meta_path)


def log_als_to_mlflow(metadata: dict, metrics: dict) -> str:
    """Log ALS run to MLflow (parameters + metrics only; factors stored locally)."""
    for uri in [MLFLOW_TRACKING_URI, "./mlruns"]:
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.search_experiments()
            logger.info("MLflow connected at %s", uri)
            break
        except Exception:
            logger.warning("MLflow URI %s unreachable.", uri)

    mlflow.set_experiment("CartSense_Hybrid_Recommendation")

    with mlflow.start_run(run_name="als_training") as run:
        mlflow.log_params({
            "rank": metadata["rank"],
            "max_iter": metadata["max_iter"],
            "reg_param": metadata["reg_param"],
            "seed": metadata["seed"],
            "n_users": metadata["n_users"],
            "n_items": metadata["n_items"],
            "n_interactions_train": metadata["n_interactions_train"],
            "cold_start_strategy": "drop",
            "training_timestamp": metadata["training_timestamp"],
        })
        mlflow.log_metrics(metrics)
        # Log local artifact folder
        mlflow.log_artifacts(ARTIFACTS_DIR, artifact_path="als_artifacts")
        return run.info.run_id


# Fallback: mean_squared_error not imported at top level — guard
try:
    from sklearn.metrics import mean_squared_error
except ImportError:
    def mean_squared_error(y_true, y_pred):  # type: ignore
        arr = np.array(y_true) - np.array(y_pred)
        return float(np.mean(arr ** 2))


def main() -> None:
    t0 = time.time()
    eng = get_engine()

    interactions_df, products_df = load_data(eng)
    df, user_to_int, int_to_user, item_to_int, int_to_item = build_integer_maps(
        interactions_df, products_df
    )

    train_df, test_df = temporal_split(df, train_frac=0.80)
    logger.info("Train size: %d | Test size: %d", len(train_df), len(test_df))

    _, user_factors, item_factors = train_als_spark(
        train_df,
        rank=ALS_RANK,
        max_iter=ALS_MAX_ITER,
        reg_param=ALS_REG_PARAM,
        seed=ALS_SEED,
    )

    metrics = evaluate_als(test_df, user_factors, item_factors, int_to_item, K=5)

    metadata = {
        "rank": ALS_RANK,
        "max_iter": ALS_MAX_ITER,
        "reg_param": ALS_REG_PARAM,
        "seed": ALS_SEED,
        "n_users": len(user_to_int),
        "n_items": len(item_to_int),
        "n_interactions_train": len(train_df),
        "n_interactions_test": len(test_df),
        "training_timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics,
    }

    save_als_artifacts(
        user_factors, item_factors,
        user_to_int, int_to_user,
        item_to_int, int_to_item,
        metadata,
    )

    run_id = log_als_to_mlflow(metadata, metrics)

    elapsed = time.time() - t0
    print(f"✅ ALS training complete in {elapsed:.1f}s")
    print(f"   Users  : {metadata['n_users']}")
    print(f"   Items  : {metadata['n_items']}")
    print(f"   Rank   : {ALS_RANK}")
    print(f"   RMSE   : {metrics['rmse']:.4f}")
    print(f"   P@5    : {metrics['precision_at_5']:.4f}")
    print(f"   R@5    : {metrics['recall_at_5']:.4f}")
    print(f"   NDCG@5 : {metrics['ndcg_at_5']:.4f}")
    print(f"   MLflow : {run_id}")


if __name__ == "__main__":
    main()
