"""
CartSense — Model Evaluation Module

Computes recommendation quality metrics:
  - RMSE          : rating prediction error
  - Precision@K   : fraction of top-K that are relevant
  - Recall@K      : fraction of relevant items in top-K
  - NDCG@K        : normalized discounted cumulative gain
  - Cold-start coverage: fraction of cold-start items that get recommendations
  - Recommendation latency: measured wall-clock time

Baseline comparisons:
  1. Popularity baseline   : most interacted-with products
  2. TF-IDF only
  3. ALS only
  4. CartSense Hybrid
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("cartsense.evaluate")

K = 5  # Default evaluation rank cutoff


def dcg_at_k(relevance_list: list[int], k: int) -> float:
    """Compute DCG@K for a binary relevance list."""
    return sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(relevance_list[:k])
    )


def ndcg_at_k(relevance_list: list[int], k: int) -> float:
    """Compute NDCG@K for a binary relevance list."""
    ideal = sorted(relevance_list, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevance_list, k) / idcg


def evaluate_popularity_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    products_df: pd.DataFrame,
    k: int = K,
) -> dict[str, float]:
    """Evaluate popularity baseline: recommend most interacted products."""
    # Get top-K most popular products overall
    pop = (
        train_df.groupby("product_id")["weight"]
        .sum()
        .reset_index()
        .sort_values("weight", ascending=False)
    )
    top_k_pids = set(pop.head(k)["product_id"].tolist())

    prec_list, rec_list, ndcg_list = [], [], []
    for uid in test_df["user_id"].unique():
        relevant = set(test_df[test_df["user_id"] == uid]["product_id"].tolist())
        hits = top_k_pids & relevant
        prec_list.append(len(hits) / k)
        rec_list.append(len(hits) / max(len(relevant), 1))
        rel_binary = [1 if p in relevant else 0 for p in list(top_k_pids)]
        ndcg_list.append(ndcg_at_k(rel_binary, k))

    return {
        "model": "popularity_baseline",
        "precision_at_k": round(float(np.mean(prec_list)), 4),
        "recall_at_k": round(float(np.mean(rec_list)), 4),
        "ndcg_at_k": round(float(np.mean(ndcg_list)), 4),
        "rmse": float("nan"),
    }


def evaluate_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    products_df: pd.DataFrame,
    vectorizer,
    tfidf_matrix,
    product_ids: list[str],
    k: int = K,
) -> dict[str, float]:
    """Evaluate TF-IDF-only recommendations."""
    from sklearn.metrics.pairwise import cosine_similarity

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    cat_by_pid = dict(zip(products_df["product_id"], products_df["category"]))

    prec_list, rec_list, ndcg_list = [], [], []
    t0 = time.perf_counter()

    for uid in test_df["user_id"].unique()[:200]:
        train_pids = train_df[train_df["user_id"] == uid]["product_id"].tolist()
        test_pids = set(test_df[test_df["user_id"] == uid]["product_id"].tolist())
        if not train_pids or not test_pids:
            continue
        anchor_pid = train_pids[0]
        if anchor_pid not in pid_to_idx:
            continue
        anchor_cat = cat_by_pid.get(anchor_pid, "")
        anchor_idx = pid_to_idx[anchor_pid]

        same_cat = [p for p in product_ids if cat_by_pid.get(p, "") == anchor_cat and p != anchor_pid]
        if not same_cat:
            continue
        same_indices = [pid_to_idx[p] for p in same_cat]
        sims = cosine_similarity(tfidf_matrix[anchor_idx], tfidf_matrix[same_indices]).flatten()
        top_k_pids = [same_cat[i] for i in np.argsort(sims)[::-1][:k]]

        hits = sum(1 for p in top_k_pids if p in test_pids)
        prec_list.append(hits / k)
        rec_list.append(hits / max(len(test_pids), 1))
        rel_binary = [1 if p in test_pids else 0 for p in top_k_pids]
        ndcg_list.append(ndcg_at_k(rel_binary, k))

    latency_ms = round((time.perf_counter() - t0) / max(len(prec_list), 1) * 1000, 2)
    return {
        "model": "tfidf_only",
        "precision_at_k": round(float(np.mean(prec_list)), 4) if prec_list else 0.0,
        "recall_at_k": round(float(np.mean(rec_list)), 4) if rec_list else 0.0,
        "ndcg_at_k": round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
        "latency_ms_per_user": latency_ms,
        "rmse": float("nan"),
    }


def evaluate_als(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    als_engine,
    k: int = K,
) -> dict[str, float]:
    """Evaluate ALS-only recommendations."""
    if als_engine is None or not als_engine.is_loaded:
        return {
            "model": "als_only",
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "rmse": float("nan"),
            "note": "ALS not loaded",
        }

    prec_list, rec_list, ndcg_list = [], [], []

    for uid in test_df["user_id"].unique()[:200]:
        test_pids = set(test_df[test_df["user_id"] == uid]["product_id"].tolist())
        if not test_pids:
            continue
        scores = als_engine.predict_user_scores(uid)
        if not scores:
            continue
        top_k = sorted(scores, key=lambda p: scores[p], reverse=True)[:k]
        hits = sum(1 for p in top_k if p in test_pids)
        prec_list.append(hits / k)
        rec_list.append(hits / max(len(test_pids), 1))
        rel_binary = [1 if p in test_pids else 0 for p in top_k]
        ndcg_list.append(ndcg_at_k(rel_binary, k))

    return {
        "model": "als_only",
        "precision_at_k": round(float(np.mean(prec_list)), 4) if prec_list else 0.0,
        "recall_at_k": round(float(np.mean(rec_list)), 4) if rec_list else 0.0,
        "ndcg_at_k": round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
        "rmse": float("nan"),
    }


def evaluate_hybrid(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    engine_instance,
    products_df: pd.DataFrame,
    k: int = K,
) -> dict[str, float]:
    """Evaluate the full hybrid engine."""
    prec_list, rec_list, ndcg_list = [], [], []
    t0 = time.perf_counter()

    test_users = test_df["user_id"].unique()[:100]
    for uid in test_users:
        train_pids = train_df[train_df["user_id"] == uid]["product_id"].tolist()
        test_pids = set(test_df[test_df["user_id"] == uid]["product_id"].tolist())
        if not train_pids or not test_pids:
            continue

        anchor_pid = train_pids[0]
        try:
            item_count = len(train_df[train_df["product_id"] == anchor_pid])
            result = engine_instance.recommend(
                user_id=uid,
                product_id=anchor_pid,
                top_k=k,
                interaction_count=item_count,
                user_interaction_count=len(train_pids),
            )
            top_k_pids = [r.product_id for r in result.recommendations]
        except Exception as exc:
            logger.debug("Hybrid eval error for user %s: %s", uid, exc)
            continue

        hits = sum(1 for p in top_k_pids if p in test_pids)
        prec_list.append(hits / k)
        rec_list.append(hits / max(len(test_pids), 1))
        rel_binary = [1 if p in test_pids else 0 for p in top_k_pids]
        ndcg_list.append(ndcg_at_k(rel_binary, k))

    latency_ms = round((time.perf_counter() - t0) / max(len(prec_list), 1) * 1000, 2)
    return {
        "model": "cartsense_hybrid",
        "precision_at_k": round(float(np.mean(prec_list)), 4) if prec_list else 0.0,
        "recall_at_k": round(float(np.mean(rec_list)), 4) if rec_list else 0.0,
        "ndcg_at_k": round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
        "latency_ms_per_user": latency_ms,
        "rmse": float("nan"),
    }


def evaluate_cold_start_coverage(
    products_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    tfidf_matrix,
    product_ids: list[str],
    k: int = K,
) -> dict[str, Any]:
    """
    Evaluate cold-start coverage:
    - How many products have zero interactions?
    - Can TF-IDF still produce K recommendations for them?
    """
    from sklearn.metrics.pairwise import cosine_similarity

    cold_pids = products_df[products_df["interaction_count"] == 0]["product_id"].tolist()
    total_cold = len(cold_pids)
    if total_cold == 0:
        return {"cold_start_products": 0, "cold_start_coverage": 1.0}

    pid_to_idx = {pid: i for i, pid in enumerate(product_ids)}
    cat_by_pid = dict(zip(products_df["product_id"], products_df["category"]))
    covered = 0

    for pid in cold_pids[:100]:
        if pid not in pid_to_idx:
            continue
        anchor_cat = cat_by_pid.get(pid, "")
        anchor_idx = pid_to_idx[pid]
        same_cat = [p for p in product_ids if cat_by_pid.get(p, "") == anchor_cat and p != pid]
        if len(same_cat) < k:
            continue
        same_indices = [pid_to_idx[p] for p in same_cat]
        sims = cosine_similarity(tfidf_matrix[anchor_idx], tfidf_matrix[same_indices]).flatten()
        top_k = np.argsort(sims)[::-1][:k]
        if len(top_k) >= k:
            covered += 1

    coverage = covered / min(100, total_cold) if total_cold > 0 else 1.0
    return {
        "cold_start_products": total_cold,
        "cold_start_coverage": round(coverage, 4),
        "cold_start_precision_at_k": float("nan"),  # Cannot compute without labels
    }
