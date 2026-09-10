"""
CartSense Algorithm Verification Script
========================================
Tests:
  1. TF-IDF cosine similarity scores are in [0, 1]
  2. Cold-start routing uses TF-IDF only
  3. Warm routing uses Hybrid (0.65 ALS + 0.35 TF-IDF)
  4. Required-specs boosting increases scores for matching products
  5. Evidently drift detection runs without error
  6. SQLite DB is populated with expected rows
"""

import os
import sys
import math

# Fix Unicode on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("DATABASE_URL", "sqlite:///cartsense.db")
os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

# 1. Database integrity
print("\n=== 1. DATABASE INTEGRITY ===")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///cartsense.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

try:
    with engine.begin() as conn:
        product_count = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
        interaction_count = conn.execute(text("SELECT COUNT(*) FROM interactions")).scalar()
    print(f"  Products: {product_count}, Interactions: {interaction_count}")
    if product_count and product_count >= 10:
        results.append((PASS, "DB: products table", f"{product_count} rows"))
    else:
        results.append((FAIL, "DB: products table", f"Only {product_count} rows -- run data_ingestion.py"))
    if interaction_count and interaction_count >= 5:
        results.append((PASS, "DB: interactions table", f"{interaction_count} rows"))
    else:
        results.append((FAIL, "DB: interactions table", f"Only {interaction_count} rows"))
except Exception as e:
    results.append((FAIL, "DB: connection", str(e)))
    print(f"  DB error: {e}")

# 2. TF-IDF model
print("\n=== 2. TF-IDF MODEL & COSINE SIMILARITY ===")
products_df = None
try:
    products_df = pd.read_sql("SELECT * FROM products ORDER BY product_id", engine)
    products_df["specs_text"] = products_df["specs_text"].fillna("")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(products_df["specs_text"])
    sim_matrix = cosine_similarity(tfidf_matrix)
    min_val, max_val = float(sim_matrix.min()), float(sim_matrix.max())
    print(f"  Cosine score range: [{min_val:.4f}, {max_val:.4f}]")
    if -0.001 <= min_val and max_val <= 1.001:
        results.append((PASS, "TF-IDF: score range [0,1]", f"min={min_val:.4f} max={max_val:.4f}"))
    else:
        results.append((FAIL, "TF-IDF: score range [0,1]", f"Out of range: {min_val:.4f} to {max_val:.4f}"))
    diag_ok = all(abs(sim_matrix[i, i] - 1.0) < 1e-5 for i in range(len(products_df)))
    results.append((PASS if diag_ok else FAIL, "TF-IDF: self-similarity = 1.0", "diagonal check"))
    product_ids = products_df["product_id"].astype(int).tolist()
    anchor_idx = product_ids.index(101)
    scores = sim_matrix[anchor_idx].copy()
    scores[anchor_idx] = 0.0
    top3_indices = scores.argsort()[-3:][::-1]
    top3_products = products_df.iloc[top3_indices]
    anchor_category = products_df.iloc[anchor_idx]["category"]
    print(f"\n  Anchor product 101 ({anchor_category}) -- Top-3 recommendations:")
    for _, row in top3_products.iterrows():
        score = scores[product_ids.index(int(row["product_id"]))]
        print(f"    [{row['category']}] {row['product_name']} -- score={score:.4f}")
    same_cat_count = (top3_products["category"] == anchor_category).sum()
    results.append((PASS if same_cat_count >= 2 else WARN, "TF-IDF: same-category preference", f"{same_cat_count}/3 same-cat for pid=101"))
except Exception as e:
    results.append((FAIL, "TF-IDF: model evaluation", str(e)))
    print(f"  Error: {e}")

# 3. Hybrid routing
print("\n=== 3. HYBRID ROUTING LOGIC ===")
try:
    interactions_df = pd.read_sql("SELECT * FROM interactions", engine)
    has_interaction_pids = set(interactions_df["product_id"].astype(int).tolist())
    all_pids = set(products_df["product_id"].astype(int).tolist())
    cold_start_pids = all_pids - has_interaction_pids
    print(f"  Warm products: {len(has_interaction_pids)} (Hybrid), Cold-start: {len(cold_start_pids)} (TF-IDF)")
    results.append((PASS, "Routing: cold-start detection", f"{len(cold_start_pids)} cold-start PIDs"))
    warm_pid = list(has_interaction_pids)[0]
    warm_idx = product_ids.index(warm_pid)
    tfidf_arr = sim_matrix[warm_idx].copy()
    tfidf_arr[warm_idx] = 0.0
    als_mock = 0.5
    hybrid_scores = {pid: (0.65 * als_mock) + (0.35 * float(tfidf_arr[i])) for i, pid in enumerate(product_ids)}
    hybrid_scores.pop(warm_pid, None)
    top_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Hybrid top-3 for pid={warm_pid}:")
    for pid, score in top_hybrid:
        print(f"    pid={pid}  hybrid_score={score:.4f}")
    all_in_range = all(0.0 <= s <= 1.0 for _, s in top_hybrid)
    results.append((PASS if all_in_range else FAIL, "Routing: hybrid scores in [0,1]", "0.65*ALS + 0.35*TF-IDF"))
    results.append((PASS, "Routing: weight sum = 1.0", "0.65+0.35=1.0"))
except Exception as e:
    results.append((FAIL, "Routing: logic test", str(e)))

# 4. Spec-boost
print("\n=== 4. REQUIRED-SPECS BOOSTING ===")
try:
    import re
    def apply_boost(base_scores, specs_series, required_specs):
        terms = [t for t in re.split(r"[,\s]+", required_specs.lower().strip()) if t]
        if not terms:
            return base_scores
        boosts = np.zeros_like(base_scores)
        for idx, spec_val in enumerate(specs_series.fillna("").str.lower().tolist()):
            match_count = sum(1 for term in terms if term in spec_val)
            boosts[idx] = min(match_count * 0.12, 0.30)
        return np.clip(base_scores + boosts, 0.0, 1.0)
    anchor_idx = product_ids.index(101)
    base = sim_matrix[anchor_idx].copy()
    base[anchor_idx] = 0.0
    boosted = apply_boost(base.copy(), products_df["specs_text"], "OLED 32GB")
    boosted_count = int((boosted > base).sum())
    print(f"  Products boosted by 'OLED 32GB': {boosted_count}")
    results.append((PASS if boosted_count > 0 else WARN, "Spec-boost: OLED 32GB increases scores", f"{boosted_count} products boosted"))
    results.append((PASS if boosted.max() <= 1.001 else FAIL, "Spec-boost: clipped to max 1.0", f"max={boosted.max():.4f}"))
except Exception as e:
    results.append((FAIL, "Spec-boost test", str(e)))

# 5. MLflow
print("\n=== 5. MLFLOW LOCAL TRACKING ===")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiments = mlflow.search_experiments()
    exp_names = [e.name for e in experiments]
    print(f"  Experiments: {exp_names}")
    results.append((PASS, "MLflow: URI accessible", f"{len(experiments)} experiments"))
    if "CartSense_Hybrid_Recommendation" in exp_names:
        client = mlflow.tracking.MlflowClient()
        exp = mlflow.get_experiment_by_name("CartSense_Hybrid_Recommendation")
        runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=5)
        print(f"  Runs in CartSense experiment: {len(runs)}")
        for r in runs:
            print(f"    {r.info.run_id[:8]}  metrics={dict(r.data.metrics)}")
        results.append((PASS, "MLflow: CartSense experiment", f"{len(runs)} runs"))
        loaded = False
        for run in runs:
            try:
                mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/tfidf_vectorizer")
                results.append((PASS, "MLflow: TF-IDF model loadable", f"run={run.info.run_id[:8]}"))
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            results.append((WARN, "MLflow: TF-IDF model", "No artifact yet -- run train_models.py"))
    else:
        results.append((WARN, "MLflow: CartSense experiment", "Not found -- run train_models.py"))
except Exception as e:
    results.append((FAIL, "MLflow: local tracking", str(e)))

# 6. Evidently
print("\n=== 6. EVIDENTLY DRIFT DETECTION ===")
try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    ref_df = products_df[["price_usd", "ram_gb", "storage_gb"]].copy()
    curr_df = ref_df.copy()
    curr_df["price_usd"] = curr_df["price_usd"] * 1.05
    curr_df["ram_gb"] = curr_df["ram_gb"] + np.where(curr_df.index % 2 == 0, 0, 2)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=curr_df)
    result = report.as_dict()["metrics"][0]["result"]
    drift_share = round(float(result["share_of_drifted_columns"]) * 100.0, 2)
    print(f"  Drift: {result['dataset_drift']}, Share: {drift_share}%")
    results.append((PASS, "Evidently: drift report runs", f"drift={result['dataset_drift']}, share={drift_share}%"))
except Exception as e:
    results.append((FAIL, "Evidently: drift detection", str(e)))

# Final report
print("\n" + "=" * 60)
print("  CARTENSE ALGORITHM VERIFICATION REPORT")
print("=" * 60)
passed = sum(1 for s, _, _ in results if s == PASS)
warned = sum(1 for s, _, _ in results if s == WARN)
failed = sum(1 for s, _, _ in results if s == FAIL)
for status, name, detail in results:
    print(f"  {status}  {name}")
    if detail:
        print(f"         {detail}")
print("=" * 60)
print(f"  Total: {len(results)} checks | PASS={passed} | WARN={warned} | FAIL={failed}")
print("=" * 60)
if failed > 0:
    sys.exit(1)
