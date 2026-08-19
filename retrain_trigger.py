import logging
import os
import sys
import time
import numpy as np
import pandas as pd

# Fix UnicodeEncodeError for emoji characters on Windows terminals (cp1252)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress noisy urllib3 retry warnings when MLflow server is offline
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Suppress GitPython warning when git is not on PATH
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Import MLflow and Evidently components
import mlflow
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

# Setup MLflow Tracking Server URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def generate_mock_data():
    """
    Generates two DataFrames: Reference (baseline) and Current (production live data).
    To simulate Concept Drift:
    - Reference data represents stable user interactions with laptops & phones.
    - Current data represents a sudden shift (e.g., high-frequency robotic clicks or unexpected spikes).
    """
    np.random.seed(42)
    
    # 1. Reference Data (Stable baseline)
    ref_data = {
        "price_usd": np.random.normal(loc=800, scale=300, size=100),
        "clicks_per_session": np.random.poisson(lam=4, size=100),
        "session_duration_sec": np.random.exponential(scale=120, size=100),
        "bounce_rate": np.random.binomial(n=1, p=0.25, size=100)
    }
    df_ref = pd.DataFrame(ref_data)
    
    # Ensure no negative prices
    df_ref["price_usd"] = df_ref["price_usd"].clip(lower=50)

    # 2. Current Data (Simulate substantial Concept Drift / Market Shift)
    # We will simulate a drastic shift: significantly higher prices, higher clicks, and lower durations
    curr_data = {
        "price_usd": np.random.normal(loc=1600, scale=200, size=100), # Double the prices (e.g., premium shift)
        "clicks_per_session": np.random.poisson(lam=12, size=100),   # Spiked robotic interaction
        "session_duration_sec": np.random.exponential(scale=35, size=100), # Drastically reduced duration
        "bounce_rate": np.random.binomial(n=1, p=0.65, size=100)     # Massive increase in bounce rates
    }
    df_curr = pd.DataFrame(curr_data)
    df_curr["price_usd"] = df_curr["price_usd"].clip(lower=50)
    
    return df_ref, df_curr

def run_retrain_and_log_to_mlflow(drift_score: float):
    """
    Triggers retraining of the model and logs artifacts to the MLflow tracking server.
    """
    print("\n🔄 [CT TRIGGERED] Initializing PySpark & Scikit-Learn CT Retraining Loop...")
    time.sleep(1)
    
    # Setting active experiment in MLflow — fall back to local file tracking if server is offline
    try:
        mlflow.set_experiment("CartSense_Hybrid_Recommendation")
    except Exception as e:
        print(f"⚠️ Could not connect to MLflow server at {MLFLOW_TRACKING_URI}: {e}")
        print("💻 Falling back to local MLflow file tracking (./mlruns).")
        mlflow.set_tracking_uri("./mlruns")
        mlflow.set_experiment("CartSense_Hybrid_Recommendation")

    with mlflow.start_run() as run:
        # 1. Log Hyperparameters of Retrained Models
        mlflow.log_param("scikit_vectorizer", "TF-IDF")
        mlflow.log_param("scikit_similarity_metric", "Cosine")
        mlflow.log_param("sentence_transformer_model", "sentence-transformers/all-MiniLM-L6-v2")
        mlflow.log_param("pyspark_als_max_iterations", 15)
        mlflow.log_param("pyspark_als_reg_param", 0.1)
        mlflow.log_param("trigger_cause", f"Concept Drift Detected (Score: {drift_score:.2f})")
        
        # 2. Log Model Performance/Data Metrics
        mlflow.log_metric("concept_drift_score", drift_score)
        mlflow.log_metric("validation_recall_at_k", 0.885)
        mlflow.log_metric("validation_precision_at_k", 0.742)
        mlflow.log_metric("training_loss", 0.045)
        
        print("📊 [MLflow] Logged hyperparameters, metrics, and parameters successfully.")
        
        # 3. Simulate logging a mock model artifact representing the new model weights
        # In a real environment, you would log with mlflow.pyfunc.log_model() or mlflow.spark.log_model()
        model_info = {
            "model_type": "HybridRouter",
            "model_version": "v2.4.1_stable",
            "tf_idf_vocab_size": 2048,
            "spark_latent_factors_user": 10,
            "spark_latent_factors_item": 10,
        }
        
        # Save a local mock artifact file to simulate file logging
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/model_metadata.txt", "w") as f:
            f.write(str(model_info))
            
        mlflow.log_artifacts("artifacts", artifact_path="model_artifacts")
        print("📦 [MLflow] Successfully uploaded retrained model metadata and artifacts.")
        print(f"🏆 Active MLflow Run ID: {run.info.run_id}")

def evaluate_drift_and_trigger_ct():
    print("📈 Initiating Continuous Training (CT) Drift Assessment...")
    
    # 1. Generate reference and live production data
    df_ref, df_curr = generate_mock_data()
    
    # 2. Use Evidently report with DatasetDriftMetric to calculate dataset-level drift
    print("🧮 Running Evidently AI Drift Report Analyzer...")
    drift_report = Report(metrics=[DatasetDriftMetric()])
    drift_report.run(reference_data=df_ref, current_data=df_curr)
    
    # Parse report outputs
    report_dict = drift_report.as_dict()
    # Evidently stores the share of drifted features under DatasetDriftMetric
    metric_results = report_dict["metrics"][0]["result"]
    
    # Evidently v0.4.x uses 'columns' keys (not 'features')
    number_of_drifted_features = metric_results["number_of_drifted_columns"]
    total_features = metric_results["number_of_columns"]
    # Ratio of drifted columns represents the Drift Score
    drift_score = metric_results["share_of_drifted_columns"]
    
    print("\n📋 ================= DRIFT REPORT SUMMARY ================= 📋")
    print(f"🔹 Total Features Evaluated: {total_features}")
    print(f"🔹 Number of Drifted Features Detected: {number_of_drifted_features}")
    print(f"🔹 Calculated Dataset Drift Score: {drift_score:.2f} (Threshold: 0.75)")
    print("==========================================================")

    # 3. Decision Logic: Trigger retraining if drift score exceeds sensitivity threshold
    if drift_score > 0.75:
        print(f"🚨 [ALERT] Data Drift score {drift_score:.2f} is ABOVE the critical sensitivity threshold of 0.75!")
        run_retrain_and_log_to_mlflow(drift_score)
        print("\n🎉 CT Loop Completed Successfully! New model logged and prepared for deployment.")
    else:
        print(f"✅ [OK] Data Drift score {drift_score:.2f} is within acceptable parameters (<= 0.75). No retraining needed.")

if __name__ == "__main__":
    evaluate_drift_and_trigger_ct()
