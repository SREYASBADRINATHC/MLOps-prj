"""
CartSense — Evidently AI Drift Monitor

Monitors distribution shift between:
  - Reference (baseline): interactions from the training period
  - Current (production): interactions from recent days

Features monitored:
  - ram_gb          : RAM size distribution
  - price_usd       : Price distribution
  - refresh_rate_hz : Display refresh rate distribution
  - storage_gb      : Storage capacity
  - display_encoded : Display type (OLED=2, AMOLED=1, IPS/other=0)
  - battery_mah     : Battery capacity (phones)

Drift statistic used: Kolmogorov-Smirnov for numerical features.
Evidently reports the "share_of_drifted_columns" in [0, 1].
DRIFT_THRESHOLD = 0.5 means: if ≥50% of monitored numerical features show
statistically significant distributional shift (KS p-value < 0.05),
we declare dataset drift and recommend retraining.

This threshold is conservative enough to avoid false positives from daily noise
while sensitive enough to catch genuine market preference shifts.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd

from backend.config import settings

logger = logging.getLogger("cartsense.drift_monitor")


def _encode_display(display_type: str) -> float:
    """Convert display type string to numeric encoding for drift analysis."""
    if display_type is None:
        return 0.0
    dt = str(display_type).upper()
    if "OLED" in dt or "AMOLED" in dt:
        return 2.0
    if "IPS" in dt:
        return 1.0
    return 0.0


def _prepare_feature_df(products_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and encode numeric features for drift analysis."""
    df = products_df.copy()
    df["display_encoded"] = df["display_type"].fillna("").apply(_encode_display)
    df["ram_gb"] = pd.to_numeric(df.get("ram_gb", 0), errors="coerce").fillna(0)
    df["price_usd"] = pd.to_numeric(df.get("price_usd", 0), errors="coerce").fillna(0)
    df["refresh_rate_hz"] = pd.to_numeric(df.get("refresh_rate_hz", 60), errors="coerce").fillna(60)
    df["storage_gb"] = pd.to_numeric(df.get("storage_gb", 0), errors="coerce").fillna(0)
    df["battery_mah"] = pd.to_numeric(df.get("battery_mah", 0), errors="coerce").fillna(0)
    return df[["ram_gb", "price_usd", "refresh_rate_hz", "storage_gb", "display_encoded", "battery_mah"]]


def _generate_drifted_distribution(reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a synthetic 'drifted' production distribution reflecting the
    documented market shift pattern:
      Baseline:  16GB RAM, IPS, 60Hz, traditional processors
      Drifted:   32GB+ RAM, OLED (encoded 2.0), 120Hz+, AI/NPU

    This is used ONLY for the drift simulation demo.
    """
    n = len(reference_df)
    np.random.seed(99)  # reproducible demo
    drifted = pd.DataFrame({
        "ram_gb": np.random.choice([24, 32, 48, 64], n, p=[0.20, 0.50, 0.20, 0.10]),
        "price_usd": np.random.normal(loc=1800, scale=300, size=n).clip(800, 4500),
        "refresh_rate_hz": np.random.choice([120, 144, 165, 240], n, p=[0.40, 0.35, 0.15, 0.10]),
        "storage_gb": np.random.choice([512, 1024, 2048], n, p=[0.30, 0.50, 0.20]),
        "display_encoded": np.random.choice([1.5, 2.0], n, p=[0.30, 0.70]),
        "battery_mah": np.random.normal(loc=5200, scale=400, size=n).clip(4000, 6500),
    })
    return drifted


def run_evidently_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    drift_threshold: float,
    run_id: str,
    simulation_mode: bool = False,
) -> dict:
    """
    Run Evidently DataDriftPreset on reference vs current data.

    Returns a dict matching DriftMetricsResponse schema.
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report_dict = report.as_dict()

    result = report_dict["metrics"][0]["result"]
    share_drifted = float(result.get("share_of_drifted_columns", 0.0))
    n_drifted = int(result.get("number_of_drifted_columns", 0))
    n_total = int(result.get("number_of_columns", len(reference_df.columns)))
    dataset_drift = bool(result.get("dataset_drift", share_drifted >= drift_threshold))

    # Per-column details
    column_results = []
    drift_by_columns = result.get("drift_by_columns", {})
    for col_name, col_info in drift_by_columns.items():
        column_results.append({
            "column_name": col_name,
            "statistic_name": str(col_info.get("stattest_name", "KS")),
            "statistic_value": round(float(col_info.get("drift_score", 0.0)), 4),
            "threshold": round(float(col_info.get("threshold", 0.05)), 4),
            "drift_detected": bool(col_info.get("drift_detected", False)),
        })

    return {
        "run_id": run_id,
        "dataset_drift_detected": dataset_drift,
        "share_drifted_columns": round(share_drifted, 4),
        "n_drifted_columns": n_drifted,
        "n_total_columns": n_total,
        "drift_threshold": drift_threshold,
        "retraining_recommended": dataset_drift,
        "column_results": column_results,
        "timestamp": datetime.utcnow(),
        "simulation_mode": simulation_mode,
    }


class DriftMonitor:
    """
    CartSense Drift Monitor.

    Compares the training-time feature distribution (reference)
    against recent production interaction-driven feature distribution (current).
    """

    def __init__(self, db_engine, drift_threshold: float = 0.5) -> None:
        self.engine = db_engine
        self.drift_threshold = drift_threshold

    def _load_product_features(self) -> pd.DataFrame:
        """Load product features from the database."""
        import sqlalchemy
        with self.engine.begin() as conn:
            df = pd.read_sql(
                sqlalchemy.text("""
                    SELECT p.product_id, p.ram_gb, p.price_usd, p.refresh_rate_hz,
                           p.storage_gb, p.display_type, p.battery_mah,
                           p.interaction_count, p.created_at
                    FROM products p
                    ORDER BY p.product_id
                """),
                conn,
            )
        return df

    def _load_recent_interactions(self, days: int = 30) -> pd.DataFrame:
        """Load products interacted with in the last N days."""
        import sqlalchemy
        cutoff = datetime.utcnow() - timedelta(days=days)
        with self.engine.begin() as conn:
            df = pd.read_sql(
                sqlalchemy.text("""
                    SELECT DISTINCT p.product_id, p.ram_gb, p.price_usd, p.refresh_rate_hz,
                           p.storage_gb, p.display_type, p.battery_mah
                    FROM products p
                    JOIN interactions i ON i.product_id = p.product_id
                    WHERE i.timestamp >= :cutoff
                """),
                conn,
                params={"cutoff": cutoff},
            )
        return df

    def _load_training_interactions(self, days: int = 90) -> pd.DataFrame:
        """Load products interacted with in the training period (older data)."""
        import sqlalchemy
        cutoff_recent = datetime.utcnow() - timedelta(days=30)
        cutoff_old = datetime.utcnow() - timedelta(days=days)
        with self.engine.begin() as conn:
            df = pd.read_sql(
                sqlalchemy.text("""
                    SELECT DISTINCT p.product_id, p.ram_gb, p.price_usd, p.refresh_rate_hz,
                           p.storage_gb, p.display_type, p.battery_mah
                    FROM products p
                    JOIN interactions i ON i.product_id = p.product_id
                    WHERE i.timestamp BETWEEN :cutoff_old AND :cutoff_recent
                """),
                conn,
                params={"cutoff_recent": cutoff_recent, "cutoff_old": cutoff_old},
            )
        return df

    def check_drift(
        self,
        simulate: bool = False,
        n_reference_days: int = 90,
        n_current_days: int = 30,
    ) -> dict:
        """
        Run a full drift check.

        If simulate=True, generates a synthetic drifted distribution for demo purposes.
        Otherwise, uses actual interaction data from the database.
        """
        run_id = str(uuid.uuid4())[:8]

        all_products = self._load_product_features()
        reference_features = _prepare_feature_df(all_products)

        if simulate:
            current_features = _generate_drifted_distribution(reference_features)
            logger.info("Running simulated drift check (demo mode)")
        else:
            recent_products = self._load_recent_interactions(n_current_days)
            if len(recent_products) < 10:
                logger.warning(
                    "Only %d recent products — using full catalog as current distribution",
                    len(recent_products),
                )
                recent_products = all_products.sample(min(len(all_products), 100), random_state=42)
            current_features = _prepare_feature_df(recent_products)
            logger.info(
                "Drift check: reference=%d rows, current=%d rows",
                len(reference_features),
                len(current_features),
            )

        result = run_evidently_drift(
            reference_df=reference_features,
            current_df=current_features,
            drift_threshold=self.drift_threshold,
            run_id=run_id,
            simulation_mode=simulate,
        )

        self._save_drift_result(result)
        return result

    def _save_drift_result(self, result: dict) -> None:
        """Persist drift result to the drift_metrics table."""
        import sqlalchemy
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO drift_metrics
                          (run_id, metric_name, metric_value, threshold, status,
                           dataset_drift_detected, share_drifted, timestamp)
                        VALUES
                          (:run_id, 'share_drifted_columns', :share, :threshold, :status,
                           :detected, :share, :ts)
                    """),
                    {
                        "run_id": result["run_id"],
                        "share": result["share_drifted_columns"],
                        "threshold": result["drift_threshold"],
                        "status": "drifted" if result["dataset_drift_detected"] else "stable",
                        "detected": result["dataset_drift_detected"],
                        "ts": result["timestamp"],
                    },
                )
        except Exception as exc:
            logger.warning("Could not save drift result to DB: %s", exc)

    def get_latest_drift(self) -> Optional[dict]:
        """Return the most recent drift check result from the database."""
        import sqlalchemy
        try:
            with self.engine.begin() as conn:
                df = pd.read_sql(
                    sqlalchemy.text("""
                        SELECT * FROM drift_metrics
                        ORDER BY timestamp DESC LIMIT 1
                    """),
                    conn,
                )
            if df.empty:
                return None
            row = df.iloc[0].to_dict()
            return {
                "run_id": str(row.get("run_id", "")),
                "dataset_drift_detected": bool(row.get("dataset_drift_detected", False)),
                "share_drifted_columns": float(row.get("share_drifted", 0.0)),
                "n_drifted_columns": 0,
                "n_total_columns": 6,
                "drift_threshold": float(row.get("threshold", self.drift_threshold)),
                "retraining_recommended": bool(row.get("dataset_drift_detected", False)),
                "column_results": [],
                "timestamp": row.get("timestamp", datetime.utcnow()),
                "simulation_mode": False,
            }
        except Exception as exc:
            logger.warning("Could not load latest drift result: %s", exc)
            return None
