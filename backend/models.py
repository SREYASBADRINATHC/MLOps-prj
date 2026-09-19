"""
CartSense — SQLAlchemy ORM Models
All 7 database tables with proper indexes.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)

from backend.database import Base


class Product(Base):
    """Electronics product catalog."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(String(64), unique=True, nullable=False, index=True)
    category = Column(String(32), nullable=False, index=True)  # laptop | smartphone
    brand = Column(String(64), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    spec_text = Column(Text, nullable=False)  # Free-text for TF-IDF

    # Structured numeric specs (for spec matching)
    processor = Column(String(128), nullable=True)
    ram_gb = Column(Float, nullable=True)
    storage_gb = Column(Float, nullable=True)
    display_type = Column(String(64), nullable=True)   # OLED | IPS | AMOLED | LCD
    refresh_rate_hz = Column(Integer, nullable=True)
    gpu = Column(String(128), nullable=True)
    battery_mah = Column(Integer, nullable=True)
    camera_mp = Column(Integer, nullable=True)   # main/primary camera
    chipset = Column(String(128), nullable=True)
    os = Column(String(64), nullable=True)
    price_usd = Column(Float, nullable=False)
    image_url = Column(Text, nullable=True)
    product_url = Column(Text, nullable=True)
    release_date = Column(Date, nullable=True)
    is_upcoming = Column(Boolean, nullable=True)

    interaction_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(
        DateTime, default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_products_category_brand", "category", "brand"),
        Index("ix_products_price", "price_usd"),
    )


class User(Base):
    """Application users."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), unique=True, nullable=False, index=True)
    segment = Column(String(64), nullable=True)  # e.g. gaming, budget, photography
    created_at = Column(DateTime, default=func.now(), nullable=False)


class Interaction(Base):
    """User–product interaction events."""

    __tablename__ = "interactions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(64), nullable=False)
    product_id = Column(String(64), nullable=False)
    interaction_type = Column(String(32), nullable=False)  # view|click|wishlist|cart|purchase|rating
    rating = Column(Float, nullable=True)   # explicit rating 1-5 if provided
    weight = Column(Float, nullable=False)  # derived weight for ALS
    timestamp = Column(DateTime, default=func.now(), nullable=False)

    __table_args__ = (
        Index("ix_interactions_user_product", "user_id", "product_id"),
        Index("ix_interactions_timestamp", "timestamp"),
    )


class PredictionLog(Base):
    """Log of every recommendation API call."""

    __tablename__ = "prediction_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    product_id = Column(String(64), nullable=False, index=True)
    recommendation_mode = Column(String(32), nullable=False)  # cold_start|hybrid|new_user_fallback
    model_version = Column(String(64), nullable=True)
    top_k = Column(Integer, nullable=True)
    latency_ms = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)


class TrainingRun(Base):
    """Record of each model training execution."""

    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(128), nullable=True)   # MLflow run ID
    model_type = Column(String(32), nullable=False)   # tfidf|als|hybrid
    model_version = Column(String(64), nullable=True)
    rmse = Column(Float, nullable=True)
    precision_at_k = Column(Float, nullable=True)
    recall_at_k = Column(Float, nullable=True)
    ndcg_at_k = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="pending")  # pending|success|failed|rejected
    promoted = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)


class DriftMetric(Base):
    """Evidently drift detection results stored per-run."""

    __tablename__ = "drift_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=True)
    metric_name = Column(String(128), nullable=False)
    metric_value = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    status = Column(String(32), nullable=True)   # drifted|stable
    dataset_drift_detected = Column(Boolean, default=False, nullable=False)
    share_drifted = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True)


class ModelRegistryMetadata(Base):
    """Local shadow of MLflow model registry state."""

    __tablename__ = "model_registry_metadata"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False, index=True)
    model_version = Column(String(64), nullable=False)
    mlflow_run_id = Column(String(128), nullable=True)
    stage = Column(String(32), nullable=False, default="Staging")   # Staging|Production|Archived
    rmse = Column(Float, nullable=True)
    precision_at_k = Column(Float, nullable=True)
    recall_at_k = Column(Float, nullable=True)
    ndcg_at_k = Column(Float, nullable=True)
    artifact_path = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False, nullable=False)
    registered_at = Column(DateTime, default=func.now(), nullable=False)
    promoted_at = Column(DateTime, nullable=True)
