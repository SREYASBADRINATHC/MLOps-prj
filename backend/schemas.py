"""
CartSense — Pydantic v2 Data Contracts

All request/response models validated at the FastAPI boundary.
No malformed input reaches model or database code.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Request models ─────────────────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    """POST /api/recommend request body."""
    user_id: str = Field(..., min_length=1, max_length=64, description="User ID (e.g. U0001)")
    product_id: str = Field(..., min_length=1, max_length=64, description="Anchor product ID")
    required_specs: str = Field(default="", max_length=500, description="Free-text hardware requirements")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")
    category: Optional[str] = Field(default=None, description="Filter by category: laptop|smartphone")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v_lower = v.lower()
        if v_lower not in ("laptop", "smartphone"):
            raise ValueError("category must be 'laptop' or 'smartphone'")
        return v_lower

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("top_k must be between 1 and 20")
        return v


class DriftCheckRequest(BaseModel):
    """POST /api/drift/check request body."""
    simulate_drift: bool = Field(
        default=False,
        description="If True, inject simulated drifted data for demonstration"
    )
    drift_type: str = Field(
        default="market_shift",
        description="Type of drift to simulate: market_shift"
    )
    n_reference_days: int = Field(default=90, ge=7, le=365)
    n_current_days: int = Field(default=30, ge=1, le=90)


class RetrainRequest(BaseModel):
    """POST /api/retrain request body."""
    force: bool = Field(default=False, description="Retrain even if drift not detected")
    reason: str = Field(default="manual_trigger", max_length=256)


# ── Response models ────────────────────────────────────────────────────────────

class HealthCheck(BaseModel):
    """GET /api/health response."""
    status: str
    database: bool
    models_loaded: bool
    tfidf_loaded: bool
    als_loaded: bool
    mlflow: bool
    products_count: int
    version: str


class ProductSpec(BaseModel):
    """Full product details."""
    product_id: str
    category: str
    brand: str
    name: str
    description: Optional[str]
    spec_text: str
    processor: Optional[str]
    ram_gb: Optional[float]
    storage_gb: Optional[float]
    display_type: Optional[str]
    refresh_rate_hz: Optional[int]
    gpu: Optional[str]
    battery_mah: Optional[int]
    camera_mp: Optional[int]
    chipset: Optional[str]
    os: Optional[str]
    price_usd: float
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    release_date: Optional[date] = None
    is_upcoming: Optional[bool] = None
    cold_start: bool = False
    interaction_count: int


class ProductSummary(BaseModel):
    """Lightweight product summary for catalog listing."""
    product_id: str
    category: str
    brand: str
    name: str
    price_usd: float
    spec_text: Optional[str] = None
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    release_date: Optional[date] = None
    is_upcoming: Optional[bool] = None
    cold_start: bool = False
    ram_gb: Optional[float]
    storage_gb: Optional[float]
    display_type: Optional[str]
    refresh_rate_hz: Optional[int]
    interaction_count: int


class UserProfile(BaseModel):
    """User info + interaction summary."""
    user_id: str
    segment: Optional[str]
    interaction_count: int
    created_at: Optional[datetime]


class RecommendationItemResponse(BaseModel):
    """A single recommendation item in the response."""
    product_id: str
    name: str
    category: str
    brand: str
    price_usd: float
    image_url: Optional[str] = None
    product_url: Optional[str] = None
    release_date: Optional[date] = None
    is_upcoming: Optional[bool] = None
    spec_text: str
    als_score: float = Field(description="Normalized ALS collaborative score [0-1]")
    tfidf_score: float = Field(description="Normalized TF-IDF cosine similarity [0-1]")
    spec_match_score: float = Field(description="Specification match score [0-1]")
    final_score: float = Field(description="Blended final hybrid score [0-1]")
    reason: str
    price_category: Optional[str] = None
    newer_than_anchor: Optional[bool] = None
    attribute_scores: dict[str, float] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """POST /api/recommend response."""
    request_id: str
    user_id: str
    product_id: str
    recommendation_mode: str = Field(description="cold_start | hybrid | new_user_fallback")
    anchor_product_name: Optional[str] = None
    interaction_count: Optional[int] = None
    model_version: str
    latency_ms: float
    als_used: bool
    tfidf_used: bool
    spec_matching_used: bool
    recommendations: list[RecommendationItemResponse]
    explanation: dict[str, Any] = Field(default_factory=dict)


class ColumnDriftResult(BaseModel):
    """Drift result for a single feature column."""
    column_name: str
    statistic_name: str
    statistic_value: float
    threshold: float
    drift_detected: bool


class DriftMetricsResponse(BaseModel):
    """GET /api/drift or POST /api/drift/check response."""
    run_id: str
    dataset_drift_detected: bool
    share_drifted_columns: float = Field(description="Fraction of features that drifted [0-1]")
    n_drifted_columns: int
    n_total_columns: int
    drift_threshold: float = Field(description="Threshold for share_drifted_columns [0-1]")
    retraining_recommended: bool
    column_results: list[ColumnDriftResult] = Field(default_factory=list)
    timestamp: datetime
    simulation_mode: bool = False


class ModelStatus(BaseModel):
    """GET /api/model/status response."""
    tfidf_loaded: bool
    als_loaded: bool
    model_version: str
    tfidf_vocabulary_size: int
    tfidf_n_documents: int
    als_rank: int
    als_n_users: int
    als_n_items: int
    training_timestamp: Optional[str]
    tfidf_metrics: dict[str, float] = Field(default_factory=dict)
    als_metrics: dict[str, float] = Field(default_factory=dict)


class RetrainingStatus(BaseModel):
    """POST /api/retrain response / GET /api/retrain/status response."""
    status: str = Field(description="started | running | success | failed | rejected")
    message: str
    run_id: Optional[str] = None
    model_version: Optional[str] = None
    metrics: dict[str, float] = Field(default_factory=dict)
    promoted: bool = False
    timestamp: datetime


class TrainingRunSummary(BaseModel):
    """Summary of a historical training run."""
    id: int
    run_id: Optional[str]
    model_type: str
    model_version: Optional[str]
    rmse: Optional[float]
    precision_at_k: Optional[float]
    recall_at_k: Optional[float]
    ndcg_at_k: Optional[float]
    status: str
    promoted: bool
    timestamp: datetime
