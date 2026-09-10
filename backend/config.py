"""
CartSense — Centralized Application Configuration
All settings are read from environment variables with sensible defaults.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://cartsense_user:cartsense_pass@postgres_db:5432/cartsense",
        )
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
    )
    tfidf_model_name: str = field(
        default_factory=lambda: os.getenv("TFIDF_MODEL_NAME", "CartSense_TFIDF")
    )
    als_model_name: str = field(
        default_factory=lambda: os.getenv("ALS_MODEL_NAME", "CartSense_ALS")
    )

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # ── Hybrid scoring weights ────────────────────────────────────────────────
    # alpha: collaborative (ALS) weight in hybrid blend
    alpha: float = field(default_factory=lambda: float(os.getenv("ALPHA", "0.65")))
    # beta: specification matching boost weight
    beta: float = field(default_factory=lambda: float(os.getenv("BETA", "0.20")))

    # ── Recommendation ────────────────────────────────────────────────────────
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))

    # ── ALS Hyperparameters ───────────────────────────────────────────────────
    als_rank: int = field(default_factory=lambda: int(os.getenv("ALS_RANK", "20")))
    als_max_iter: int = field(default_factory=lambda: int(os.getenv("ALS_MAX_ITER", "15")))
    als_reg_param: float = field(
        default_factory=lambda: float(os.getenv("ALS_REG_PARAM", "0.1"))
    )
    als_seed: int = field(default_factory=lambda: int(os.getenv("ALS_SEED", "42")))

    # ── Drift Detection ───────────────────────────────────────────────────────
    # Fraction of features that must drift to declare dataset drift (0-1 scale)
    # 0.5 means ≥50% of monitored features show statistically significant shift
    drift_threshold: float = field(
        default_factory=lambda: float(os.getenv("DRIFT_THRESHOLD", "0.5"))
    )

    # ── Artifact Storage ──────────────────────────────────────────────────────
    artifacts_dir: str = field(
        default_factory=lambda: os.getenv("ARTIFACTS_DIR", "/app/artifacts")
    )

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")


# Singleton settings object used across all modules
settings = Settings()
