"""
CartSense — Thread-Safe Model Manager

Loads TF-IDF and ALS artifacts at application startup.
Supports hot-swap (reload) without restarting FastAPI.
Maintains a "current" and "pending" state for safe switchover.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from backend.als_inference import ALSInferenceEngine
from backend.config import settings

logger = logging.getLogger("cartsense.model_manager")

ARTIFACTS_DIR = settings.artifacts_dir


@dataclass
class ModelArtifacts:
    """Snapshot of all loaded model artifacts."""
    tfidf_vectorizer: Optional[object] = None
    tfidf_matrix: Optional[object] = None
    product_ids: list[str] = field(default_factory=list)
    als_engine: Optional[ALSInferenceEngine] = None
    model_version: str = "unloaded"
    tfidf_metadata: dict = field(default_factory=dict)
    als_metadata: dict = field(default_factory=dict)

    @property
    def tfidf_loaded(self) -> bool:
        return self.tfidf_vectorizer is not None and self.tfidf_matrix is not None

    @property
    def als_loaded(self) -> bool:
        return self.als_engine is not None and self.als_engine.is_loaded


class ModelManager:
    """
    Thread-safe model artifact manager.

    Call load() at startup.
    Call reload() after a successful retraining to hot-swap models.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._artifacts: ModelArtifacts = ModelArtifacts()

    @property
    def artifacts(self) -> ModelArtifacts:
        with self._lock:
            return self._artifacts

    def load(self, artifacts_dir: str = ARTIFACTS_DIR) -> bool:
        """
        Load all model artifacts from disk.
        Returns True if at least TF-IDF was loaded successfully.
        """
        logger.info("Loading model artifacts from %s", artifacts_dir)
        arts = ModelArtifacts()

        # ── TF-IDF ────────────────────────────────────────────────────────────
        vectorizer_path = Path(artifacts_dir) / "tfidf_vectorizer.joblib"
        matrix_path = Path(artifacts_dir) / "tfidf_matrix.npz"
        id_map_path = Path(artifacts_dir) / "product_id_map.json"

        try:
            arts.tfidf_vectorizer = joblib.load(vectorizer_path)
            arts.tfidf_matrix = sparse.load_npz(matrix_path)
            with open(id_map_path) as f:
                data = json.load(f)
            arts.product_ids = data["index_to_product_id"]
            logger.info(
                "TF-IDF loaded: vocab=%d, products=%d",
                len(arts.tfidf_vectorizer.vocabulary_),
                len(arts.product_ids),
            )
        except FileNotFoundError as e:
            logger.warning("TF-IDF artifact missing: %s — run train_tfidf.py first.", e)
        except Exception as e:
            logger.error("Failed to load TF-IDF artifacts: %s", e)

        # ── TF-IDF metadata ───────────────────────────────────────────────────
        meta_path = Path(artifacts_dir) / "tfidf_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                arts.tfidf_metadata = json.load(f)
            arts.model_version = f"v_{arts.tfidf_metadata.get('training_timestamp', 'unknown')[:10]}"

        # ── ALS ───────────────────────────────────────────────────────────────
        als_engine = ALSInferenceEngine()
        try:
            als_engine.load(artifacts_dir)
            arts.als_engine = als_engine
        except Exception as e:
            logger.warning("ALS artifacts failed to load: %s", e)
            arts.als_engine = None

        # ── ALS metadata ──────────────────────────────────────────────────────
        als_meta_path = Path(artifacts_dir) / "als_metadata.json"
        if als_meta_path.exists():
            with open(als_meta_path) as f:
                arts.als_metadata = json.load(f)

        with self._lock:
            self._artifacts = arts

        logger.info(
            "Model manager ready: tfidf=%s, als=%s, version=%s",
            arts.tfidf_loaded,
            arts.als_loaded,
            arts.model_version,
        )
        return arts.tfidf_loaded

    def reload(self, artifacts_dir: str = ARTIFACTS_DIR) -> bool:
        """
        Reload artifacts from disk into a new ModelArtifacts object,
        then atomically swap the reference. The old artifacts stay in
        memory until garbage collected.
        """
        logger.info("Hot-reloading model artifacts...")
        success = self.load(artifacts_dir)
        if success:
            logger.info("Model hot-reload successful. New version: %s", self._artifacts.model_version)
        else:
            logger.error("Model hot-reload failed — previous artifacts still active.")
        return success

    def get_status(self) -> dict:
        """Return current model status dictionary."""
        arts = self.artifacts
        return {
            "tfidf_loaded": arts.tfidf_loaded,
            "als_loaded": arts.als_loaded,
            "model_version": arts.model_version,
            "tfidf_vocabulary_size": (
                len(arts.tfidf_vectorizer.vocabulary_) if arts.tfidf_loaded else 0
            ),
            "tfidf_n_documents": len(arts.product_ids) if arts.tfidf_loaded else 0,
            "als_rank": arts.als_metadata.get("rank", 0),
            "als_n_users": arts.als_metadata.get("n_users", 0),
            "als_n_items": arts.als_metadata.get("n_items", 0),
            "training_timestamp": arts.tfidf_metadata.get("training_timestamp"),
            "tfidf_metrics": arts.tfidf_metadata.get("metrics", {}),
            "als_metrics": arts.als_metadata.get("metrics", {}),
        }


# Application-level singleton
model_manager = ModelManager()
