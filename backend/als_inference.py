"""
CartSense — ALS Inference Engine

Loads pre-computed PySpark ALS latent factors (saved by train_als.py)
and performs fast dot-product scoring at inference time.

NO Spark session is started during inference.
The user factor × item factor matrix product is computed with numpy.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("cartsense.als_inference")

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/app/artifacts")


class ALSInferenceEngine:
    """
    Fast ALS recommender using pre-computed latent factors.

    Usage:
        engine = ALSInferenceEngine()
        engine.load(artifacts_dir)
        scores = engine.predict_user_scores("U0001")
    """

    def __init__(self) -> None:
        self._user_factors: Optional[np.ndarray] = None   # shape: (n_users, rank)
        self._item_factors: Optional[np.ndarray] = None   # shape: (n_items, rank)
        self._user_to_int: dict[str, int] = {}
        self._int_to_item: dict[int, str] = {}
        self._item_to_int: dict[str, int] = {}
        self._item_ids_sorted: list[str] = []             # ordered list of product_id
        self._rank: int = 0
        self._loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, artifacts_dir: str = ARTIFACTS_DIR) -> None:
        """Load factor matrices and ID maps from disk."""
        artifacts_path = Path(artifacts_dir)

        user_factors_path = artifacts_path / "als_user_factors.parquet"
        item_factors_path = artifacts_path / "als_item_factors.parquet"
        id_maps_path = artifacts_path / "als_id_maps.json"

        for p in [user_factors_path, item_factors_path, id_maps_path]:
            if not p.exists():
                logger.warning("ALS artifact missing: %s — ALS inference unavailable.", p)
                self._loaded = False
                return

        # Load factor DataFrames
        user_df = pd.read_parquet(user_factors_path)
        item_df = pd.read_parquet(item_factors_path)

        # Load ID maps
        with open(id_maps_path) as f:
            maps = json.load(f)
        self._user_to_int = {uid: int(i) for uid, i in maps["user_to_int"].items()}
        self._int_to_item = {int(k): v for k, v in maps["int_to_item"].items()}
        self._item_to_int = {pid: int(i) for pid, i in maps["item_to_int"].items()}

        # Build sorted item factor matrix (items sorted by integer index)
        item_df = item_df.set_index("id")
        item_ids_int = sorted(item_df.index.tolist())
        self._item_ids_sorted = [self._int_to_item[i] for i in item_ids_int]
        self._item_factors = np.vstack(
            [np.array(item_df.loc[i, "features"], dtype=np.float32) for i in item_ids_int]
        )

        # Build user factor lookup
        user_df = user_df.set_index("id")
        self._user_factors_df = user_df

        self._rank = self._item_factors.shape[1]
        self._loaded = True
        logger.info(
            "ALS inference engine loaded: %d users, %d items, rank=%d",
            len(self._user_to_int),
            len(self._item_ids_sorted),
            self._rank,
        )

    def predict_user_scores(self, user_id: str) -> dict[str, float]:
        """
        Return a dict {product_id: als_score} for all items.

        For unknown users, returns an empty dict (caller should fall back).
        Scores are dot products of user × item latent factors (non-negative ALS).
        """
        if not self._loaded:
            return {}

        user_int = self._user_to_int.get(user_id)
        if user_int is None:
            logger.debug("Unknown user_id=%s — ALS has no factors for this user.", user_id)
            return {}

        if user_int not in self._user_factors_df.index:
            return {}

        u_vec = np.array(self._user_factors_df.loc[user_int, "features"], dtype=np.float32)
        raw_scores = self._item_factors @ u_vec  # shape: (n_items,)

        return {
            pid: float(raw_scores[idx])
            for idx, pid in enumerate(self._item_ids_sorted)
        }

    def get_item_factor(self, product_id: str) -> Optional[np.ndarray]:
        """Return the latent factor vector for a product, or None if unknown."""
        if not self._loaded:
            return None
        item_int = self._item_to_int.get(product_id)
        if item_int is None:
            return None
        item_df = getattr(self, "_user_factors_df", None)
        # Use item factor matrix
        idx = self._item_ids_sorted.index(product_id) if product_id in self._item_ids_sorted else -1
        if idx < 0:
            return None
        return self._item_factors[idx]

    def normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalize ALS scores to [0, 1]."""
        if not scores:
            return scores
        vals = np.array(list(scores.values()), dtype=np.float32)
        min_v, max_v = vals.min(), vals.max()
        if max_v - min_v < 1e-9:
            return {k: 0.5 for k in scores}
        return {k: float((v - min_v) / (max_v - min_v)) for k, v in scores.items()}
