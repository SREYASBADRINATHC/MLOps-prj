"""
CartSense — Hybrid Recommendation Engine

Implements the core routing and scoring logic:

  NEW ITEM (interaction_count == 0):
      Cold-start path: TF-IDF + Cosine Similarity + Spec Matching
      ALS is SKIPPED — no behavioral data exists.

  EXISTING ITEM (interaction_count > 0) + KNOWN USER:
      Hybrid path: alpha * ALS + (1-alpha) * TF-IDF + beta * SpecBoost

  NEW USER (no interaction history):
      Fallback: TF-IDF + Spec Matching (labeled "new_user_fallback")

# Configuration weights (using existing config names for compatibility):
#   ALPHA = ALS weight
#   TFIDF_WEIGHT = 1 - ALPHA
#   SPEC_BOOST_WEIGHT = BETA (additive boost up to a max of 1.0)
#
#   For cold-start:
#   TFIDF_WEIGHT_COLD_START = 1 - BETA
#   SPEC_BOOST_WEIGHT = BETA

Scores are normalized before blending to handle different raw score ranges.
Final formula:
    FinalScore = (ALPHA * NormalizedALS) + ((1 - ALPHA) * TF-IDF) + (BETA * SpecBoost)

All normalization is min-max within the candidate set.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from backend.spec_matcher import parse_spec_query, score_product_against_query

logger = logging.getLogger("cartsense.hybrid_engine")


@dataclass
class RecommendationItem:
    """A single recommendation result with full score breakdown."""
    product_id: str
    name: str
    category: str
    brand: str
    price_usd: float
    image_url: Optional[str]
    product_url: Optional[str]
    release_date: Optional[str]
    is_upcoming: Optional[bool]
    spec_text: str
    als_score: float
    tfidf_score: float
    spec_match_score: float
    final_score: float
    reason: str
    price_category: Optional[str]
    newer_than_anchor: Optional[bool]
    attribute_scores: dict = field(default_factory=dict)


@dataclass
class RecommendationResult:
    """Full recommendation response."""
    user_id: str
    product_id: str
    anchor_product_name: str
    interaction_count: int
    recommendation_mode: str   # cold_start | hybrid | new_user_fallback
    model_version: str
    latency_ms: float
    als_used: bool
    tfidf_used: bool
    spec_matching_used: bool
    recommendations: list[RecommendationItem]
    explanation: dict = field(default_factory=dict)


class HybridRecommendationEngine:
    """
    CartSense Hybrid Recommendation Engine.

    Requires:
        - TF-IDF vectorizer (fitted sklearn TfidfVectorizer)
        - TF-IDF matrix (scipy sparse, shape=[n_products, vocab_size])
        - Product ID list (ordered list matching TF-IDF matrix rows)
        - ALS inference engine (ALSInferenceEngine instance, may be None)
        - Products DataFrame (from PostgreSQL, used for spec matching + category filter)
    """

    def __init__(
        self,
        tfidf_vectorizer,
        tfidf_matrix,
        product_ids: list[str],
        als_engine,              # ALSInferenceEngine | None
        products_df: pd.DataFrame,
        alpha: float = 0.65,
        beta: float = 0.20,
        model_version: str = "v1.0",
    ) -> None:
        self.vectorizer = tfidf_vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.product_ids = product_ids
        self.als_engine = als_engine
        self.products_df = products_df.copy()
        self.alpha = alpha
        self.beta = beta
        self.model_version = model_version

        # Build fast lookup structures
        self._pid_to_idx: dict[str, int] = {pid: i for i, pid in enumerate(product_ids)}
        self._pid_to_row: dict[str, dict] = {
            str(row["product_id"]): row.to_dict()
            for _, row in products_df.iterrows()
        }
        logger.info(
            "HybridRecommendationEngine initialized: %d products, alpha=%.2f, beta=%.2f",
            len(product_ids),
            alpha,
            beta,
        )

    def recommend(
        self,
        user_id: str,
        product_id: str,
        required_specs: str = "",
        top_k: int = 5,
        interaction_count: int = -1,
        user_interaction_count: int = -1,
    ) -> RecommendationResult:
        """
        Main recommendation entry point.

        Args:
            user_id: The requesting user's ID (string)
            product_id: The anchor product ID
            required_specs: Free-text hardware requirements
            top_k: Number of recommendations to return
            interaction_count: How many interactions the anchor product has had
                               (-1 = let engine query; caller should preload)
            user_interaction_count: How many interactions this user has had
                                    (-1 = treat as known user)
        """
        t0 = time.perf_counter()

        # ── Determine recommendation mode ─────────────────────────────────────
        item_is_cold_start = interaction_count == 0
        user_is_new = user_interaction_count == 0

        if item_is_cold_start:
            mode = "cold_start"
        elif user_is_new:
            mode = "new_user_fallback"
        else:
            mode = "hybrid"

        logger.info(
            "Recommend: user=%s product=%s mode=%s item_interactions=%d",
            user_id,
            product_id,
            mode,
            interaction_count,
        )

        # ── Get anchor product info ───────────────────────────────────────────
        anchor = self._pid_to_row.get(product_id)
        if anchor is None:
            raise ValueError(f"Unknown product_id: {product_id}")

        anchor_category = str(anchor.get("category", ""))
        anchor_idx = self._pid_to_idx.get(product_id)
        if anchor_idx is None:
            raise ValueError(f"Product {product_id} not in TF-IDF index.")

        # Extract anchor price and release date for candidate comparison
        anchor_price = _to_float(anchor.get("price_usd"))
        anchor_release = _parse_release(anchor.get("release_date"))


        # ── Get candidates (same category, exclude anchor) ────────────────────
        same_cat_pids = [
            pid for pid in self.product_ids
            if pid != product_id
            and str(self._pid_to_row.get(pid, {}).get("category", "")) == anchor_category
        ]
        if not same_cat_pids:
            same_cat_pids = [pid for pid in self.product_ids if pid != product_id]

        same_cat_indices = [self._pid_to_idx[pid] for pid in same_cat_pids]

        # ── TF-IDF scores (always computed) ───────────────────────────────────
        tfidf_raw = cosine_similarity(
            self.tfidf_matrix[anchor_idx], self.tfidf_matrix[same_cat_indices]
        ).flatten()
        tfidf_norm = self._minmax_normalize(tfidf_raw)
        tfidf_scores: dict[str, float] = {
            pid: float(tfidf_norm[i]) for i, pid in enumerate(same_cat_pids)
        }

        # ── ALS scores ────────────────────────────────────────────────────────
        als_used = False
        als_scores: dict[str, float] = {}

        if mode == "hybrid" and self.als_engine is not None and self.als_engine.is_loaded:
            raw_als = self.als_engine.predict_user_scores(user_id)
            if raw_als:
                # Restrict to same-category candidates
                raw_als_filtered = {pid: raw_als.get(pid, 0.0) for pid in same_cat_pids}
                als_scores = self.als_engine.normalize_scores(raw_als_filtered)
                als_used = True
            else:
                logger.info("ALS returned no scores for user=%s — falling back to TF-IDF only", user_id)
                mode = "new_user_fallback"

        # ── Specification matching ────────────────────────────────────────────
        spec_scores: dict[str, float] = {}
        spec_used = False
        if required_specs and required_specs.strip():
            req = parse_spec_query(required_specs)
            if req.has_requirements:
                spec_used = True
                for pid in same_cat_pids:
                    product_row = self._pid_to_row.get(pid, {})
                    result = score_product_against_query(product_row, required_specs)
                    spec_scores[pid] = result["overall_score"]

        # ── Combine scores ────────────────────────────────────────────────────
        final_scores: dict[str, float] = {}
        per_product_breakdown: dict[str, dict] = {}

        for pid in same_cat_pids:
            tf = tfidf_scores.get(pid, 0.0)
            als = als_scores.get(pid, 0.0)
            spec = spec_scores.get(pid, 0.0)

            if mode == "hybrid" and als_used:
                combined = self.alpha * als + (1 - self.alpha) * tf + self.beta * spec
            else:
                # Cold-start or new-user: TF-IDF is primary, spec is boost
                combined = (1 - self.beta) * tf + self.beta * spec if spec_used else tf

            final_scores[pid] = float(np.clip(combined, 0.0, 1.0))
            per_product_breakdown[pid] = {
                "als": round(als, 4),
                "tfidf": round(tf, 4),
                "spec": round(spec, 4),
                "final": round(float(np.clip(combined, 0.0, 1.0)), 4),
            }

        # ── Top-K selection ───────────────────────────────────────────────────
        top_pids = sorted(final_scores, key=lambda p: final_scores[p], reverse=True)[:top_k]

        # ── Build recommendation items ────────────────────────────────────────
        recommendations: list[RecommendationItem] = []
        for pid in top_pids:
            prod = self._pid_to_row.get(pid, {})
            breakdown = per_product_breakdown[pid]
            spec_attr_scores: dict = {}
            if pid in spec_scores and required_specs:
                from backend.spec_matcher import parse_product_specs, compute_spec_match_score, parse_spec_query as _pq
                result = score_product_against_query(prod, required_specs)
                spec_attr_scores = result.get("attribute_scores", {})

            reason = self._build_reason(
                pid, mode, breakdown, spec_attr_scores, prod, required_specs
            )

            recommendations.append(
                RecommendationItem(
                    product_id=pid,
                    name=str(prod.get("name", "")),
                    category=str(prod.get("category", "")),
                    brand=str(prod.get("brand", "")),
                    price_usd=float(prod.get("price_usd", 0)),
                    image_url=_clean_optional_str(prod.get("image_url")),
                    product_url=_clean_optional_str(prod.get("product_url")),
                    release_date=_format_release(prod.get("release_date")),
                    is_upcoming=_to_optional_bool(prod.get("is_upcoming")),
                    spec_text=str(prod.get("spec_text", "")),
                    als_score=breakdown["als"],
                    tfidf_score=breakdown["tfidf"],
                    spec_match_score=breakdown["spec"],
                    final_score=breakdown["final"],
                    reason=reason,
                    price_category=_price_category(anchor_price, _to_float(prod.get("price_usd"))),
                    newer_than_anchor=_is_newer_product(anchor_release, _parse_release(prod.get("release_date"))),
                    attribute_scores=spec_attr_scores,
                )
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        explanation = {
            "alpha": self.alpha,
            "beta": self.beta,
            "formula": (
                f"FinalScore = {self.alpha}*(ALS) + {1-self.alpha:.2f}*(TF-IDF) + {self.beta}*(SpecBoost)"
                if mode == "hybrid"
                else f"FinalScore = {1-self.beta:.2f}*(TF-IDF) + {self.beta}*(SpecBoost) (ALS skipped)"
            ),
            "als_skipped_reason": (
                None if als_used else (
                    "Item has zero interaction history (cold-start)"
                    if mode == "cold_start"
                    else "User has no interaction history"
                    if mode == "new_user_fallback"
                    else "ALS model not loaded"
                )
            ),
        }

        return RecommendationResult(
            user_id=user_id,
            product_id=product_id,
            anchor_product_name=str(anchor.get("name", product_id)),
            interaction_count=max(interaction_count, 0),
            recommendation_mode=mode,
            model_version=self.model_version,
            latency_ms=latency_ms,
            als_used=als_used,
            tfidf_used=True,
            spec_matching_used=spec_used,
            recommendations=recommendations,
            explanation=explanation,
        )

    @staticmethod
    def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
        """Normalize a 1-D array to [0, 1] using min-max scaling."""
        min_v, max_v = arr.min(), arr.max()
        if max_v - min_v < 1e-9:
            return np.full_like(arr, 0.5, dtype=float)
        return (arr - min_v) / (max_v - min_v)

    @staticmethod
    def _build_reason(
        pid: str,
        mode: str,
        breakdown: dict,
        spec_attr_scores: dict,
        product: dict,
        required_specs: str,
    ) -> str:
        """Build a human-readable explanation for the recommendation."""
        parts = []
        if mode == "hybrid":
            if breakdown["als"] > 0.6:
                parts.append("similar users have interacted with this product")
            if breakdown["tfidf"] > 0.6:
                parts.append("similar hardware specifications")
        elif mode == "cold_start":
            parts.append("content-based matching (item is new, no behavioral data)")
            if breakdown["tfidf"] > 0.7:
                parts.append("high specification similarity to anchor product")
        elif mode == "new_user_fallback":
            parts.append("content-based matching (new user, no behavioral data)")

        if spec_attr_scores and required_specs:
            matched = [attr for attr, v in spec_attr_scores.items() if v >= 0.75]
            if matched:
                parts.append(f"matches required: {', '.join(matched)}")

        return "Recommended because: " + "; ".join(parts) if parts else "Specification and behavior similarity match"


def _clean_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_release(value: Any):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime().date()
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _format_release(value: Any) -> Optional[str]:
    parsed = _parse_release(value)
    return parsed.isoformat() if parsed else None


def _price_category(anchor_price: Optional[float], candidate_price: Optional[float]) -> Optional[str]:
    if anchor_price is None or candidate_price is None or anchor_price <= 0:
        return None
    lower = anchor_price * 0.90
    upper = anchor_price * 1.10
    if candidate_price < lower:
        return "cheaper"
    if candidate_price > upper:
        return "premium"
    return "similar_price"


def _is_newer_product(anchor_release, candidate_release) -> Optional[bool]:
    if anchor_release is None or candidate_release is None:
        return None
    return candidate_release > anchor_release
