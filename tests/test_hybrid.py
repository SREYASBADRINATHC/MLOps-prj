"""
CartSense — Hybrid Engine Tests
Tests the hybrid routing and scoring logic with mock artifacts.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer


def _build_mock_products():
    """Build a small mock product catalog."""
    data = [
        {
            "product_id": "L0001", "category": "laptop", "brand": "Dell",
            "name": "Dell XPS 15 OLED", "spec_text": "Dell XPS 15 OLED 32GB RAM 1TB SSD Intel Core Ultra 9 120Hz",
            "ram_gb": 32.0, "storage_gb": 1024.0, "display_type": "OLED",
            "refresh_rate_hz": 120, "price_usd": 1999.0, "battery_mah": 86,
            "camera_mp": None, "processor": "Intel Core Ultra 9", "chipset": None, "os": "Windows 11",
            "gpu": "RTX 4070", "description": "Premium laptop",
            "interaction_count": 5,
        },
        {
            "product_id": "L0002", "category": "laptop", "brand": "Lenovo",
            "name": "ThinkPad X1 Carbon", "spec_text": "ThinkPad X1 Carbon 32GB RAM 1TB SSD Intel Core Ultra 7 IPS 60Hz",
            "ram_gb": 32.0, "storage_gb": 1024.0, "display_type": "IPS",
            "refresh_rate_hz": 60, "price_usd": 1799.0, "battery_mah": 57,
            "camera_mp": None, "processor": "Intel Core Ultra 7", "chipset": None, "os": "Windows 11",
            "gpu": None, "description": "Business laptop",
            "interaction_count": 3,
        },
        {
            "product_id": "L0003", "category": "laptop", "brand": "ASUS",
            "name": "ROG Zephyrus G16", "spec_text": "ASUS ROG Zephyrus G16 32GB RAM 2TB SSD OLED 165Hz RTX 4080",
            "ram_gb": 32.0, "storage_gb": 2048.0, "display_type": "OLED",
            "refresh_rate_hz": 165, "price_usd": 2499.0, "battery_mah": 90,
            "camera_mp": None, "processor": "Intel Core Ultra 9", "chipset": None, "os": "Windows 11",
            "gpu": "RTX 4080", "description": "Gaming laptop",
            "interaction_count": 0,  # cold-start
        },
        {
            "product_id": "S0001", "category": "smartphone", "brand": "Samsung",
            "name": "Galaxy S24 Ultra", "spec_text": "Samsung Galaxy S24 Ultra Snapdragon 8 Gen 3 12GB RAM 256GB AMOLED 120Hz",
            "ram_gb": 12.0, "storage_gb": 256.0, "display_type": "AMOLED",
            "refresh_rate_hz": 120, "price_usd": 1299.0, "battery_mah": 5000,
            "camera_mp": 200, "processor": None, "chipset": "Snapdragon 8 Gen 3", "os": "Android 14",
            "gpu": None, "description": "Flagship phone",
            "interaction_count": 10,
        },
    ]
    return pd.DataFrame(data)


def _build_mock_tfidf(products_df):
    """Fit a tiny TF-IDF on mock products."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(products_df["spec_text"].tolist())
    return vectorizer, matrix, products_df["product_id"].tolist()


class TestHybridEngine:
    def setup_method(self):
        self.products_df = _build_mock_products()
        self.vectorizer, self.matrix, self.product_ids = _build_mock_tfidf(self.products_df)

    def _make_engine(self, als_engine=None, alpha=0.65, beta=0.20):
        from backend.hybrid_engine import HybridRecommendationEngine
        return HybridRecommendationEngine(
            tfidf_vectorizer=self.vectorizer,
            tfidf_matrix=self.matrix,
            product_ids=self.product_ids,
            als_engine=als_engine,
            products_df=self.products_df,
            alpha=alpha,
            beta=beta,
            model_version="test_v1",
        )

    def test_cold_start_routing(self):
        """Cold-start product (interaction_count=0) must route to cold_start mode."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0003",  # interaction_count=0
            top_k=2,
            interaction_count=0,
            user_interaction_count=5,
        )
        assert result.recommendation_mode == "cold_start"
        assert result.als_used is False
        assert result.tfidf_used is True

    def test_new_user_fallback(self):
        """New user (0 interactions) must route to new_user_fallback."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U9999",
            product_id="L0001",
            top_k=2,
            interaction_count=5,
            user_interaction_count=0,
        )
        assert result.recommendation_mode == "new_user_fallback"
        assert result.als_used is False

    def test_warm_product_no_als_still_works(self):
        """Warm product without ALS engine should fall back gracefully."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0001",
            top_k=2,
            interaction_count=5,
            user_interaction_count=5,
        )
        # Without ALS, falls back to content only
        assert result.tfidf_used is True
        assert len(result.recommendations) > 0

    def test_category_filtering(self):
        """Recommendations must only contain same-category products."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0001",  # laptop
            top_k=3,
            interaction_count=5,
            user_interaction_count=5,
        )
        for rec in result.recommendations:
            assert rec.category == "laptop", f"Expected laptop, got {rec.category}"

    def test_anchor_product_excluded(self):
        """Anchor product must never appear in its own recommendations."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0001",
            top_k=3,
            interaction_count=5,
            user_interaction_count=5,
        )
        rec_ids = [r.product_id for r in result.recommendations]
        assert "L0001" not in rec_ids

    def test_scores_in_range(self):
        """All scores must be in [0, 1]."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0001",
            top_k=2,
            interaction_count=5,
            user_interaction_count=5,
        )
        for rec in result.recommendations:
            assert 0.0 <= rec.tfidf_score <= 1.0
            assert 0.0 <= rec.final_score <= 1.0

    def test_spec_matching_used(self):
        """When required_specs is provided, spec_matching_used should be True."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0001",
            required_specs="32GB RAM OLED",
            top_k=2,
            interaction_count=5,
            user_interaction_count=5,
        )
        assert result.spec_matching_used is True

    def test_latency_is_positive(self):
        """Latency measurement must be a positive number."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001",
            product_id="L0001",
            top_k=2,
            interaction_count=5,
            user_interaction_count=5,
        )
        assert result.latency_ms > 0

    def test_model_version_propagated(self):
        """Model version from engine constructor must appear in result."""
        engine = self._make_engine(als_engine=None)
        result = engine.recommend(
            user_id="U0001", product_id="L0001", top_k=1,
            interaction_count=5, user_interaction_count=5,
        )
        assert result.model_version == "test_v1"
