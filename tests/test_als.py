"""
CartSense — ALS Inference Engine Tests
Tests the factor loading and scoring logic.
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.als_inference import ALSInferenceEngine


def _make_mock_factors(n_users: int = 3, n_items: int = 5, rank: int = 4):
    """Create mock ALS factor files in a temp directory."""
    np.random.seed(42)

    user_ids_int = list(range(n_users))
    item_ids_int = list(range(n_items))

    user_factors = pd.DataFrame({
        "id": user_ids_int,
        "features": [np.random.rand(rank).tolist() for _ in user_ids_int],
    })
    item_factors = pd.DataFrame({
        "id": item_ids_int,
        "features": [np.random.rand(rank).tolist() for _ in item_ids_int],
    })

    user_to_int = {f"U{i:04d}": i for i in user_ids_int}
    int_to_item = {i: f"P{i:04d}" for i in item_ids_int}
    item_to_int = {f"P{i:04d}": i for i in item_ids_int}

    return user_factors, item_factors, user_to_int, int_to_item, item_to_int


@pytest.fixture
def als_engine_with_data(tmp_path):
    """Create a loaded ALS engine with mock data."""
    user_factors, item_factors, user_to_int, int_to_item, item_to_int = _make_mock_factors()

    user_factors.to_parquet(tmp_path / "als_user_factors.parquet", index=False)
    item_factors.to_parquet(tmp_path / "als_item_factors.parquet", index=False)

    maps = {
        "user_to_int": user_to_int,
        "int_to_item": {str(k): v for k, v in int_to_item.items()},
        "item_to_int": item_to_int,
    }
    with open(tmp_path / "als_id_maps.json", "w") as f:
        json.dump(maps, f)

    engine = ALSInferenceEngine()
    engine.load(str(tmp_path))
    return engine


class TestALSInferenceEngine:
    def test_loads_successfully(self, als_engine_with_data):
        assert als_engine_with_data.is_loaded is True

    def test_predict_known_user(self, als_engine_with_data):
        scores = als_engine_with_data.predict_user_scores("U0000")
        assert isinstance(scores, dict)
        assert len(scores) > 0
        # All scores should be product IDs
        for pid in scores:
            assert pid.startswith("P")

    def test_predict_unknown_user_returns_empty(self, als_engine_with_data):
        scores = als_engine_with_data.predict_user_scores("U9999")
        assert scores == {}

    def test_scores_are_floats(self, als_engine_with_data):
        scores = als_engine_with_data.predict_user_scores("U0001")
        for v in scores.values():
            assert isinstance(v, float)

    def test_normalize_scores(self, als_engine_with_data):
        raw = {"P0001": 5.0, "P0002": 10.0, "P0003": 0.0}
        normalized = als_engine_with_data.normalize_scores(raw)
        assert normalized["P0003"] == 0.0
        assert normalized["P0002"] == 1.0
        assert 0.0 < normalized["P0001"] < 1.0

    def test_empty_artifacts_dir_does_not_crash(self, tmp_path):
        """Loading from directory without factors should not crash, just set is_loaded=False."""
        engine = ALSInferenceEngine()
        engine.load(str(tmp_path))
        assert engine.is_loaded is False

    def test_predict_on_unloaded_engine(self, tmp_path):
        """Predicting from unloaded engine should return empty dict."""
        engine = ALSInferenceEngine()
        engine.load(str(tmp_path))
        scores = engine.predict_user_scores("U0000")
        assert scores == {}
