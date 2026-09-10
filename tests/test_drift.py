"""
CartSense — Drift Monitor Tests
Tests the Evidently-based drift detection logic.
"""
import pytest
import numpy as np
import pandas as pd


def _make_reference_df(n=100, seed=42):
    """Baseline (training-era) distribution."""
    np.random.seed(seed)
    return pd.DataFrame({
        "ram_gb": np.random.choice([8, 16], n),
        "price_usd": np.random.normal(900, 200, n).clip(400, 1800),
        "refresh_rate_hz": np.random.choice([60, 90], n, p=[0.8, 0.2]),
        "storage_gb": np.random.choice([256, 512], n),
        "display_encoded": np.random.choice([0.0, 1.0], n, p=[0.6, 0.4]),
        "battery_mah": np.random.normal(4000, 300, n).clip(3000, 5000),
    })


def _make_drifted_df(n=100, seed=99):
    """Drifted (new market) distribution."""
    np.random.seed(seed)
    return pd.DataFrame({
        "ram_gb": np.random.choice([32, 48, 64], n),
        "price_usd": np.random.normal(2000, 300, n).clip(1200, 4000),
        "refresh_rate_hz": np.random.choice([120, 144, 165], n),
        "storage_gb": np.random.choice([1024, 2048], n),
        "display_encoded": np.random.choice([1.5, 2.0], n, p=[0.3, 0.7]),
        "battery_mah": np.random.normal(5200, 400, n).clip(4500, 6500),
    })


class TestDriftDetection:
    def test_drift_detected_on_shifted_data(self):
        """Strongly shifted distributions should be detected as drifted."""
        from backend.drift_monitor import run_evidently_drift

        ref = _make_reference_df()
        cur = _make_drifted_df()

        result = run_evidently_drift(
            reference_df=ref,
            current_df=cur,
            drift_threshold=0.5,
            run_id="test_run_001",
            simulation_mode=True,
        )
        # With a major shift, drift should be detected
        assert result["dataset_drift_detected"] is True
        assert result["share_drifted_columns"] >= 0.5
        assert result["n_total_columns"] > 0

    def test_no_drift_on_identical_data(self):
        """Identical distributions should not be flagged as drifted."""
        from backend.drift_monitor import run_evidently_drift

        ref = _make_reference_df(seed=1)
        cur = _make_reference_df(seed=2)  # same distribution, different seed

        result = run_evidently_drift(
            reference_df=ref,
            current_df=cur,
            drift_threshold=0.5,
            run_id="test_run_002",
        )
        # Identical distributions should show low drift share
        assert result["share_drifted_columns"] < 0.5 or not result["dataset_drift_detected"]

    def test_drift_result_structure(self):
        """Result dict must contain all required keys."""
        from backend.drift_monitor import run_evidently_drift

        ref = _make_reference_df()
        cur = _make_drifted_df()
        result = run_evidently_drift(ref, cur, 0.5, "test_run_003")

        required_keys = [
            "run_id", "dataset_drift_detected", "share_drifted_columns",
            "n_drifted_columns", "n_total_columns", "drift_threshold",
            "retraining_recommended", "column_results", "timestamp",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_display_encoding(self):
        """Display type encoding must map OLED/AMOLED to 2.0, IPS to 1.0, others to 0.0."""
        from backend.drift_monitor import _encode_display

        assert _encode_display("OLED") == 2.0
        assert _encode_display("AMOLED") == 2.0
        assert _encode_display("IPS") == 1.0
        assert _encode_display("LCD") == 0.0
        assert _encode_display(None) == 0.0

    def test_generate_drifted_distribution(self):
        """Generated drifted distribution should have higher RAM and refresh rate."""
        from backend.drift_monitor import _generate_drifted_distribution

        ref = _make_reference_df()
        drifted = _generate_drifted_distribution(ref)

        assert drifted["ram_gb"].mean() > ref["ram_gb"].mean()
        assert drifted["refresh_rate_hz"].mean() > ref["refresh_rate_hz"].mean()
