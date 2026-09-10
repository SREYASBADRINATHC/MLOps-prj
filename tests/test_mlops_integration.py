import pytest
from fastapi.testclient import TestClient
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./cartsense_test.db")
os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
os.environ.setdefault("ARTIFACTS_DIR", "./artifacts")

@pytest.fixture(scope="module")
def client():
    from backend.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c

def test_health_dependency_state(client):
    resp = client.get("/api/health")
    data = resp.json()
    assert "status" in data
    # Real test of the health dependency state logic added in main.py
    if data.get("database") and data.get("mlflow") and data.get("tfidf_loaded"):
        assert data["status"] == "healthy"
    else:
        assert data["status"] == "degraded"

def test_drift_threshold_configuration(client):
    resp = client.get("/api/drift")
    if resp.status_code == 200:
        data = resp.json()
        assert data["drift_threshold"] == 0.75

def test_cold_start_and_hybrid_recommendation(client):
    # Find a product with 0 interactions and one with > 0 interactions
    resp_prod = client.get("/api/products?limit=500")
    if resp_prod.status_code == 503:
        pytest.skip("Models not loaded or db not seeded")
    products = resp_prod.json()
    
    cold_start_product = next((p for p in products if p["interaction_count"] == 0), None)
    hybrid_product = next((p for p in products if p["interaction_count"] > 0), None)
    
    user_id = "U0001"
    
    # 1. Test Cold Start
    if cold_start_product:
        resp = client.post("/api/recommend", json={
            "user_id": user_id,
            "product_id": cold_start_product["product_id"],
            "top_k": 5
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendation_mode"] == "cold_start"
        assert not data["als_used"]
        assert data["tfidf_used"]
        assert len(data["recommendations"]) > 0

    # 2. Test Hybrid
    if hybrid_product:
        resp = client.post("/api/recommend", json={
            "user_id": user_id,
            "product_id": hybrid_product["product_id"],
            "top_k": 5
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendation_mode"] == "hybrid"

def test_unknown_user_fallback(client):
    resp_prod = client.get("/api/products?limit=10")
    if resp_prod.status_code == 503:
        pytest.skip("Models not loaded or db not seeded")
    
    products = resp_prod.json()
    if not products:
        pytest.skip("No products found")
        
    pid = products[0]["product_id"]
    
    resp = client.post("/api/recommend", json={
        "user_id": "UNKNOWN_USER_999",
        "product_id": pid,
        "top_k": 5
    })
    
    # System should not crash
    assert resp.status_code == 200
    data = resp.json()
    # It might be new_user_fallback or cold_start if the product has 0 interactions.
    assert data["recommendation_mode"] in ("new_user_fallback", "cold_start")
    assert len(data["recommendations"]) > 0
    assert not data["als_used"]

def test_drift_threshold_configuration():
    import os
    # Read docker-compose.yml to ensure DRIFT_THRESHOLD is set to 0.75
    compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
    if os.path.exists(compose_path):
        with open(compose_path, "r") as f:
            content = f.read()
            assert 'DRIFT_THRESHOLD: "0.75"' in content or "DRIFT_THRESHOLD: 0.75" in content

def test_mlflow_connectivity_configuration():
    import os
    from backend.config import Settings
    # Create fresh settings instance without env override to check default
    original = os.environ.get("MLFLOW_TRACKING_URI")
    if "MLFLOW_TRACKING_URI" in os.environ:
        del os.environ["MLFLOW_TRACKING_URI"]
    
    test_settings = Settings()
    
    if original is not None:
        os.environ["MLFLOW_TRACKING_URI"] = original
        
    assert "mlflow_server:5000" in test_settings.mlflow_tracking_uri
def test_quality_gate_logic():
    from training.retrain_pipeline import quality_gate
    from unittest.mock import patch, MagicMock
    
    # Mock MLflow client
    mock_client = MagicMock()
    
    with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
        # Helper to setup mock for champion metrics
        def setup_champion_mock(tfidf_prec, als_prec):
            def get_model_version_by_alias_side_effect(name, alias):
                mock_version = MagicMock()
                mock_version.run_id = f"run_{name}"
                return mock_version
                
            def get_run_side_effect(run_id):
                mock_run = MagicMock()
                if "TFIDF" in run_id:
                    mock_run.data.metrics = {"precision_at_5": tfidf_prec}
                else:
                    mock_run.data.metrics = {"precision_at_5": als_prec}
                return mock_run
                
            mock_client.get_model_version_by_alias.side_effect = get_model_version_by_alias_side_effect
            mock_client.get_run.side_effect = get_run_side_effect

        # Case A: Good TF-IDF + Good ALS
        setup_champion_mock(0.1000, 0.1000)
        passed, msg = quality_gate({"precision_at_5": 0.0960}, {"precision_at_5": 0.0960})
        assert passed is True

        # Case B: Good TF-IDF + Bad ALS
        setup_champion_mock(0.1000, 0.1000)
        passed, msg = quality_gate({"precision_at_5": 0.0960}, {"precision_at_5": 0.0900}) # 0.0900 < 0.0950
        assert passed is False

        # Case C: Bad TF-IDF + Good ALS
        setup_champion_mock(0.1000, 0.1000)
        passed, msg = quality_gate({"precision_at_5": 0.0900}, {"precision_at_5": 0.0960})
        assert passed is False

        # Case D: Valid but < 0.01 (too low absolute performance)
        setup_champion_mock(0.0050, 0.0050)
        passed, msg = quality_gate({"precision_at_5": 0.0060}, {"precision_at_5": 0.0060})
        assert passed is False
        
        # Case E: Production comparison unavailable (first run)
        mock_client.get_model_version_by_alias.side_effect = Exception("Not found")
        passed, msg = quality_gate({"precision_at_5": 0.1000}, {"precision_at_5": 0.1000})
        assert passed is True
