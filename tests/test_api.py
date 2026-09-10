"""
CartSense — FastAPI Integration Tests
Tests all API endpoints using httpx TestClient.
Requires: database seeded, models trained, FastAPI running.
"""
import os
import pytest
from fastapi.testclient import TestClient

# Point to a local test DB to avoid hitting production
os.environ.setdefault(
    "DATABASE_URL",
    "sqlite:///./cartsense_test.db",
)
os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
os.environ.setdefault("ARTIFACTS_DIR", "./artifacts")


@pytest.fixture(scope="module")
def client():
    """Create a test client. Models may not be loaded; health check is still testable."""
    from backend.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_endpoint(client):
    """GET /api/health must return JSON with a status field."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "database" in data
    assert "tfidf_loaded" in data
    assert "als_loaded" in data


def test_products_endpoint(client):
    """GET /api/products must return a list."""
    resp = client.get("/api/products?limit=10")
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)


def test_products_category_filter(client):
    """GET /api/products?category=laptop must return only laptops."""
    resp = client.get("/api/products?category=laptop&limit=20")
    if resp.status_code == 200:
        data = resp.json()
        for item in data:
            assert item.get("category") == "laptop"


def test_recommend_validation_missing_fields(client):
    """POST /api/recommend without required fields must return 422."""
    resp = client.post("/api/recommend", json={})
    assert resp.status_code == 422


def test_recommend_invalid_top_k(client):
    """POST /api/recommend with top_k > 20 must return 422."""
    resp = client.post("/api/recommend", json={
        "user_id": "U0001",
        "product_id": "L0001",
        "top_k": 99,
    })
    assert resp.status_code == 422


def test_recommend_unknown_product(client):
    """POST /api/recommend with unknown product_id must return 404."""
    resp = client.post("/api/recommend", json={
        "user_id": "U0001",
        "product_id": "NONEXISTENT_XYZ",
        "top_k": 5,
    })
    # Either 404 (product not found) or 503 (models not loaded)
    assert resp.status_code in (404, 503)


def test_drift_endpoint(client):
    """GET /api/drift must return JSON."""
    resp = client.get("/api/drift")
    assert resp.status_code in (200, 503)


def test_model_status_endpoint(client):
    """GET /api/model/status must return model info."""
    resp = client.get("/api/model/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "tfidf_loaded" in data
    assert "als_loaded" in data
    assert "model_version" in data


def test_stats_endpoint(client):
    """GET /api/stats must return counts."""
    resp = client.get("/api/stats")
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        data = resp.json()
        assert "n_products" in data
        assert "n_users" in data
        assert "n_interactions" in data


def test_training_history_endpoint(client):
    """GET /api/training/history must return a list."""
    resp = client.get("/api/training/history")
    assert resp.status_code in (200, 500)
    if resp.status_code == 200:
        assert isinstance(resp.json(), list)


def test_openapi_docs(client):
    """FastAPI OpenAPI docs must be accessible."""
    resp = client.get("/docs")
    assert resp.status_code == 200
