import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_api_health(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "DebtOx"
    assert data["loaded_models_count"] >= 0

def test_api_version(client):
    res = client.get("/api/v1/version")
    assert res.status_code == 200
    assert res.json()["version"] == "1.0.0"

def test_api_models(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

def test_api_analyses_list(client):
    res = client.get("/api/v1/analyses")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
