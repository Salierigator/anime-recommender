"""API contract ở mock mode (MOCK_MODE=1) — không load model. Xem service/API_CONTRACT.md."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_mock():
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "mock"


def test_recommend_mock_shape_and_cut():
    with TestClient(app) as client:
        r = client.post("/api/recommend",
                        json={"username": "anyone", "top_k": 5, "cold_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["main"]) <= 5
    assert len(body["cold"]) <= 3
    assert body["meta"]["mode"] == "mock"


def test_recommend_requires_user_or_ids():
    with TestClient(app) as client:
        r = client.post("/api/recommend", json={"top_k": 5})
    assert r.status_code == 422
