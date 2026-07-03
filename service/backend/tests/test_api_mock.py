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


def test_recommend_mock_has_map_xy():
    with TestClient(app) as client:
        r = client.post("/api/recommend", json={"username": "anyone"})
    assert r.status_code == 200
    assert len(r.json()["meta"]["map_xy"]) == 2       # fixture có sẵn [x, y]


def test_map_mock_shape():
    with TestClient(app) as client:
        r = client.get("/api/map")
    assert r.status_code == 200
    body = r.json()
    pts = body["points"]
    n = len(pts["mal_id"])
    assert n > 0
    assert all(len(pts[k]) == n
               for k in ("title", "x", "y", "label", "popularity", "is_cold"))
    assert body["clusters"]
    assert {"label", "name", "size", "examples", "cx", "cy"} <= set(body["clusters"][0])
    assert len(body["meta"]["extent"]) == 4


def test_map_territory_route_wired():
    # mock mượn artifacts/map/territory.png nếu repo đã export; clone frontend-only -> 404.
    with TestClient(app) as client:
        r = client.get("/api/map/territory.png")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers["content-type"] == "image/png"
