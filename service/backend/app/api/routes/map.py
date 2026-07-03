"""GET /api/map + /api/map/territory.png — bản đồ anime (points/clusters/meta + nền PNG).

Payload tĩnh per-deploy: service serialize sẵn bytes (mock đọc fixture, real đọc map/outputs/service
qua AnimeMap) — trả thẳng, không qua pydantic (21k điểm, khỏi re-validate mỗi request).
Shape: service/API_CONTRACT.md. "You are here" KHÔNG ở đây — nằm ở meta.map_xy của /api/recommend.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response

from app.dependencies import get_service
from app.services.base import RecommenderService

router = APIRouter()


@router.get("/map")
def get_map(service: RecommenderService = Depends(get_service)) -> Response:
    return Response(content=service.map_payload(), media_type="application/json",
                    headers={"Cache-Control": "public, max-age=3600"})


@router.get("/map/territory.png")
def territory(service: RecommenderService = Depends(get_service)) -> FileResponse:
    p = service.territory_path()
    if p is None or not p.exists():
        raise HTTPException(status_code=404,
                            detail="territory.png chưa có (chạy map/export_service.py)")
    return FileResponse(p, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=3600"})
