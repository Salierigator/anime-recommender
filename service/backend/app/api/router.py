"""Gom toàn bộ route con. Thêm feature mới = thêm module trong routes/ rồi include ở đây."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, map as map_route, posters, recommend

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(recommend.router, tags=["recommend"])
api_router.include_router(posters.router, tags=["posters"])
api_router.include_router(map_route.router, tags=["map"])
