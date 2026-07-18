"""Gom toàn bộ route con. Thêm feature mới = thêm module trong routes/ rồi include ở đây."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import anime, health, map as map_route, posters, recommend, users

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(recommend.router, tags=["recommend"])
api_router.include_router(posters.router, tags=["posters"])
api_router.include_router(anime.router, tags=["anime"])
api_router.include_router(users.router, tags=["users"])
api_router.include_router(map_route.router, tags=["map"])
