"""Gom toàn bộ route con. Thêm feature mới = thêm module trong routes/ rồi include ở đây."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import health, recommend

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(recommend.router, tags=["recommend"])
