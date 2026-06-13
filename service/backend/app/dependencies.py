"""FastAPI dependencies — service được tạo 1 lần trong lifespan, lưu ở app.state."""
from __future__ import annotations

from fastapi import Request

from app.services.base import RecommenderService


def get_service(request: Request) -> RecommenderService:
    return request.app.state.service
