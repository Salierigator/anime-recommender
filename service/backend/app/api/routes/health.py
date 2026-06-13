"""GET /api/health — trạng thái server + chế độ (mock/live)."""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_settings

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "mode": "mock" if settings.mock_mode else "live",
        "model_loaded": getattr(request.app.state.service, "model_loaded", False),
    }
