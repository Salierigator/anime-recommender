"""FastAPI app — entrypoint. Chạy: `uvicorn app.main:app --reload` (CWD = service/backend).

Lifespan tạo service 1 lần:
  - mock_mode (mặc định): MockService đọc fixtures, KHÔNG load model.
  - real mode (MOCK_MODE=0): RealService load Recommender (wiring build sau).
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.mock_mode:
        from app.services.mock_service import MockService
        app.state.service = MockService(settings)
    else:
        from app.services.real_service import RealService
        app.state.service = RealService(settings)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Anime Recommender API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix="/api")
    return app


app = create_app()
