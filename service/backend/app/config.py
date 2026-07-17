"""Settings tập trung (pydantic-settings) — đọc từ env / service/.env.

Bật mock: `MOCK_MODE=1` (mặc định True ở giai đoạn skeleton để frontend dev không cần model).
Real mode (`MOCK_MODE=0`): lifespan load `Recommender` thật — wiring build sau.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py = service/backend/app/config.py
BACKEND = Path(__file__).resolve().parents[1]             # service/backend/
ROOT = Path(__file__).resolve().parents[3]                # anime recommender/ (repo root)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / "service" / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    mock_mode: bool = True
    cors_origins: List[str] = ["http://localhost:5173"]   # Vite dev server
    map_enabled: bool = False                             # map/ đóng băng — real mode bỏ qua AnimeMap
    default_top_k: int = 20
    default_cold_k: int = 10
    fixtures_dir: Path = BACKEND / "fixtures"
    mal_client_id: Optional[str] = None                   # từ service/.env (MAL_CLIENT_ID)


@lru_cache
def get_settings() -> Settings:
    return Settings()
