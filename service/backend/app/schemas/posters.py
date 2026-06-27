"""Schemas cho POST /api/posters — batch poster URL theo mal_id. Xem service/API_CONTRACT.md."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PostersRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=300)


class PostersResponse(BaseModel):
    posters: Dict[int, Optional[str]]  # mal_id -> URL ảnh (null nếu không lấy được)
