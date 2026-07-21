"""Schemas cho POST /api/posters — batch poster + score/members MỚI theo mal_id (MAL v2).
Xem service/API_CONTRACT.md."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PostersRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=300)


class PosterEntry(BaseModel):
    poster: Optional[str] = None    # URL ảnh (null nếu không lấy được)
    score: Optional[float] = None   # MAL mean score mới nhất (null nếu chưa có điểm)
    members: Optional[int] = None   # num_list_users — số user có anime trong list


class PostersResponse(BaseModel):
    posters: Dict[int, PosterEntry]  # mal_id -> {poster, score, members}
