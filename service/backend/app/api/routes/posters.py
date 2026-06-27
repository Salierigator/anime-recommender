"""POST /api/posters — {ids:[...]} -> {posters: {mal_id: url|null}}.

Tách khỏi /recommend để engine recommend giữ tốc độ. Backend fetch MAL v2 (song song +
cache); frontend fallback Jikan khi miss. Xem service/API_CONTRACT.md.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.posters import PostersRequest, PostersResponse
from app.services.posters import fetch_posters

router = APIRouter()


@router.post("/posters", response_model=PostersResponse)
def posters(req: PostersRequest) -> PostersResponse:
    return PostersResponse(posters=fetch_posters(req.ids))
