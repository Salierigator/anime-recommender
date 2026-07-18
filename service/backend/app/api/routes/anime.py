"""GET /api/anime/{mal_id} — chi tiết cho modal (MAL v2 + cache, thay Jikan client-side).
GET /api/search?q= — autocomplete title cho guest picker. Xem service/API_CONTRACT.md."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from app.dependencies import get_service
from app.schemas.anime import AnimeDetails, SearchResponse
from app.services.base import RecommenderService
from app.services.details import fetch_details

router = APIRouter()


@router.get("/anime/{mal_id}", response_model=AnimeDetails)
def anime_details(mal_id: int, response: Response) -> AnimeDetails:
    detail = fetch_details(mal_id)
    if detail is None:
        raise HTTPException(status_code=404,
                            detail=f"can't fetch details for anime {mal_id}")
    response.headers["Cache-Control"] = "public, max-age=86400"
    return AnimeDetails(**detail)


@router.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=2, max_length=100),
           limit: int = Query(default=10, ge=1, le=25),
           service: RecommenderService = Depends(get_service)) -> SearchResponse:
    return SearchResponse(results=service.search(q, limit))
