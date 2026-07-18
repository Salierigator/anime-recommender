"""GET /api/users/{username}/exists — check nhanh username MAL cho hint valid/invalid trên UI.

Vì sao backend: Jikan /users/ đang 504 diện rộng (17/07) mà frontend cần câu trả lời chắc;
MAL v2 cần client id nên phải đi qua server. Volume thấp (debounce phía client) + cache RAM.
Không xác định được (network/5xx) → 502, frontend im lặng (đừng báo invalid oan).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()

_cache: dict = {}                      # username đã normalize -> bool (chỉ cache câu trả lời chắc)


@router.get("/users/{username}/exists")
def username_exists(username: str) -> dict:
    key = username.strip().lower()
    if key in _cache:
        return {"exists": _cache[key]}
    try:
        from app.clients.mal_api import user_exists  # lazy: cần MAL_CLIENT_ID
        result = user_exists(username.strip())
    except RuntimeError:
        result = None                                # thiếu MAL_CLIENT_ID
    if result is None:
        raise HTTPException(status_code=502, detail="can't verify username")
    if len(_cache) > 1024:
        _cache.clear()
    _cache[key] = result
    return {"exists": result}
