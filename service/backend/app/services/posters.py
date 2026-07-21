"""Lấy poster + score/members MỚI từ MAL v2, fetch song song + cache file. TÁCH khỏi path
/recommend để engine recommend giữ nguyên tốc độ — chỉ cần MAL client id.

Card frontend dùng batch này để hiển thị số fresh (khớp MAL) thay vì gọi Jikan client-side
(Jikan 3 req/s + hay 504). Thiếu client id / lỗi fetch -> poster None (frontend fallback Jikan
qua <img onError>); score/members None -> card dùng snapshot backend.
Chỉ cache entry lấy được; miss không cache để lần sau thử lại.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "posters.json"
_LOCK = Lock()

# Fields MAL v2 cần cho card: ảnh + điểm + số members.
_FIELDS = "main_picture,mean,num_list_users"


def _load_cache() -> Dict[int, dict]:
    if CACHE_PATH.exists():
        try:
            raw = json.loads(CACHE_PATH.read_text())
        except (json.JSONDecodeError, ValueError, OSError):
            return {}
        # Chỉ nhận entry format mới (dict). Format cũ (str url) bị bỏ → refetch kèm score/members.
        return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
    return {}


def _save_cache(cache: Dict[int, dict]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def _entry(data: dict) -> dict:
    pic = data.get("main_picture") or {}
    return {
        "poster": pic.get("large") or pic.get("medium"),
        "score": data.get("mean"),
        "members": data.get("num_list_users"),
    }


def fetch_posters(ids: List[int]) -> Dict[int, dict]:
    cache = _load_cache()
    missing = [i for i in dict.fromkeys(ids) if i not in cache]

    if missing:
        try:
            from app.clients.mal_api import get_anime_metadata  # lazy: cần MAL_CLIENT_ID
        except RuntimeError:
            get_anime_metadata = None  # không có MAL_CLIENT_ID -> để frontend fallback

        if get_anime_metadata is not None:
            def one(mal_id: int) -> Optional[dict]:
                data = get_anime_metadata(mal_id, fields=_FIELDS)
                return _entry(data) if data else None

            with ThreadPoolExecutor(max_workers=8) as ex:
                fetched = dict(zip(missing, ex.map(one, missing)))
            with _LOCK:
                cache = _load_cache()
                cache.update({k: v for k, v in fetched.items() if v})
                _save_cache(cache)

    empty = {"poster": None, "score": None, "members": None}
    return {i: cache.get(i, empty) for i in ids}
