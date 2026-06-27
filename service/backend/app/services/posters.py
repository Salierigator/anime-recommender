"""Lấy URL poster từ MAL v2, fetch song song + cache file. TÁCH khỏi path /recommend
để engine recommend giữ nguyên tốc độ — posters không cần model, chỉ cần MAL client id.

Thiếu client id / lỗi fetch -> None (frontend tự fallback sang Jikan qua <img onError>).
Chỉ cache URL lấy được; miss không cache để lần sau thử lại.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "posters.json"
_LOCK = Lock()


def _load_cache() -> Dict[int, str]:
    if CACHE_PATH.exists():
        try:
            return {int(k): v for k, v in json.loads(CACHE_PATH.read_text()).items()}
        except (json.JSONDecodeError, ValueError, OSError):
            return {}
    return {}


def _save_cache(cache: Dict[int, str]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def fetch_posters(ids: List[int]) -> Dict[int, Optional[str]]:
    cache = _load_cache()
    missing = [i for i in dict.fromkeys(ids) if i not in cache]

    if missing:
        try:
            from app.clients.mal_api import get_main_picture  # lazy: cần MAL_CLIENT_ID
            with ThreadPoolExecutor(max_workers=8) as ex:
                fetched = dict(zip(missing, ex.map(get_main_picture, missing)))
        except RuntimeError:
            fetched = {}  # không có MAL_CLIENT_ID -> để frontend fallback Jikan
        with _LOCK:
            cache = _load_cache()
            cache.update({k: v for k, v in fetched.items() if v})
            _save_cache(cache)

    return {i: cache.get(i) for i in ids}
