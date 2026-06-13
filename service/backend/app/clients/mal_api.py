"""
MyAnimeList data fetchers for the recommender service (test flow only — no model/UI).

Two data needs:
  1. Anime metadata by mal_id           -> MAL API v2  (X-MAL-CLIENT-ID header)
  2. User history + demographics + favs:
       - scored interaction history     -> MAL API v2  /users/{user}/animelist (paginated)
       - demographics + stats + favs    -> Jikan API v4 /users/{user}/full

Why Jikan for the profile: MAL API v2 only exposes the *authenticated* user's profile
(gender/birthday/location), not arbitrary users, and it has no favorites endpoint.
Jikan reads public MAL profiles, so demographics + favorites come from there.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import requests

# --- config -----------------------------------------------------------------
ENV_PATH = Path(__file__).resolve().parents[3] / ".env"  # service/.env (app/clients/ → service/)
MAL_BASE = "https://api.myanimelist.net/v2"
JIKAN_BASE = "https://api.jikan.moe/v4"
TIMEOUT = 30

# Full anime field set (MAL v2). Sub-objects must be requested by name.
ANIME_FIELDS = (
    "id,title,main_picture,alternative_titles,start_date,end_date,synopsis,"
    "mean,rank,popularity,num_list_users,num_scoring_users,nsfw,created_at,"
    "updated_at,media_type,status,genres,num_episodes,start_season,broadcast,"
    "source,average_episode_duration,rating,studios,pictures,background,"
    "related_anime,related_manga,recommendations,statistics"
)

# list_status carries the user<->anime interaction (the training signal).
ANIMELIST_FIELDS = (
    "list_status{status,score,num_episodes_watched,is_rewatching,"
    "updated_at,start_date,finish_date}"
)


def _load_client_id() -> str:
    cid = os.environ.get("MAL_CLIENT_ID")
    if not cid and ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("MAL_CLIENT_ID") and "=" in line:
                cid = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not cid:
        raise RuntimeError(f"MAL_CLIENT_ID not found in {ENV_PATH} or environment.")
    return cid


MAL_HEADERS = {"X-MAL-CLIENT-ID": _load_client_id()}


def _get(url, *, headers=None, params=None, retries=2):
    """GET -> parsed JSON, with a small backoff retry on 429 (rate limit)."""
    for attempt in range(retries + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 429 and attempt < retries:
            time.sleep(2 * (attempt + 1))
            continue
        print(f"[-] {resp.status_code} {url} -> {resp.text[:200]}")
        return None
    return None


# --- 1. anime metadata ------------------------------------------------------
def get_anime_metadata(anime_id, fields=ANIME_FIELDS):
    return _get(
        f"{MAL_BASE}/anime/{anime_id}",
        headers=MAL_HEADERS,
        params={"fields": fields},
    )


# --- 2a. user scored history (MAL v2, paginated) ----------------------------
def get_user_anime_list(username, fields=ANIMELIST_FIELDS, limit=1000):
    url = f"{MAL_BASE}/users/{username}/animelist"
    params = {"limit": limit, "fields": fields, "nsfw": "true"}
    out = []
    page = 1
    while url:
        print(f"[+] animelist '{username}' page {page} ...")
        data = _get(url, headers=MAL_HEADERS, params=params)
        if data is None:
            break
        out.extend(data.get("data", []))
        url = data.get("paging", {}).get("next")  # full URL incl. query, or None
        params = {}  # paging.next already carries the query string
        page += 1
        time.sleep(0.5)
    return out


# --- 2b. demographics + statistics + favorites (Jikan v4) -------------------
def get_user_profile(username):
    """Jikan /users/{user}/full -> demographics + statistics + favorites in one call."""
    data = _get(f"{JIKAN_BASE}/users/{username}/full")
    return data.get("data") if data else None
